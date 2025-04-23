from utils import *

"""
Known issues: trajectory not updated properly if gromacs encounters an error
and needs to overlap some frames from a cpt file.
Current workaround solution: detect error and recreate pathensemble object 
manually before restarting simulation.
"""

# specific arguments
parser = argparse.ArgumentParser(description='AIMMD equilibrium')
parser.add_argument('directory', type=str,
    help='project directory, would simulate in <directory>/equilibrium<state>')
parser.add_argument('worker_id', type=int)
parser.add_argument('total_workers', type=int)
parser.add_argument('target_state', type=str)
parser.add_argument('-s', '--slurm', action='store_true')
args = parser.parse_args()
directory = args.directory
worker_id = args.worker_id
total_workers = args.total_workers
slurm = args.slurm
target_state = args.target_state
target_states = sorted([target_state, 'R'])

# change directory / load params
os.chdir(directory)
exec(open('params.py', 'r').read())
os.chdir(f'equilibrium{target_state}')

# prepare
topology = '../../run.gro'
if not gromacs:
    sys.path.append('../..')
    from integrator import run

# auxiliary functions
def condition_stop():
    """
    Also updates trajectory.
    """
    global trajectory, initial_frames
    
    if not all_jobs_are_running('../jobids.txt',
        slurm=slurm, cancel_jobs_if_false=True):
        sleep(5.)
        raise
    
    # update trajectory
    try:
        l0 = trajectory.nframes + 0 
        fname = trajectory.trajectory_files[-1][:10]
        part = int(trajectory.trajectory_files[-1][15:19])
        filename = f'{fname}.part{part:04g}{trajectory_extension}'
        filenames = os.listdir()
        while filename in filenames:
            trajectory.append(filename)
            part += 1
            filename = f'{fname}.part{part:04g}{trajectory_extension}'
        if trajectory.nframes > l0:
            trajectory.remove_overlapping_frames()
            trajectory.split(max_excursion_length)
            trajectory.save(f'{fname}.h5')
        
        if trajectory.nframes <= 2:
            return False
    except:
        pass
    
    # stop condition (gives initial frames to restart
    if len(k := np.where(~trajectory.are_accepted)[0]):
        initial_frames = trajectory.frame_indices(k[0])[0][[1, 0]]
        return True
    if len(k := np.where(trajectory.are_transitions)[0]):
        initial_frames = trajectory.frame_indices(k[0])[0][[1, 0]]
        return True
    return False


def simulate():
    global fname
    stop_event.clear()
    
    # part 1: recover (get the last part)
    part = 0
    filenames = os.listdir() 
    while f'{fname}.part{part + 1:04g}{trajectory_extension}' in filenames:
        part += 1
    filename = f'{fname}.part{part:04g}{trajectory_extension}'
    
    # is the last part recoverable?
    os.rename(filename, f'_{filename}')
    try:
        universe = mda.Universe(topology, f'_{filename}')
    except:
        remove(f'_{filename}')
        return simulate()
    
    # recover the last part
    atomgroup = universe.atoms
    frames = universe.trajectory
    writer = mda.Writer(filename, atomgroup.n_atoms)
    for nframes, frame in enumerate(frames):
        writer.write(atomgroup)
        x, y = frame._pos[0, :2] / 10.  # otw reset trajectory
        t = frame.time + 0
    remove(f'_{filename}')
    write(f'picking {filename} with {nframes + 1} frames')
    t0 = t - nframes
    
    while True:
        nframes += 1  # update count (now frame number)
        if stop_event.is_set():
            writer.close()
            break
        if nframes >= max_length:  # new part?
            write(f'{filename} hit {nframes} frames')
            writer.close()
            nframes = 0
            t0 = t + 1  # advancing to next time
            part += 1
            filename = f'{fname}.part{part:04g}{trajectory_extension}'
            write(f'    advancing to {filename}')
            writer = mda.Writer(filename, atomgroup.n_atoms)
        sleep(slowdown)  # simulate
        x, y = run(x, y, timestep)
        frame._pos[0, :2] = x * 10., y * 10.  # update and write frame
        t = t0 + nframes
        frame.time = t
        writer.write(atomgroup)


# wait
wait_flag('../proceed.txt', '../jobids.txt', slurm=slurm)

# simplest algorithm possible
n = worker_id + 1
while True:
    
    # step 1: wait for first part and get trajectory
    filenames = os.listdir()
    while True:
        fname = f'traj{n:06g}'  # current trajectory
        new_fname = f'traj{n + total_workers:06g}'  # current trajectory
        if f'{new_fname}.part0000{trajectory_extension}' not in filenames:
            break
        n += total_workers
    
    # ensure that everything is running fine
    os.system(f'rm .{fname}*')  # 'cause MDAnalysis is a ***
    sleep(1.)
    if not all_jobs_are_running('../jobids.txt',
        slurm=slurm, cancel_jobs_if_false=True):
        raise
    mda.Universe(topology, 
        f'{fname}.part0000{trajectory_extension}').trajectory[0]
    
    write(f'Now {fname}')
    trajectory = PathEnsemble(
        '.', topology, states_function, descriptors_function)
    trajectory.trajectory_files.append(
        f'{fname}.part0000{trajectory_extension}')
    if f'{fname}.h5' in os.listdir():
        try:
            trajectory.load(f'{fname}.h5')
        except:
            trajectory.load(f'{fname}_backup.h5')
        write(f'reloaded trajectory: {trajectory}')
    else:
        write(f' empty trajectory: {trajectory}')
    initial_frames = None
    
    # step 2: initialize simulation
    if gromacs:
        filename = f'{fname}.part0000{trajectory_extension}'
        if f'{fname}.tpr' not in os.listdir():
            
            write('initialization')
            if trajectory_extension == '.trr':
                command = (f'{grompp} -f ../../run.mdp -r {topology} '
                           f'-c {topology} -t {filename} -o {fname}.tpr')
                run_command(command)
            else:
                # randomize initial velocities
                md.load(filename, top='../../run.gro')[-1].save('temp.gro')
                write('initial velocities randomization')
                command = (f'{grompp} -f ../../randomvelocities.mdp '
                           f'-r temp.gro -c temp.gro -o temp.tpr')
                run_command(command)
                command = f'{mdrun} -deffnm temp -nsteps 0'
                run_command(command)
                command = (f'{grompp} -f ../../run.mdp -r {topology} '
                           f'-c {topology} -t temp.trr -o {fname}.tpr')
                run_command(command)
            
            # remove last frame from part 0
            os.rename(filename, f'_{filename}')
            universe = mda.Universe(topology, f'_{filename}')
            atomgroup = universe.atoms
            frame = universe.trajectory[0]
            with mda.Writer(filename, atomgroup.n_atoms) as writer:
                writer.write(atomgroup)
            remove(f'_{filename}')
    
    # step 3: simulate other parts
    write('simulate')
    if gromacs:
        command = (f'while {mdrun} -deffnm {fname} -cpo {fname}.cpt '
                   f'-cpi {fname}.cpt -noappend; do : sleep 1; done')
        run_command(command, condition_stop)
    else:
        run_task(simulate, condition_stop)
    
    # step 4: advance to next trajectory (create first part)
    write('advance to next trajectory')
    write(f'    create {new_fname}.part0000{trajectory_extension}')
    if initial_frames is None:
        write(f'    attention! copying {fname}.part0000{trajectory_extension}')
        shutil.copy(f'{fname}.part0000{trajectory_extension}', 
                    f'{new_fname}.part0000{trajectory_extension}')
    else:
        write(f'    using {initial_frames} initial frames')
        trajectory.frames(initial_frames).write(
        f'{new_fname}.part0000{trajectory_extension}',
            invert_velocities=True, reset_time=True)
