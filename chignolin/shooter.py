from utils import *

"""
Adapted with retinal in mind: force SP kinetic energy without
randomizing the velocities to avoid instabilities (heating up).
"""

# specific arguments
parser = argparse.ArgumentParser(description='AIMMD shooter')
parser.add_argument('directory', type=str,
    help='project directory, would simulate in <directory>/shoot')
parser.add_argument('worker_id', type=int)
parser.add_argument('-s', '--slurm', action='store_true')
args = parser.parse_args()
directory = args.directory
worker_id = args.worker_id
slurm = args.slurm

# change directory / load params
os.chdir(directory)
exec(open('params.py', 'r').read())
if 'randomize_shooting_velocities' not in globals():
    randomize_shooting_velocities = trajectory_extension == '.xtc'
os.chdir(f'shots{worker_id}')

# prepare
topology = '../../run.gro'
if not gromacs:
    sys.path.append('../..')
    from integrator import run

# auxiliary functions
def stop(frame):
    global nframes, base
    state = states_function([frame])[0]
    if state != 'R':
        return True
    if nframes + base >= max_excursion_length:
        return True
    return False

class dummy_xdr:  # avoid stupid traceback
    def __init__(self):
        pass
    def close(self):
        pass

def condition_stop():
    outcome = False
    global fname, nframes, base
    
    if not all_jobs_are_running('../jobids.txt',
        slurm=slurm, cancel_jobs_if_false=True):
        sleep(5.)
        raise
    
    try:
        if f'{fname}{trajectory_extension}' not in os.listdir():
            raise
        trajectory = mda.Universe(topology,
            f'{fname}{trajectory_extension}').trajectory
        while True:
            frame = trajectory._read_frame_with_aux(nframes - 1)
            if stop(frame):
                outcome = True
                break
            nframes += 1
    except:
        pass
    
    if 'trajectory' in locals():
        if not hasattr(trajectory, '_xdr'):
            trajectory._xdr = dummy_xdr()
        del trajectory
    return outcome

def simulate_and_check(fname):
    global nframes, base
    filename = f'{fname}{trajectory_extension}'
    
    # is the last part recoverable?
    try:
        os.rename(filename, f'_{filename}')
        universe = mda.Universe(topology, f'_{filename}')
    except:
        remove(f'_{filename}')
        md.load(f'shoot.trr', top='../../run.gro').save(filename)
        return simulate_and_check(fname)
    
    # recover the last part
    atomgroup = universe.atoms
    frames = universe.trajectory
    writer = mda.Writer(filename, atomgroup.n_atoms)
    for nframes, frame in enumerate(frames):
        frame.time = float(nframes)
        writer.write(atomgroup)
        x, y = frame._pos[0, :2] / 10.  # otw reset trajectory
    remove(f'_{filename}')
    write(f'picking {filename} with {nframes + 1} frames')
    frame._pos[0, :2] = x * 10., y * 10.  # otw reset the trajectory
    
    if not all_jobs_are_running('../jobids.txt',
        slurm=slurm, cancel_jobs_if_false=True):
        sleep(5.)
        raise
    
    # part 2: append
    while True:
        nframes += 1  # update count (now frame number)
        if stop(frame):
            writer.close()
            break
        sleep(slowdown)  # simulate
        x, y = run(x, y, timestep)
        frame._pos[0, :2] = x * 10., y * 10.  # update and write frame
        frame.time = float(nframes)
        writer.write(atomgroup)


# get shooting chain and check for errors
chain = PathEnsemble('.', topology, states_function, descriptors_function)
filenames = sorted([filename for filename in os.listdir()
                    if filename[:4] == 'path' and
                    filename[4:10].isdigit() and
                    filename[10:14] == f'{trajectory_extension}'])
if len(filenames):
    try:
        chain.load('chain.h5')
        assert len(chain) == len(filenames)
        write(f'reloaded shooting chain: {chain}')
    except:
        write(f'chain.h5 corrupt: reloaded chain_backup.h5 '
              f'and adding missing paths')
        try:
            chain.load('chain_backup.h5')
        except:
            pass
        for filename in filenames:
            if filename not in chain.trajectory_files:
                write(f'***** {filename} *****')
                if not chain.add_path(filename)[0]:
                    if filename == filenames[-1]:
                        remove(filename)  # just remove the last
                        os.system(f'rm .{filename}*')
                    else:
                        raise  # no frames added

# wait
wait_flag('../proceed.txt', '../jobids.txt', slurm=slurm)

# simplest algorithm possible
while True:
    
    filename = f'path{len(chain) + 1:06g}'
    write(f'\nnow waiting for shooting point for {filename}')
    
    # step 1: wait for shooter
    while True:
        if not all_jobs_are_running('../jobids.txt',
            slurm=slurm, cancel_jobs_if_false=True):
            sleep(5.)
            raise
        try:
            md.load('shoot.trr', top=topology)
            break
        except:
            continue
    
    # step 2: initialize simulations
    if gromacs:
        filenames = os.listdir()
        
        # create simulation files
        if ((f'back.tpr' not in filenames)
        or  (f'forw.tpr' not in filenames)
        or  (f'back.tpr' in filenames and not os.path.getsize('back.tpr'))
        or  (f'forw.tpr' in filenames and not os.path.getsize('forw.tpr'))):
            # randomize shooting point velocities
            command = (f'{grompp} -f ../../randomvelocities.mdp '
                       f'-r {topology} -c {topology} '
                       f'-t shoot.trr -o temp.tpr')
            run_command(command)
            command = f'{mdrun} -deffnm temp -nsteps 0'
            run_command(command)
            
            if randomize_shooting_velocities:
                shutil.copy('temp.trr', 'shoot.trr')
            else:  # target kinetic energy
                universe = mda.Universe(topology, 'temp.trr')
                frame = universe.trajectory[0]
                k = np.sum(frame._velocities ** 2)
                
                # load shooting frame and correct for velocities
                shutil.copy('shoot.trr', '_shoot.trr')
                universe = mda.Universe(topology, '_shoot.trr')
                atomgroup = universe.atoms
                frame = universe.trajectory[0]
                k2 = np.sum(frame._velocities ** 2)
                #write(f'>> velocities correction {np.sqrt(k2 / k)}')
                #frame._velocities /= np.sqrt(k2 / k)
                with mda.Writer('shoot.trr', atomgroup.n_atoms) as writer:
                    writer.write(atomgroup)
            
            # backward
            command = (f'{grompp} -f ../../run.mdp -r {topology} '
                       f'-c {topology} -t shoot.trr -o back.tpr')
            run_command(command)
            
            # forward (invert backward's velocities)
            shutil.copy('shoot.trr', '_shoot.trr')
            universe = mda.Universe(topology, '_shoot.trr')
            atomgroup = universe.atoms
            frame = universe.trajectory[0]
            write(f'>> SP kinetic energy: {np.sum(frame._velocities ** 2):0e}')
            frame._velocities *= -1
            with mda.Writer('shoot.trr', atomgroup.n_atoms) as writer:
                writer.write(atomgroup)
            command = (f'{grompp} -f ../../run.mdp -r {topology} '
                       f'-c {topology} -t shoot.trr -o forw.tpr')
            run_command(command)
    
    # step 3: backward simulation
    write('backward simulation')
    fname = 'back'
    base = 0
    nframes = 1
    if gromacs:
        command = (f'while {mdrun} -deffnm {fname} -cpo {fname}.cpt '
                   f'-cpi {fname}.cpt; do :; done')
        run_command(command, condition_stop)
    else:
        simulate_and_check(fname)
    nframes_back = nframes + 0
    
    # step 4: forward simulation
    write('forward simulation')
    fname = 'forw'
    base = nframes_back
    nframes = 1
    if gromacs:
        command = (f'while {mdrun} -deffnm {fname} -cpo {fname}.cpt '
                   f'-cpi {fname}.cpt; do : sleep 1; done')
        run_command(command, condition_stop)
    else:
        simulate_and_check(fname)
    nframes_forw = nframes + 0
    
    # step 5: writing result: invert and join the two
    write(f'inverting and joining the two into {filename}{trajectory_extension}')
    universe1 = mda.Universe(topology, f'back{trajectory_extension}')
    atomgroup1 = universe1.atoms
    trajectory1 = universe1.trajectory[nframes_back - 1:0:-1]
    universe2 = mda.Universe(topology, f'forw{trajectory_extension}')
    atomgroup2 = universe2.atoms
    trajectory2 = universe2.trajectory[:nframes_forw]
    with mda.Writer(f'{filename}{trajectory_extension}',
                    atomgroup1.n_atoms) as writer:
        for frame in trajectory1:
            frame._velocities *= -1  # if present
            writer.write(atomgroup1)
        for frame in trajectory2:
            writer.write(atomgroup2)
    
    # step 5b: also merge energy (workaround solution)
    try:
        os.system(f'{eneconv} -f back.edr forw.edr -o result.edr')
        os.rename('result.edr', f'{filename}.edr')
    except:
        pass
    
    # step 6: update chain
    write('update chain')
    try:
        selection_bias = np.load('shoot_bias.npy')
    except:
        selection_bias = 1.
    chain.add_path(f'{filename}{trajectory_extension}',
                   selection_bias=selection_bias,
                   weight=0.)
    if ((chain.lengths[-1] > max_excursion_length) or
        (chain.initial_states[-1] == 'R') or
        (chain.final_states[-1] == 'R')):
        chain.are_accepted[-1] = False  # ensure no bad paths are added
    
    write(f'***** {filename} *****')
    chain.save('chain.h5')
    
    # step 7: cleanup and reset
    write('cleanup and reset')
    for fname in os.listdir():
        if 'back' in fname and 'backup' not in fname:
            remove(fname)
        if 'forw' in fname:
            remove(fname)
        if 'shoot' in fname:
            remove(fname)
        if 'temp' in fname:
            remove(fname)
