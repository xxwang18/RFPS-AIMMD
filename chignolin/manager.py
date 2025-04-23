"""
AIMMD-parallel sampling. Credits: Gianmarco Lazzeri, Dec 2024
Usage: `python manager.py <directory> <nsteps> <n> <nA> <nB>`
The code appends to the data already present in `directory`.
"""

from utils import *

directory, nsteps, n, nA, nB, slurm = parse_arguments()
os.chdir(directory)
exec(open('params.py', 'r').read())

# initialize objects
pathensemble = None
densities = np.zeros(n_bins)
bins = np.zeros(n_bins + 1)
network = Network()
save = []

# load pathensemble
write(f'\nLoading pathensemble ({now()})')
pathensemble = load_pathensemble(
    states_function, descriptors_function, values_function)
shots, equilibriumA, equilibriumB = scorporate_pathensembles(pathensemble)

# in case of TPS: chain weights
if do_tps and len(shots):
    for chain, folder in zip(shots.pathensembles, shots.directories):
        weights = np.load(f'{folder}/chain_weights.npy')
        chain.weights[:len(weights)] = weights[:len(chain)]

# initialize steps counter
step_number = tqdm(total=nsteps, initial=len(shots))

# initialize equilibrium simulations
write(f'\nInitializing equilibrium simulations ({now()})')
initial_path = load_initial_path(
    states_function, descriptors_function, values_function)
initialize_equilibrium_simulations(initial_path, nA, nB,
                                  trajectory_extension)

# load a selection pool for each chain
write(f'\nLoading a selection pool for each chain ({now()})')
pools = load_selection_pools(
    shots, selection_pool_size,
    at_least_one_transition_in_pool, initial_path)

# update info about available workers
available = np.zeros(n, dtype=bool)
available[workers_available()[:len(available)]] = True

# train pathensemble and stop condition
def train_pathensemble_and_stop_condition():
    global shots, equilibriumA, equilibriumB, \
           densities, bins, save, step_number
    
    try:
        # update the free simulations data
        write(f'\nUpdating the free simulations data ({now()})')
        ne0 = equilibriumA.nframes + equilibriumB.nframes
        equilibriumA, equilibriumB = update_equilibrium_pathensembles(
                equilibriumA, equilibriumB, initial_path)
        pathensemble = shots + equilibriumA + equilibriumB
        ne1 = equilibriumA.nframes + equilibriumB.nframes
        write(f'    {ne1-ne0} new frames')
        
        # save the network:
        while len(save) > 1:
            write(f'\nSaving network{save[0]:06g}.h5 ({now()})')
            shutil.copy(f'cp network.h5', f'network{save.pop(0):06g}.h5')
        
        # train the network
        write(f'\nTraining the network ({now()})')
        network, check, *_ = fit(Network(), pathensemble)
        if not len(check):
            # load most recent network because training failed
            state_dict = torch.load('network.h5')
            device = next(iter(state_dict.values())).device
            network = Network().to(device)
            network.load_state_dict(state_dict)
        
        # update values
        pathensemble = shots + equilibriumA + equilibriumB  # in case changed
        write(f'\nUpdating the values for {pathensemble} ({now()})')
        update_values(pathensemble, initial_path,
            network=network, process_descriptors=process_descriptors)
        
        # get adaptation bins
        write(f'\nObtaining the adaptation bins ({now()})')
        if 'cutoff_max' in globals():
            _bins = get_bins(pathensemble, n_bins, cutoff_max=cutoff_max,
                             initial_path=initial_path)
        else:
            _bins = get_bins(pathensemble, n_bins, initial_path=initial_path)
        write(f'    adaptation bins  : {_bins}')
        
        # reweight the PE
        if not do_tps:
            write(f'\nReweighting the PE ({now()})')
            if shooting_bias_correction:
                reweight(pathensemble, bins=bins, densities=densities,
                         **reweight_parameters)
            else:
                reweight(pathensemble, bins=bins, **reweight_parameters)
        else:  # only TPS weights
            equilibriumA.weights = 0.
            equilibriumB.weights = 0.
        
        # project density
        write(f'\nProjecting the {"T" if do_tps else ""}PE density ({now()})')
        _densities = pathensemble.project(_bins)
        if not hasattr(_densities, '__len__'):
            _densities = np.zeros(len(_bins) - 1)
        _densities[_densities == 0.] = 1e-9
        _densities /= np.sum(_densities)
        
        # update network, bins, densities only now (avoid overwriting)
        bins[:] = _bins
        densities[:] = _densities
        write(f'  densities in bins: {densities}')
        
        # save the network (2nd time)
        remove('network.h5')
        write(f'\nSaving network.h5 ({now()})')
        torch.save(network.state_dict(), f'network.h5')
        while len(save):
            write(f'\nSaving network{save[0]:06g}.h5 ({now()})')
            shutil.copy('network.h5', f'network{save.pop(0):06g}.h5')
        
    except:
        pass
    
    return False

# run "train_pathensemble_and_stop_condition" once before loop
train_pathensemble_and_stop_condition()

# initialize update time
if step_number.n == 0:
    t0 = [0., 0.]
elif 'current_t0.npy' in os.listdir():
    t0 = list(np.load('current_t0.npy'))
else:
    t0 = [time.time(), time.time()]
if type(occasionally_override_chain_with_equilibrium) is bool:
    if not occasionally_override_chain_with_equilibrium:
        t0 = [np.inf, np.inf]
elif 'A' not in occasionally_override_chain_with_equilibrium:
    t0[0] = np.inf
elif 'B' not in occasionally_override_chain_with_equilibrium:
    t0[1] = np.inf

# loop that updates selection pool and assigns shooting points to nodes
def shooting_loop():
    global shots, equilibriumA, equilibriumB, \
           network, densities, bins, save, step_number, t0
    
    while step_number.n <= nsteps:
        
        # check that all jobs are running
        if not all_jobs_are_running('../jobids.txt',
            slurm=slurm, cancel_jobs_if_false=True):
            sleep(5.)
            raise
        
        # check which chain got bigger
        for k in range(n):
            chain = shots.pathensembles[k]
            chain_size = len(chain)
            shots.pathensembles[k] = update_pathensemble(
                f'{chain.directory}/chain.h5', chain)
            chain = shots.pathensembles[k]
            if len(chain) == chain_size:  # still the same length
                continue
            
            # that means we have obtained a new path
            write(f'\nObtained '
                  f'{chain.directory}/{chain.trajectory_files[-1]} ({now()})')

            # update step number and save
            step_number.update(len(chain) - chain_size); write('')
            if step_number.n % save_interval == 0:
                save.append(step_number.n + 0)
            
            # run acceptance/rejection algorithm
            if do_tps:
                if len(chain) - 1 > 0:
                    chain_weights = np.load(
                        f'{chain.directory}/chain_weights.npy')
                    chain.weights[:len(chain_weights)] = chain_weights
                run_acceptance_rejection_on_latest_path(chain, bins, densities)
                np.save(f'{chain.directory}/chain_weights.npy', chain.weights)
                write(f'Chain TPS weights: {chain.weights}')
            
            # update pool: add new path
            if (not do_tps) or chain.weights[-1]:
                pool_index = np.load(f'shots{k}/pool_index.npy')
                pools[k] = update_selection_pool(
                    pools[k], chain, pool_index,
                    selection_pool_size, at_least_one_transition_in_pool,
                    initial_path)
                pools[k].save(f'shots{k}/pool.h5')
            
            # update workers available
            available[k] = True
        
        # launch simulations on free shooting nodes
        for k in np.random.permutation(np.where(available)[0]):
            if k >= n:
                continue
            
            # load info
            chain = shots.pathensembles[k]
            i = len(chain)
            old_filename = f'path{i:06g} -> ' if i else ''
            new_filename = f'path{i+1:06g}'
            write(f'\nShooting for chain {k}: {old_filename}{new_filename} '
                  f'({now()})')
            _bins = bins.copy()  # "freeze" bins and densities
            _densities = densities.copy()
            pathensemble = shots + equilibriumA + equilibriumB
            populations = get_shot_histogram(shots, _bins) + .1
            write(f'    shot_histogram  {populations}')
            
            # lorentzian correction
            A = (bins[:-1] + bins[1:]) / 2
            if lorentzian < np.inf:
                populations *= 1 / (A ** 2 + lorentzian ** 2)
            
            # ensure SP disappeared
            while 'shoot.trr' in os.listdir(chain.directory):
                continue
            
            # load most recent network
            try:
                state_dict = torch.load('network.h5')
            except:
                write(f'   updating nn, will try again later')
                continue
            device = next(iter(state_dict.values())).device
            network = Network().to(device)
            network.load_state_dict(state_dict)
            
            try:
              if do_tps:
                shooting_point, bias, pool_index, t0 = select_shooting_point(
                  pathensemble, pools[k], _bins, _densities, populations, None)
              else:
                shooting_point, bias, pool_index, t0 = select_shooting_point(
                  pathensemble, pools[k], _bins, _densities, populations, t0)
            except:
                write('    shooting point selection failed, will try again')
                continue
            shooting_point.write(f'shots{k}/shoot.trr')
            write(f'({now()})')
            np.save(f'shots{k}/shoot_bias.npy', bias)
            np.save(f'shots{k}/pool_index.npy', pool_index)
            np.save('current_t0.npy', t0)
            
            # update workers available
            available[k] = False
    write(f'\nReached target nsteps {step_number.n} >= {nsteps} ({now()})')

# make simulations start
write('')
wait_that_all_jobs_are_running('jobids.txt', slurm)
write(f'\nMake simulations start ({now()})')
os.system('touch proceed.txt')

# run simulations
run_task(shooting_loop, train_pathensemble_and_stop_condition)

# complete
step_number.close()
cancel_all_jobs('jobids.txt', slurm=slurm)

