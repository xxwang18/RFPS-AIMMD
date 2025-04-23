'''
The following code allows to run AIMMD simulations.
Please do not change the name of the functions.
'''

###############################################################################
# Problem specification #######################################################
###############################################################################

'''
States boundaries definition.
The function state_function(trajectory) takes a MDAnalysis trajectory as input.
It returns a <U1 array of chars: the states/transition regions for each frame
in the trajectory.
'''

import numpy as np
from mdtraj.geometry import compute_distances_core as md_distances

# Hummer Overlap funtion between the 'frame' and 'signature'
# To quantify the similarity between a given molecular configuration 'frame' and a reference state 'signature'
def _best_hummer_q(frame, signature, r0):
    positions = frame.positions
    r = md_distances(positions.reshape((-1, *positions.shape)),
                     signature,
                     frame.triclinic_dimensions.reshape((-1, 3, 3))) / 10.
    q = np.mean(1.0 / (1 + np.exp(50. * (r - 1.8 * r0))), axis=1)
    return q[0]

signature = np.load('../signature.npy')
r0 = np.load('../r0.npy')

def states_function(traj):
    q = np.array([_best_hummer_q(frame, signature, r0) for frame in traj])
    results = np.repeat('R', len(q))
    results[q >= .99] = 'A'
    results[q <= .01] = 'B'
    return results


'''
Descriptor function.
The function descriptor_function(trajectory) takes a MDAnalysis trajectory as
input. It returns the input to the neural network.
'''

import itertools

topology = mda.Universe('../run.gro')
atoms = topology.select_atoms('protein').indices
heavy = topology.select_atoms('protein and not type H')
resids = topology.atoms.resids
heavy = heavy.indices
heavy_pairs = np.array(
    [(i,j) for (i,j) in itertools.combinations(heavy, 2)
        if abs(resids[i] - resids[j]) > 3])
#why is the diff resids[i->j] >3?
dmin = np.load('../dmin.npy')
dmax = np.load('../dmax.npy')
#where are they?

def descriptors_function(traj):
    results = []
    for frame in traj:
        positions = frame.positions / 10
        box = frame.triclinic_dimensions.reshape((-1, 3, 3)) / 10
        results.append(
            md_distances(positions.reshape((-1, *positions.shape)),
                         heavy_pairs, box)[0])
    return (np.array(results) - dmin[None, :]) / (
        dmax[None, :] - dmin[None, :])


'''
Neural network architecture.
The class Network contains all the info for initializing the neural network
learning the reaction coordinate WITHOUT any additional parameter:
model = Network()
'''

import torch
import aimmd
# model architecture definition
# we use a pyramidal ResNet as described in "Machine-guided path sampling to discover mechanisms of molecular self-organization" (Nat.Comput.Sci 2023)

n_lay_pyramid = 5  # number of resunits
n_unit_top = 10  # number of units in the last layer before the log_predictor
dropout_base = 0.3  # dropot fraction in the first layer (will be reduced going to the top)
n_unit_base = cv_ndim = 2064  # input dimension
# the factor by which we reduce the number of units per layer (the width) and the dropout fraction
fact = (n_unit_top / n_unit_base)**(1./(n_lay_pyramid))

# create a list of modules to build our pytorch reaction coodrinate model from
modules = []

for i in range(1, n_lay_pyramid + 1):
    modules += [aimmd.pytorch.networks.FFNet(n_in=max(n_unit_top, int(n_unit_base * fact**(i-1))),
                                             n_hidden=[max(n_unit_top, int(n_unit_base * fact**i))],  # 1 hidden layer network
                                             activation=torch.nn.Identity(),
                                             dropout={"0": dropout_base * fact**i}
                                             )
                ]
    print(f"ResUnit {i} is {max(n_unit_top, int(n_unit_base * fact**(i)))} units wide.")
    print(f"Dropout before it is {dropout_base * fact**i}.")
    modules += [aimmd.pytorch.networks.ResNet(n_units=max(n_unit_top, int(n_unit_base * fact**i)),
                                              n_blocks=1)
                ]

class Network:
    def __init__(self, modules=modules, n_out=1):
        """
        Wrapper class that initializes an instance of aimmd.pytorch.networks.ModuleStack
        with the specified modules and output configuration.
        
        Parameters:
        - modules (list): List of initialized torch.nn.Modules from arcd.pytorch.networks.
        - n_out (int): Number of output features. Default is 1, suited for binary output.
        """
        self.model = aimmd.pytorch.networks.ModuleStack(
            n_out=n_out,
            modules=modules
        )

    def __call__(self, *args, **kwargs):
        # Allows the instance to be used as a callable, passing calls to the underlying model
        return self.model(*args, **kwargs)
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Parameters:
        - x (torch.Tensor): Input tensor to be passed through the model.
        
        Returns:
        - Output tensor after processing through the model.
        """
        return self.model(x)

    def __getattr__(self, name):
        """
        Delegate attribute access to the underlying model.
        
        Parameters:
        - name (str): Name of the attribute or method being accessed.
        
        Returns:
        - The attribute or method from self.model if it exists.
        """
        return getattr(self.model, name)

network = Network()


def process_descriptors(descriptors):
    return descriptors


# standard for NN
def values_function(descriptors):
    global network
    return evaluate(network, process_descriptors(descriptors))


###############################################################################
# AIMMD run settings ##########################################################
###############################################################################

# simulations
max_excursion_length = 2000  # frames

# reweight parameters
reweight_parameters={
    'equilibrium_threshold': 5}
shooting_bias_correction = False

# selection
do_tps = False
lorentzian = np.inf
n_bins = 10
memory = 1.
selection_pool_size = 1  # target
at_least_one_transition_in_pool = False
occasionally_override_chain_with_equilibrium = False

# initialization
randomize_shooting_velocities = True

# gromacs engine
gromacs = True
grompp = 'gmx -nobackup grompp -maxwarn 5 -p ../../topol.top'
# no -r, -f, -c, -o options (will be filled automatically)
mdrun = f'gmx -nobackup mdrun -v -maxh .5 '
# no deffnm or continuation settings (will be filled automatically)
eneconv = None#f'printf "c\nc\n" | gmx -nobackup eneconv -settime'

# find trajectory extension
trajectory_extension = '.xtc'
if gromacs:
    with open('../run.mdp', 'r') as f:
        for line in f:
            line = line.split('=')
            if len(line) < 2:
                continue
            if line[0].split()[0] == 'nstxout':
                if int(line[1].split(';')[0].split()[0]):
                    trajectory_extension = '.trr'
                break

# training / saving intervals
save_interval = 10

###############################################################################
# AIMMD training function #####################################################
###############################################################################

def fit(network, pathensemble, verbose=False):
    global initial_path
    
    lr = 1e-4
    smoothening_weight = 100
    regularization_weight = 1e-4
    batch_size = 4096
    epochs = 100
    stop = 50.
    
    try:
        t0 = time.time()
        losses, scales, values, weights = [], [], [], []
        if torch.cuda.is_available():
            network.to('cuda')
        
        shots, equilibriumA, equilibriumB = scorporate_pathensembles(pathensemble)

        if not equilibriumA.nframes or not equilibriumB.nframes:
            equilibrium = process_equilibrium_pathensembles(
                equilibriumA, equilibriumB, initial_path)
        else:
            equilibrium = equilibriumA + equilibriumB
        
        dim = len(equilibrium.frame_descriptors[0])
        device = next(network.parameters()).device
        dtype = next(network.parameters()).dtype
        optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        
        # collect all descriptors and results
        if len(shots):
            shooting_descriptors = shots.shooting_descriptors.reshape(
                (-1, dim)).astype(np.float32)
            shooting_results = shots.shooting_results
            shooting_weights = shots.are_accepted.astype(float)
            shooting_values = shots.shooting_values
        else:
            shooting_descriptors = np.zeros((0, dim), dtype=np.float32)
            shooting_results = np.zeros((0, 2))
            shooting_weights = np.zeros(0)
            shooting_values = np.zeros(0)
        k = (equilibrium.internal_states == 'R') * equilibrium.are_accepted
        kA = k * (equilibrium.initial_states == 'A') 
        kB = k * (equilibrium.initial_states == 'B')
        if not np.sum(kA):
            kA = equilibrium.internal_states == 'A'
        if not np.sum(kB):
            kB = equilibrium.internal_states == 'B'
        equili_A_descriptors = np.concatenate(equilibrium.descriptors(
            kA, internal=True), axis=0).astype(np.float32)
        equili_A_results = np.repeat([[2., 0.]],
            equilibrium.internal_lengths[kA].sum(), axis=0)
        equili_A_values = np.concatenate(
            equilibrium.values(kA, internal=True))
        nA = len(equili_A_results)
        equili_B_descriptors = np.concatenate(equilibrium.descriptors(
            kB, internal=True), axis=0).astype(np.float32)
        equili_B_results = np.repeat([[0., 2.]],
            equilibrium.internal_lengths[kB].sum(), axis=0)
        equili_B_values = np.concatenate(
            equilibrium.values(kB, internal=True))
        nB = len(equili_B_results)
        scaleA = 1.
        scaleB = 1.
        if np.sum(shooting_results):
            scaleA *= min(1., np.sum(shooting_results) / (nA + nB))
            scaleB *= min(1., np.sum(shooting_results) / (nA + nB))
        equili_A_weights = np.repeat(scaleA, nA)
        equili_B_weights = np.repeat(scaleB, nB)
        
        write(f'shooting size {len(shooting_weights)}')
        write(f'eq A size {len(equili_A_weights)}')
        write(f'eq B size {len(equili_B_weights)}')
        
        # put everything together
        results = np.concatenate([shooting_results,
                                  equili_A_results,
                                  equili_B_results], axis=0)
        descriptors = process_descriptors(np.concatenate([shooting_descriptors,
                                      equili_A_descriptors,
                                      equili_B_descriptors], axis=0))
        weights = np.concatenate([shooting_weights,
                                  equili_A_weights,
                                  equili_B_weights])
        values = np.concatenate([shooting_values,
                                 equili_A_values,
                                 equili_B_values])
        '''
        results = shooting_results
        descriptors = shooting_descriptors
        weights = shooting_weights
        values = shooting_values
        '''
        
        weights = np.nan_to_num(weights)
        weights /= np.sum(weights)
        
        # training loop
        losses = []
        scales = []
        for i in tqdm(range(epochs), position=0, disable=not verbose):
            
            # sample batch
            indices = np.random.choice(len(weights), batch_size, p=weights)
            d = torch.tensor(descriptors[indices], dtype=dtype, device=device)
            d.requires_grad = True
            r = torch.tensor(results[indices], dtype=dtype, device=device)
            q = network(d)
            
            # define loss function
            def closure():
                optimizer.zero_grad()
                exp_pos_q = torch.exp(+q[:, 0])
                exp_neg_q = torch.exp(-q[:, 0])
                toA_contrib = r[:, 0] * torch.log(1. + exp_pos_q)
                toB_contrib = r[:, 1] * torch.log(1. + exp_neg_q)
                loss = torch.sum((toA_contrib + toB_contrib) / torch.sum(r))
                
                # Compute the smoothness penalty
                q_grad = torch.autograd.grad(
                    outputs=q.sum(), inputs=d, create_graph=True)[0]
                smoothness_loss = (torch.abs(q_grad) ** 2).mean()
                loss += smoothening_weight * smoothness_loss
                
                # Calculate L1 regularization
                l1_norm = sum(p.abs().sum() for p in network.parameters())
                
                # Combine original loss with L1 regularization term
                loss += regularization_weight * l1_norm
                loss.backward()
                return loss
            
            # update network
            network.train()
            loss = optimizer.step(closure)
            losses.append(float(loss) / batch_size)
            
            # report scales
            scales.append(max(float(torch.max(q)), -float(torch.min(q))))
            
            # early stopping conditions
            if scales[-1] >= stop:
                    break
        
        write(f'Training took {time.time()-t0:.1f}s')
        return network, losses, scales, values, weights
    except:
        return network, [], [], [], []

