"""
Base libraries and functions for this project.
"""

import os
import sys
import time
import torch
import numpy as np
import mdtraj as md
import pickle
import psutil
import shutil
import select
import signal
import asyncio
import inspect
import argparse
import warnings
import importlib
import importlib.util
import itertools
import threading
import MDAnalysis as mda
import subprocess
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from time import sleep
from tqdm import tqdm
from datetime import datetime
from scipy.special import logit, expit
from mdtraj.formats import TRRTrajectoryFile
from pathensemble import *

# Suppress only UserWarnings in MDAnalysis
warnings.filterwarnings("ignore", category=UserWarning, module="MDAnalysis")

# More compact array visualization
np.set_printoptions(precision=2)


def now():
    return str(datetime.now())[11:19]

def write(text, path=None, wrap=False):
    if wrap:
        text = "\n".join(wrap(text, 80))
    text = text.replace('"',"'")
    
    if path is None:
        os.system(f'''echo "{text}"''')
    else:
        os.system(f'''echo "{text}" >> {path}''')

def remove(path):
    while os.path.exists(path):
        os.remove(path)
        write(f'removed {path}')

def get_last_modification_time(filepath):
    if os.path.exists(filepath):
        return os.path.getmtime(filepath)
    return 0.

def update_results(filename, key=[], result=[]):
    try:
        with open(filename, 'rb') as file:
            dictionary = pickle.load(file)
    except:
        dictionary = {}
    
    save = False
    for k, r in zip(key, result):
        dictionary[k] = r
        save = True
    
    if save:
        with open(filename, 'wb') as file:
            pickle.dump(dictionary, file)
    
    return dictionary


def get_trajectory_extension(parameters_file):
    trajectory_extension = '.xtc'
    with open(parameters_file, 'r') as f:
        for line in f:
            line = line.split('=')
            if len(line) < 2:
                continue
            if line[0].split()[0] == 'nstxout':
                if int(line[1].split(';')[0].split()[0]):
                    trajectory_extension = '.trr'
            break
    return trajectory_extension


###############################################################################
# Handle processes ############################################################
###############################################################################

stop_event = threading.Event()

import asyncio
import subprocess
import psutil

async def _run_command(command, condition_check=lambda: False,
                      interval=1, timeout=5):
    """
    Runs a bash command asynchronously, checks for a condition,
    and if met, terminates the process and all its children.
    
    Args:
        command (str): The bash command to run.
        condition_check (function): A function returning True if the
        condition is met.
        interval (int): Interval in seconds between condition checks.
        timeout (int): Time (s) to wait after terminate() before using kill().
    """
    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    # Continuously read stdout and stderr in real-time
    async def read_stream(stream):
        while True:
            try:
                line = await stream.readline()
            except:
                continue
            if not line:
                break
            write(f"{line.decode().strip()}")
    
    async def check_condition():
        while True:
            if process.returncode is not None:  # Process has finished
                if process.returncode != 0:
                    write(f"Process exited with an error {process.returncode}.")
                    raise
                else:
                    write("Process completed successfully.")
                break
            
            # Check the condition
            if condition_check():
                write("Condition met, terminating process and its children.")

                try:
                    parent_proc = psutil.Process(process.pid)
                    # error only if the process does not exist
                    
                    # Try to terminate the main process and all children
                    children = parent_proc.children(recursive=True)
                    for child in children:
                        try:
                            child.terminate()  # Sends SIGTERM to each child
                        except:
                            pass
                    parent_proc.terminate()  # Sends SIGTERM to the main process
                    
                    # Wait for termination, then forcefully kill if still running
                    await asyncio.sleep(timeout)
                    for child in children:
                        if child.is_running():
                            child.kill()  # Sends SIGKILL to any remaining child
                    if parent_proc.is_running():
                        parent_proc.kill()  # Sends SIGKILL to the main process
                    
                    await process.wait()  # Ensure the process is fully terminated
                except:
                    pass
                break
            
            # Wait a bit before checking again
            await asyncio.sleep(interval)
    
    # Wait for one of the tasks to complete
    await asyncio.gather(
        read_stream(process.stdout),
        read_stream(process.stderr),
        check_condition()
    )

def run_command(command, condition_check=lambda: False,
                      interval=1, timeout=5):
    asyncio.run(_run_command(
        command, condition_check, interval, timeout))
    
async def _run_task(task, condition=lambda: False, interval=1):
    async def task1():
        try:
            await asyncio.to_thread(task)
        except Exception as e:
            write(e)
    
    async def task2():
        stop = False
        while True:
            if condition():
                break
            await asyncio.sleep(interval)  # Use asyncio to yield control
    
    # Create tasks
    task1_coroutine = asyncio.create_task(task1())
    task2_coroutine = asyncio.create_task(task2())
    
    # Wait for one of the tasks to complete
    done, pending = await asyncio.wait(
        [task1_coroutine, task2_coroutine],
        return_when=asyncio.FIRST_COMPLETED
    )
    stop_event.set()
    task1_coroutine.cancel()

def run_task(task, condition_check=lambda: False,
                      interval=1, timeout=5):
    asyncio.run(_run_task(task, condition_check, interval))


def evaluate(network, descriptors, batch_size=4096):
    """
    Evaluates a neural network model using PyTorch.

    Args:
        network (torch.nn.Module): The neural network to evaluate.
        descriptors (numpy.ndarray): The input data for evaluation.
        device (torch.device): The device on which to perform the evaluation.
        dtype (torch.dtype): The data type for input tensors.
        batch_size (int, optional): The batch size for evaluation.
                                    Default is 4096.

    Returns:
        numpy.ndarray: The network output as a NumPy array.
    """
    
    # initialize
    results = []
    device = next(network.parameters()).device
    dtype = next(network.parameters()).dtype
    network.eval()
    
    # compute in batches
    with torch.no_grad():
        for batch in torch.utils.data.DataLoader(
            descriptors, batch_size=batch_size, shuffle=False):
            batch = batch.to(device=device, dtype=dtype)
            output = network(batch).detach().cpu().numpy().ravel()
            results.append(output)
    
    # return
    if len(results):
        return np.concatenate(results)
    else:
        return np.zeros(0)


def execute(command, timeout):
    write(f'Executing: {command}')
    def target():
        global proc, sdout, stderr
        try:
            proc = subprocess.Popen(
                command,
                shell=True,
                preexec_fn=os.setsid,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = proc.communicate()
            write(f'{stdout}')
        except Exception as e:
            writem(f'Error executing command: {e}')
            proc.returncode = -1  # Indicate failure

    thread = threading.Thread(target=target)
    thread.start()

    thread.join(timeout)
    if thread.is_alive():
        write('Timeout reached, killing process...')
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        thread.join()
    
    # Check for errors
    if proc.returncode != 0:
        write(f'Return code {proc.returncode}')
        if stderr:
            write(f'Error output: {stderr.decode()}')


def all_jobs_running(jobids_filename):
    with open(jobids_filename, 'r') as f:
        jobids = [int(jobid) for jobid in f.read().split()]
    for jobid in jobids:  # if not working
            if not int(os.popen(f'squeue -j {jobid} --noheader -t R 2>/dev/null | wc -l').read()):
                return False
    return True


def cancel_all_jobs(jobids_filename, slurm=False):
    with open(jobids_filename, 'r') as f:
        jobids = f.read()
    if slurm:
        os.system(f'scancel {jobids}')
    else:
        for jobid in jobids.split():
            os.system(f'kill {jobid}')
        


def all_jobs_are_running(jobids_filename,
                         slurm=False,
                         cancel_jobs_if_false=False):
    
    if type(jobids_filename) is str and not os.path.exists(jobids_filename):
        return True
    
    if slurm:
        def is_running(jobid):
            return bool(int(os.popen(f'squeue -j {jobid} '
                f'--noheader -t R 2>/dev/null | wc -l').read()))
    else:
        def is_running(jobid):
            return psutil.pid_exists(int(jobid))
    
    with open(jobids_filename, 'r') as f:
        for line in f:
            pass
    try:
        jobids = np.array([int(jobid) for jobid in line.split()])
    except:
        jobids = np.zeros(0, dtype=int)
    
    if not len(jobids):
        return True

    are_running = np.array([is_running(jobid) for jobid in jobids])
    
    # at least one job running but not all
    if np.sum(~are_running):
        not_running_jobids = jobids[~are_running]
        if cancel_jobs_if_false:
            write(f'jobids '
            f'{" ".join([str(jobid) for jobid in not_running_jobids])} '
            f'are not running at time {now()}')
            write(f'Canceling all jobs')
            if slurm:
                os.system(f'scancel {line}')
            else:
                os.system(f'pkill -f gmx')
                os.system(f'kill {line}')
        return False
    
    return True


def wait_that_all_jobs_are_running(jobids='jobids.txt', slurm=False):
    os.system('echo "Waiting before proceeding that all jobs are running..."')
    if type(jobids) is str and not os.path.exists(jobids):
        os.system('echo "Done."')
        return
    while not all_jobs_are_running(jobids, slurm=slurm):
        pass
    os.system('echo "Done."')


def wait_flag(flag='proceed.txt', jobids=None, slurm=False):
    while True:
        if os.path.exists(flag):
            return
        if jobids is not None and not slurm:
            if not all_jobs_are_running(jobids,
                slurm=slurm, cancel_jobs_if_false=True):
                sleep(5.)
                raise


###############################################################################
# Math ########################################################################
###############################################################################

def weighted_quantile(data, weights, q):
    """
    Calculate the weighted quantile of a 1D numpy array.

    Parameters:
    -----------
    data: numpy.array
        Input data (1D array).
    weights: numpy.array
        Weights corresponding to each data point (1D array).
    q: float or numpy.array
        Quantile(s) to compute (between 0 and 1).

    Returns:
    --------
    float or numpy.array
        The weighted quantile(s).
    """
    q = np.asarray(q)
    if np.any((q < 0) | (q > 1)):
        raise ValueError("Quantile values must be between 0 and 1.")

    if data.shape != weights.shape:
        raise ValueError("Data and weight arrays must have the same shape.")

    sorted_indices = np.argsort(data)
    sorted_data = data[sorted_indices]
    sorted_weights = weights[sorted_indices]
    cum_weights = np.cumsum(sorted_weights)
    normed_weights = (cum_weights - 0.5 * sorted_weights) / cum_weights[-1]

    return np.interp(q, normed_weights, sorted_data)


def interpolate(x, y, P, X, Y):
    """
    Interpolate P values on a X, Y grid.
    """
    k = ((y - Y[0, 0]) / (Y[1, 1] - Y[0, 0]))
    k = min(max(0, k), len(Y) - 1)
    h = ((x - X[0, 0]) / (X[1, 1] - X[0, 0]))
    h = min(max(0, h), len(X) - 1)
    k1, k2 = int(k), int(k) + 1
    a2 = k - int(k)
    a1 = 1 - a2
    h1, h2 = int(h), int(h) + 1
    b2 = h - int(h)
    b1 = 1 - b2
    return (
                   P[k1, h1] * a1 * b1 +
                   P[k1, h2] * a1 * b2 +
                   P[k2, h1] * a2 * b1 +
                   P[k2, h2] * a2 * b2
           ) / (a1 * b1 + a1 * b2 + a2 * b1 + a2 * b2)


def solve_committor_by_relaxation(
        X, Y, Fx, Fy, A, B, P0, progress=[5, 4, 2, 1]):
    """
    Compute committor in 2D with relaxation method (Brownian dynamics).

    Parameters
    ----------
    X: x-coordinates on a 2D grid
    Y: y-coordinates on a 2D grid
    Fx: force's x component on a 2D grid
    Fy: force's y component on a 2D grid
    A: points in A on a 2D grid
    B: points in B on a 2D grid
    P0: initial guess for committor on a 2D grid
    progress: iteratively increase the resolution based on the vector's values

    Returns
    -------
    P0: committor estimate
    """
    for split in tqdm(progress, position=0):
        X1 = X[::split, ::split]
        Y1 = Y[::split, ::split]
        P1 = P0[::split, ::split]
        Fx1 = Fx[::split, ::split]
        Fy1 = Fy[::split, ::split]
        A1 = A[::split, ::split]
        B1 = B[::split, ::split]
        dFx = np.diff(X1, axis=1)[1:-1, :-1] * Fx1[1:-1, 1:-1]
        dFy = np.diff(Y1, axis=0)[:-1, 1:-1] * Fy1[1:-1, 1:-1]
        dFx[dFx > +1.] = +1.
        dFx[dFx < -1.] = -1.
        dFy[dFy > +1.] = +1.
        dFy[dFy < -1.] = -1.
        r = np.max(np.abs(
            (((P1[2:, 1:-1] + P1[:-2, 1:-1] - 2 * P1[1:-1, 1:-1]) +
              (P1[1:-1, 2:] + P1[1:-1, :-2] - 2 * P1[1:-1, 1:-1])) +
             (dFx * (P1[2:, 1:-1] - P1[:-2, 1:-1]) +
              dFy * (P1[1:-1, 2:] - P1[1:-1, :-2])) / 2)
        ))
        while True:
            r1 = 0 + r
            for i in range(100):
                P1[1:-1, 1:-1] = (2 * (P1[2:, 1:-1] + P1[:-2, 1:-1] +
                                       P1[1:-1, 2:] + P1[1:-1, :-2]) +
                               (dFy * (P1[2:, 1:-1] - P1[:-2, 1:-1]) +
                                dFx * (P1[1:-1, 2:] - P1[1:-1, :-2]))) / 8
                P1[:, 0] = P1[:, 1]
                P1[:, -1] = P1[:, -2]
                P1[0, :] = P1[1, :]
                P1[-1, :] = P1[-2, :]
                P1[P1 < 0] = 0
                P1[P1 > 1] = 1
                P1[A1] = 0
                P1[B1] = 1
            r = np.max(np.abs(((
                 (P1[2:, 1:-1] + P1[:-2, 1:-1] - 2 * P1[1:-1, 1:-1]) +
                 (P1[1:-1, 2:] + P1[1:-1, :-2] - 2 * P1[1:-1, 1:-1])) +
                 (dFx * (P1[2:, 1:-1] - P1[:-2, 1:-1]) +
                  dFy * (P1[1:-1, 2:] - P1[1:-1, :-2])) / 2)))
            if np.abs(r - r1) < 1e-16:
                break
        if split > 1:
            P0 = np.array([interpolate(x, y, P1, X1, Y1)
                           for x, y in zip(X.ravel(), Y.ravel())]).reshape(X.shape)
        else:
            P0 = P1.copy()
    return P0


def reduce_grid(X, step, mode='one'):
    """
    Reduce a 2D grid by grouping nearby grid points, either by
    selecting one value,  summing the values, or taking their mean.

    Args:
        X (numpy.ndarray): The input 2D grid to be reduced.
        step (int): The size of the groups (grouping step)
                    to be used in both dimensions.
        mode (str): The reduction mode to be applied:
            - 'one': Select the center value in each group.
            - 'sum': Sum the values within each group.
            - 'mean': Calculate the mean of the values within each group.

    Returns:
        numpy.ndarray: The reduced 2D grid.
    """
    n_rows, n_cols = X.shape
    row_indices = np.arange(0, n_rows - 1, step)
    col_indices = np.arange(0, n_cols - 1, step)
    X_reduced = np.zeros((len(row_indices), len(col_indices)))

    for row_i, row in enumerate(row_indices):
        for col_i, col in enumerate(col_indices):
            if mode == 'one':
                X_reduced[row_i, col_i] = X[row + step // 2, col + step // 2]
            elif mode == 'sum':
                X_reduced[row_i, col_i] = np.sum(
                    X[row:row + step, col:col + step])
            elif mode == 'mean':
                X_reduced[row_i, col_i] = np.mean(
                    X[row:row + step, col:col + step])

    return X_reduced


def committor_error_and_offset(estimates, reference,
                               min_value=0.,
                               max_value=1.):
    """
    Args:
        estimates (numpy.ndarray): The fitted committor values.
        reference (numpy.ndarray): The true committor values.
        min_value: exclude if below
        max_value: exclude if above

    Returns:
        tuple of floats:
            - The mean error between the fitted committor values
            and true committor values.
            - The mean offset between fitted and true committor values
    """
    mask = ((reference >= min_value) *
            (reference <= max_value))
    estimates = estimates[mask]
    reference = reference[mask]
    error = (logit(estimates) - logit(reference)) / 4
    error = error[(~np.isinf(error)) * (~np.isnan(error))]
    return np.mean(error ** 2) ** .5, np.mean(error)


def interface_error_and_offset(estimates, reference,
                               n_interfaces=100,
                               min_value=.001,
                               max_value=.999):
    """
    Error and offset per interface
    """
    mask = ((reference >= min_value) *
            (reference <= max_value))
    estimates = estimates[mask]
    reference = reference[mask]
    interface_values = []
    interface_errors = []
    interface_offsets = []
    for interface_estimate, interface_reference in zip(
            np.array_split(estimates, n_interfaces),
            np.array_split(reference, n_interfaces)):
        interface_value = np.mean(interface_reference)
        interface_error, interface_offset = committor_error_and_offset(
            interface_estimate,
            interface_reference)
        interface_values.append(interface_value)
        interface_errors.append(interface_error)
        interface_offsets.append(interface_offset)
    return (np.array(interface_values),
            np.array(interface_errors),
            np.array(interface_offsets))


def initialize_plot():
    figure, ax = plt.subplots(1, 1, figsize=(3, 2.5))
    plt.subplots_adjust(left=0.18, bottom=0.18, right=0.99, top=0.8)
    return figure, ax

def recover_trr(trr_file, top_file, chunk_size=100):
    """
    Attempt to recover frames from a corrupted .trr file up to the highest possible frame.
    
    Args:
        trr_file (str): Path to the corrupted .trr file.
        top_file (str): Path to the topology file.
        chunk_size (int): Number of frames to load per chunk. Adjust if necessary.
        
    Returns:
        recovered_traj (md.Trajectory): Trajectory object with successfully loaded frames.
    """
    frames = []
    try:
        for chunk in md.iterload(trr_file, top=top_file, chunk=chunk_size):
            frames.append(chunk)
        return
    except Exception as e:
        write(f"Error encountered: {e}. Stopping recovery.")

    if frames:
        # Concatenate all recovered frames into a single trajectory
        recovered_traj = frames[0].join(frames[1:]) if len(frames) > 1 else frames[0]
        recovered_traj.save('.temp.trr')
        os.replace('.temp.trr', trr_file)
        return recovered_traj
    else:
        write("No frames recovered.")
        return None


def compute_autocorrelation(list_of_series, list_of_times, Tau, dTau,
                            list_of_weights=None, mu=None, verbose=False):
    result = np.repeat(1., len(Tau))
    n = np.zeros(len(Tau))
    if list_of_weights is None:
        list_of_weights = [np.ones(len(series)) for series in list_of_series]
    if mu is None:
        mu = np.mean(np.concatenate(list_of_series))
    for times, series, weights in zip(
        list_of_times, list_of_series, list_of_weights):
        for j in tqdm(range(len(Tau)), disable=not verbose, position=0):
            tau = Tau[j]
            dtau = dTau[j]
            for i, t in enumerate(times):
                dt = times[i:] - t
                keepers = (dt >= (tau - dtau)) * (dt <= (tau + dtau))
                nk = weights[i] * np.sum(weights[i:][keepers])
                n[j] += nk
                if nk:
                    result[j] += np.sum((series[i] - mu) *
                                        (series[i:][keepers] - mu) *
                                         weights[i:][keepers] * weights[i])
    result /= n
    return result / result[0]


###############################################################################
# Manager utils ###############################################################
###############################################################################

def initialize_equilibrium_simulations(initial_path, nA, nB,
                                       trajectory_extension='.trr'):
    
    if not os.path.exists(f'A{trajectory_extension}'):
        remove(f'A{trajectory_extension}')
        remove(f'B{trajectory_extension}')
        if initial_path.final_states[0] == 'A':
            initial_path.frames([-2,-1]).write(f'A{trajectory_extension}',
                invert_velocities=False, reset_time=True)
            initial_path.frames([+1,+0]).write(f'B{trajectory_extension}',
                invert_velocities=True, reset_time=True)
        else:
            initial_path.frames([-2,-1]).write(f'B{trajectory_extension}',
                invert_velocities=False, reset_time=True)
            initial_path.frames([+1,+0]).write(f'A{trajectory_extension}',
                invert_velocities=True, reset_time=True)
    for i in range(nA):
        path = f'equilibriumA/traj{i + 1:06g}.part0000{trajectory_extension}'
        if not os.path.exists(path):
            shutil.copy(f'A{trajectory_extension}', path)
            write(f'    {path}')
    for i in range(nB):
        path = f'equilibriumB/traj{i + 1:06g}.part0000{trajectory_extension}'
        if not os.path.exists(path):
            shutil.copy(f'B{trajectory_extension}', path)
            write(f'    {path}')


def reweight(pathensemble, bins=None, densities=None, **reweight_parameters):
    if not len(pathensemble):
        return

    # correction
    if bins is not None:

        are_accepted = pathensemble.are_accepted
        old_are_accepted = are_accepted.copy()
        are_accepted[pathensemble.are_shot *
        ((pathensemble.shooting_values < bins[0]) +
         (pathensemble.shooting_values >= bins[-1]))] = False
        pathensemble.are_accepted = are_accepted
        
        if densities is not None:
            if not np.sum(densities):
                densities = np.ones(len(bins) - 1)
            densities /= np.sum(densities)
            densities[densities == 0.] = 1e-9
            densities /= np.sum(densities)
            correction = 1 / np.concatenate([[np.inf], densities, [np.inf]])
            correction /= np.sum(correction)
            keepers = pathensemble.are_shot
            values = pathensemble.shooting_values[keepers]
            temp = correction[np.digitize(values, bins)]
            temp /= pathensemble.selection_biases[keepers]
            corrections = np.ones(len(pathensemble))
            corrections[keepers] = temp
            plt.plot(temp, '.')
        else:
            corrections = None
    else:
        corrections = None
    
    wA, *_ = pathensemble.reweight('A', **reweight_parameters,
                                   corrections=corrections)
    wB, *_ = pathensemble.reweight('B', **reweight_parameters,
                                   corrections=corrections)
    
    # bonus track: estimating rates
    kAB = np.nan_to_num(
        1 / np.sum(wA * pathensemble.internal_lengths))
    kBA = np.nan_to_num(
        1 / np.sum(wB * pathensemble.internal_lengths))
    write('Rates estimate')
    write(f'kAB estimate {kAB:.2e} [1/dt]')
    write(f'kBA estimate {kBA:.2e} [1/dt]')
    
    # exclude internal segments
    exclude = pathensemble.are_internal
    wA[exclude] = 0.
    wB[exclude] = 0.
    pathensemble.weights = wA + wB
    
    # return to old are accepted
    pathensemble.are_accepted = old_are_accepted


def parse_arguments(args=None):
    parser = argparse.ArgumentParser(description='AIMMD manager')
    parser.add_argument('directory', type=str,
        help='project directory')
    parser.add_argument('nsteps', type=int,
        help='target 2-way-shooting excursions in the transition region')
    parser.add_argument('n', type=int,
        help='number of nodes dedicated to 2-way shooting')
    parser.add_argument('nA', type=int,
        help='number of nodes dedicated to equilibrium simulations in A')
    parser.add_argument('nB', type=int,
        help='number of nodes dedicated to equilibrium simulations in B')
    parser.add_argument('-s', '--slurm', action='store_true')
    args = parser.parse_args(args.split() if args is not None else args)
    return args.directory, args.nsteps, args.n, args.nA, args.nB, args.slurm


def update_pathensemble(path, pathensemble=None, overwrite=False):
    
    # avoid errors
    if pathensemble is None:
        pathensemble = PathEnsemble()
    states_function = pathensemble.states_function
    descriptors_function = pathensemble.descriptors_function
    values_function = pathensemble.values_function
    
    # try loading
    new_pathensemble = PathEnsemble()
    try:
        new_pathensemble.load(path)
    except:
        return pathensemble
    
    # overwriting: just return
    if overwrite:
        return new_pathensemble

    # not long enough
    if pathensemble.nframes >= new_pathensemble.nframes:
        return pathensemble
    
    # not overwriting: keep frame values and return
    n = min(pathensemble.nframes, new_pathensemble.nframes)
    frame_values = pathensemble.frame_values[:n]
    new_pathensemble.frame_values[:n] = frame_values
    
    # functions
    new_pathensemble.states_function = states_function
    new_pathensemble.descriptors_function = descriptors_function
    new_pathensemble.values_function = values_function
    return new_pathensemble


def load_selection_pools(pathensemble, selection_pool_size,
                         at_least_one_transition=False,
                         initial_path=None,
                         directory='.',
                        verbose=True):
    """
    Parameters
    ----------
    shots: ensemble of two-way shooting chains
    selection_pool_size: target size of the output pools
    at_least_one_transition: if True, add the latest available transition
                             from the related chain to the pool if the pool
                             has no transitions; if the chain has none
                             either, add initial
    folder: the directory where the "shots" folders are be located
    
    Returns
    -------
    pools: list of PathEnsemble instance, each one is the selection pool
           for a chain
    """
    pools = []
    shots = extract_pathensembles(pathensemble, 'shots')
    
    # try loading the pool
    for k, chain in enumerate(shots.pathensembles):
        if verbose: write(f'pool {k}')
        pool = PathEnsemble('.', '../run.gro',
                            pathensemble.states_function,
                            pathensemble.descriptors_function,
                            pathensemble.values_function)
        if 'pool.h5' in os.listdir(f'{directory}/shots{k}'):
            try:
                pool.load(f'{directory}/shots{k}/pool.h5')
            except:  # not so bad
                if verbose:  write(
                    'warning: reloading pool_backup.h5 instead of pool.h5')
                pool.load(f'{directory}/shots{k}/pool_backup.h5')
            pool.directory = '.'
        
        # most recent if current pool has no length
        if not len(pool):
            if len(pools):
                pool = pools[np.random.choice(len(pools))][:]  # rand duplicate
            elif len(chain):
                pool = chain[-selection_pool_size:]
            else:
                pool = initial_path[:]
        
        # at most selection_pool_size elements,
            # no randomization for good restarting
        pool = pool[-selection_pool_size:]
        
        if at_least_one_transition and not np.sum(
            pool.are_transitions) and len(chain):
            candidates = np.where(chain.are_transitions * chain.are_accepted)[0]
            if len(candidates):  # try adding the latest transition in chain
                pool = (chain[candidates[-1]] + pool).merge()
            else:  # add the initial path if none are available
                pool = (initial_path + pool).merge()
        
        for fname in pool.shooting_trajectory_filenames:
            if verbose:  write(f'{pool.directory}/{fname}')
        
        # update list and return
        pools.append(pool)
    
    return pools


def load_shot_pathensembles(directory='.'):
    """
    Scans in the "shot" folders.
    """
    pathensembles = []
    child_folders = sorted(
        [child for child in next(os.walk(directory))[1]
         if child[:5] == 'shots' and child[5:].isdigit()],
    key=lambda x: int(x[5:]))
    
    # add new
    for child_folder in child_folders:
        path = f'{directory}/{child_folder}/chain.h5'
        pathensemble = update_pathensemble(path,
            PathEnsemble(f'{directory}/{child_folder}', '../../run.gro'))
        pathensembles.append(pathensemble)
    
    return PathEnsemblesCollection(*pathensembles)



def load_equilibrium_pathensembles(folder='.', filenames=[],
                                   pathensembles=[], overwrite=False):
    pathensembles = list(pathensembles)
    new_filenames = sorted(
        [filename for filename in os.listdir(folder)
         if filename[:4] == 'traj' and filename[-3:] == '.h5' and
         len(filename) == 13 and filename[-9:-3].isdigit()])
    
    # update old
    for i, filename in enumerate(filenames):
        if filename not in new_filenames:
            continue
        path = f'{folder}/{filename}'
        pathensembles[i] = update_pathensemble(
            path, pathensembles[i], overwrite)
        new_filenames.remove(filename)
    
    # add new
    for filename in new_filenames:
        path = f'{folder}/{filename}'
        pathensembles.append(update_pathensemble(path, overwrite=overwrite))
    
    return PathEnsemblesCollection(*pathensembles)
 
 
def load_pathensemble(
    states_function, descriptors_function, values_function,
    directory='.', update_descriptors=False,
    old_pathensemble=None, verbose=True):
    shots = load_shot_pathensembles(directory=directory)
    equilibriumA = load_equilibrium_pathensembles(
        folder=f'{directory}/equilibriumA')
    equilibriumB = load_equilibrium_pathensembles(
        folder=f'{directory}/equilibriumB')
    if old_pathensemble is not None:
        oldS, oldA, oldB = scorporate_pathensembles(old_pathensemble)
        
        # keep old if the other is not present
        if len(shots.pathensembles) < len(oldS.pathensembles):
            shots = oldS
        if len(equilibriumA.pathensembles) < len(oldA.pathensembles):
            equilibriumA = oldA
        if len(equilibriumB.pathensembles) < len(oldB.pathensembles):
            equilibriumB = oldB

        # keep old weights if present
        for p1, p2 in zip(shots.pathensembles, oldS.pathensembles):
            p1.weights[:] = 0.
            p1.weights[:len(p2)] = p2.weights[:len(p1)]
        for p1, p2 in zip(equilibriumA.pathensembles, oldA.pathensembles):
            p1.weights[:] = 0.
            p1.weights[:len(p2)] = p2.weights[:len(p1)]
        for p1, p2 in zip(equilibriumB.pathensembles, oldB.pathensembles):
            p1.weights[:] = 0.
            p1.weights[:len(p2)] = p2.weights[:len(p1)]
    
    pathensemble = shots + equilibriumA + equilibriumB
    pathensemble.states_function = states_function
    pathensemble.descriptors_function = descriptors_function
    pathensemble.values_function = values_function
    
    if verbose:
        write(f'shots: {shots}')
        t0 = time.time()
        for i, p in enumerate(shots.pathensembles):
            if len(p):
                t = t0 - p.completion_times[-1]
            else:
                t = 0
            nt = np.sum(p.are_transitions)
            write(f'    chain {i}: {len(p)} paths, {nt} transitions, '
                  f'last updated {t:.0f} s ago')
        write(f'equilibriumA: {equilibriumA}')
        write(f'equilibriumB: {equilibriumB}')
    
    if update_descriptors:
        if verbose:
            write(f'Updating descriptors')
        pathensemble.update_descriptors(verbose=True)
    
    return pathensemble


def extract_pathensembles(pathensemble, expression):
    return PathEnsemblesCollection(*[pathensemble for pathensemble in
        pathensemble.pathensembles if
        expression in pathensemble.directory.split('/')[-1]])


def scorporate_pathensembles(pathensemble):
    shots = extract_pathensembles(pathensemble, 'shots')
    equilibriumA = extract_pathensembles(pathensemble, 'equilibriumA')
    equilibriumB = extract_pathensembles(pathensemble, 'equilibriumB')
    return shots, equilibriumA, equilibriumB
 
 
def load_initial_path(states_function, descriptors_function, values_function,
                      directory='.'):
    initial_path = PathEnsemble(directory, '../run.gro',
            states_function, descriptors_function, values_function)
    if not initial_path.append(f'initial.trr')[0]:
        initial_path.append(f'initial.xtc')
    initial_path.split()
    initial_path = initial_path[np.where(initial_path.are_transitions)[0][0]]
    write(f'Initial path: {initial_path}')
    return initial_path

def update_equilibrium_pathensembles(equilibriumA, equilibriumB, initial_path):
    states_function = initial_path.states_function
    descriptors_function = initial_path.descriptors_function
    values_function = initial_path.values_function
    try:
        equilibriumA = load_equilibrium_pathensembles('equilibriumA',
            [f'{trajectory.trajectory_files[-1][:10]}.h5'
             for trajectory in equilibriumA.pathensembles],
            equilibriumA.pathensembles) # update equilibrium A
    except:
        pass
    try:
        equilibriumB = load_equilibrium_pathensembles('equilibriumB',
            [f'{trajectory.trajectory_files[-1][:10]}.h5'
             for trajectory in equilibriumB.pathensembles],
            equilibriumB.pathensembles) # update equilibrium B
    except:
        pass
    equilibriumA.states_function = states_function
    equilibriumA.descriptors_function = descriptors_function
    equilibriumA.values_function = values_function
    equilibriumB.states_function = states_function
    equilibriumB.descriptors_function = descriptors_function
    equilibriumB.values_function = values_function
    return equilibriumA, equilibriumB


def process_equilibrium_pathensembles(equilibriumA, equilibriumB, initial_path):
    if equilibriumA.nframes > 0 and equilibriumB.nframes > 0:
        return equilibriumA + equilibriumB
    initial_path = initial_path[:].unsplit()
    initialA = initial_path.crop(frame_indices=initial_path.frame_states =='A')
    initialB = initial_path.crop(frame_indices=initial_path.frame_states =='B')
    if equilibriumB.nframes > 0:
        return initialA + equilibriumB
    if equilibriumA.nframes > 0:
        return equilibriumA + initialB
    return initialA + initialB


def results_available(folder='.'):
    """
    Scans in the "shot" folders.
    """
    global trajectory_extension
    
    pathensembles = []
    child_folders = sorted(
        [child for child in next(os.walk(folder))[1]
         if child[:5] == 'shots' and child[5:].isdigit()])
    
    return [f'result{trajectory_extension}' in os.listdir(f'{folder}/{child}')
            for child in child_folders]


def workers_available(folder='.'):
    """
    Scans in the "shot" folders.
    """
    child_folders = sorted(
        [child for child in next(os.walk(folder))[1]
         if child[:5] == 'shots' and child[5:].isdigit()])
    
    return ['shoot.trr' not in os.listdir(f'{folder}/{child}')
            for child in child_folders]


# in case of TPS
def run_acceptance_rejection_on_latest_path(chain, bins, densities):
    def compute_sp_bias(values, sp_value, bins, densities):
        densities = np.append(densities, [np.inf])
        values = chain.values(leading, internal=True)[0]
        biases = 1 / densities[np.digitize(values, bins) - 1]
        sp_bias = 1 / densities[np.digitize(sp_value, bins) - 1]
        sp_bias /= np.sum(biases)
        return sp_bias
    
    # ensure weight is zero
    chain.weights[-1] = 0.
    
    leading = None
    if np.sum(chain.weights):  # leading trajectory exists
            leading = np.where(chain.weights)[0][-1]
    
    if chain.are_accepted[-1] and chain.are_transitions[-1]:
        if leading is not None:
            keepers = [leading, -1]
            chain.update_values(key=keepers)  # get the values
            leading_values, trial_values = chain.values(
                keepers, internal=True)
            leading_sp_value, trial_sp_value = chain.shooting_values[keepers]
            # run acceptance/rejection
            leading_sp_bias = compute_sp_bias(
                leading_values, leading_sp_value, bins, densities)
            trial_sp_bias = compute_sp_bias(
                trial_values, trial_sp_value, bins, densities)
            acceptance = trial_sp_bias / leading_sp_bias
        else:
            acceptance = np.inf  # no leading trajectory
        write(f'    acceptance probability: {acceptance:.3f}')
        if np.random.random() < acceptance:
            leading = -1
            write(f'    accepted')
        else:
            write(f'    rejected')
    else:
        write(f'    acceptance probability: {0:.3f} (not reactive)')
        write(f'    rejected')
    write('')
    
    # update weights
    if leading is not None:
        chain.weights[leading] += 1.


def update_selection_pool(pool, chain, pool_index,
                          selection_pool_size,
                          at_least_one_transition=False,
                          initial_path=None):
    """
    pool: the current PathEnsemble object from which you select shooting points
    chain: the updated chain with the last element in it
    pool_index: the index in pool you want to substitute with the curent path
    selection_pool_size: target size of selection pool
    at_least_one_transition: if True, add the latest available transition
                             from the related chain to the pool if the pool
                             has no transitions; if the chain has none
                             either, add initial
    initial_path: PathEnsemble obj required if at_least_one_transition is True
    """
    
    write('Updating the selection pool')
    pool.prune_trajectory_files()
    
    # add path
    path = chain[-1]
    fname = path.trajectory_files[0]
    if path.are_accepted[0]:
        write(f'    added {fname} to the selection pool')
        pool = (pool + path).merge()
    else:
        write(f'    NOT added {fname} to the selection pool (not accepted)')

    # remove oldest paths at random (5%)
    # to avoid getting stuck with bad shooting points
    if len(pool) > 1 and np.random.random() < .05:
        fname = pool.shooting_trajectory_filenames[0]
        pool = pool[1:]
        write(f'    removed {fname} in pool (random)')
    
    # remove pool_index path if pool is above selection_pool_size
    if len(pool) > selection_pool_size:
        keepers = np.ones(len(pool), dtype=bool)
        fname = pool[pool_index].trajectory_files[0]
        keepers[pool_index] = False
        pool = pool[keepers]
        write(f'    removed {fname} in pool')
    
    # other paths if pool is too big
    while len(pool) > selection_pool_size:
        fname = pool.shooting_trajectory_filenames[0]
        pool = pool[1:]
        write(f'    removed {fname} in pool')
    
    # (re)add latest reactive or initial in case there is no reactive one
    if at_least_one_transition and not np.sum(pool.are_transitions):
        candidates = np.where(chain.are_transitions)[0]
        if len(candidates):  # try adding the latest transition in chain
            path = chain[candidates[-1]]
        else:  # add the initial path if none are available
            path = initial_path
        fname = path.trajectory_files[0]
        pool = (path + pool).merge()
        write(f'    added {fname} to the selection pool '
              f'(latest transition in chain)')
    k = chain.directory[-1]
    write(f'pool {k} (size: {len(pool)})')
    for fname in pool.shooting_trajectory_filenames:
        write(f'{pool.directory}/{fname}')
    return pool


def update_values(*pathensembles, only_zeros=False,
        network=None, process_descriptors=lambda x : x):
    if network is not None:
        values_function = lambda x : evaluate(network, process_descriptors(x))
    else:
        values_function = None
    for pathensemble in pathensembles:
        pathensemble.update_values(only_reactive=True, only_zeros=only_zeros,
                                   values_function=values_function)


def get_bins(pathensemble, n_bins,
             cutoff_min=.5, cutoff_max=10., cutoff_default=3.,
             initial_path=None):
    shots, equilibriumA, equilibriumB = scorporate_pathensembles(pathensemble)
    limit = np.sum(shots.are_transitions)
    if not len(shots) and initial_path is not None:
        shots = initial_path
    equilibrium = equilibriumA + equilibriumB
    
    # restrict cutoff_max to shots range NO! I don't care if you
    # ...can't escape then
    begin = -cutoff_max
    end = +cutoff_max
    
    # (inverse) eq. crossing probability histogram from A and from B
    try:
        eA = np.append(np.sort(
            equilibrium.max_values(equilibrium.initial_states=='A'))[::-1],
            [-np.inf])
    except:
        eA = np.array([-np.inf])
    try:
        eB = np.append(np.sort(
            equilibrium.min_values(equilibrium.initial_states=='B')),
            [+np.inf])
    except:
        eB = np.array([+np.inf])
    
    # assign
    if not limit:
        cutoff_default = max(cutoff_min, cutoff_default)
        begin = np.clip(begin, -cutoff_default, -cutoff_min)
        end = np.clip(end, +cutoff_min, +cutoff_default)
    else:
        begin = np.clip(eA[min(limit, len(eA) - 1)], begin, -cutoff_min)
        end = np.clip(eB[min(limit, len(eB) - 1)], +cutoff_min, end)
    
    return np.linspace(begin, end, n_bins + 1)
    
    for _ in range(100):
        bins = np.linspace(begin, end, n_bins + 1)
        p = expit((bins[:-1] + bins[1:]) / 2)
        limit = round(np.mean(p) * len(shots))
        
        # assign
        if not limit:
            begin = -cutoff_default
            end = +cutoff_default
        else:
            begin = np.clip(eA[min(limit, len(eA) - 1)], begin, -cutoff_min)
            end = np.clip(eB[min(limit, len(eB) - 1)], +cutoff_min, end)
    return bins


def get_shot_histogram(shots, bins,
                       memory=1., directory='.', topology='../run.gro'):
    """
    For biased selection.
    Returns populations and equilibrium weights.
    """
    
    values = shots.shooting_values

    # process with memory
    if memory < 1:
        order = np.argsort(shots.completion_times)
        keepers = shots.are_accepted
        exclude = order[:int(len(keepers) * (1-memory))]
        keepers[exclude] = False
        keepers = np.where(keepers)[0]
    
        # shooting data
        if len(keepers):
            values = values[keepers]
        else:
            values = np.zeros(0)
    
    # current descriptors data
    current_shooting_descriptors = []
    for folder in shots.directories:
        fname = f'{folder}/shoot.trr'
        if os.path.exists(fname):
            try:
                current_shooting_descriptors.append(
                    shots.descriptors_function([mda.Universe(
                        f'{directory}/{topology}', fname).trajectory[0]])[0])
            except:
                pass
    if len(current_shooting_descriptors):
        current_values = shots.values_function(
            np.array(current_shooting_descriptors))
    else:
        current_values = np.zeros(0)
    write(f'current SP values {current_values}')
    
    return np.histogram(
            np.append(values[values != 0], current_values), bins)[0].astype(float)


def get_histograms(pathensemble, bins, memory=1.,
                   directory='.', topology='../run.gro'):
    """
    For biased selection.
    Returns populations and equilibrium weights.
    """
    
    shots, equilibriumA, equilibriumB = scorporate_pathensembles(pathensemble)
    equilibrium = equilibriumA + equilibriumB
    
    # process with memory
    order = np.argsort(shots.completion_times)
    keepers = shots.are_accepted
    exclude = order[:int(len(keepers) * (1-memory))]
    keepers[exclude] = False
    keepers = np.where(keepers)[0]
    
    # shooting data
    if len(keepers):
        values = shots.shooting_values[keepers]
    else:
        values = np.zeros(0)
    
    # current descriptors data TODO load into pathensemble
    current_shooting_descriptors = []
    for folder in shots.directories:
        fname = f'{folder}/shoot.trr'
        if os.path.exists(fname):
            try:
                current_shooting_descriptors.append(
                    pathensemble.descriptors_function([mda.Universe(
                        f'{directory}/{topology}', fname).trajectory[0]])[0])
            except:
                pass
    if len(current_shooting_descriptors):
        current_values = pathensemble.values_function(
            np.array(current_shooting_descriptors))
    else:
        current_values = np.zeros(0)
    write(f'current SP values {current_values}')
    
    shot_histogram = np.histogram(
        np.append(values, current_values), bins)[0].astype(float)
    
    # equilibrium data
    eq_A_histogram = np.zeros(len(bins))
    eq_B_histogram = np.zeros(len(bins))
    try:
        k = equilibrium.internal_states == 'R'
        kA = k * (equilibrium.initial_states == 'A')
        kB = k * (equilibrium.initial_states == 'B')
        for i in np.digitize(equilibrium.max_values(kA),
                             np.append(bins, [np.inf])):
            eq_A_histogram[:max(i - 1, 0)] += memory
        for i in np.digitize(equilibrium.min_values(kB),
                             np.append([-np.inf], bins), right=True):
            eq_B_histogram[i:] += memory
    except:
        pass
    eq_A_histogram = np.diff(eq_A_histogram[::-1])[::-1]
    eq_B_histogram = np.diff(eq_B_histogram)
    
    histogram = shot_histogram + eq_A_histogram + eq_B_histogram
    return histogram, shot_histogram, eq_A_histogram, eq_B_histogram


def extract_frame(trajectory, position, topology):
    while True:
        try:
            shooting_point = MDATrajectory([mda.Universe(
                topology, trajectory)], [0], [position])
        except Exception as exception:
            write(f'{exception}')
            sleep(.1)
            raise
        if len(shooting_point):
            break
        sleep(.1)
    return shooting_point


def mean_lengths(shots, begin, end):
    v = shots.shooting_values
    k = np.where((begin <= v) * (v < end))[0]
    if not len(k):
        return 1.
    r = []
    for v, s in zip(shots.values(k), shots.states(k)):
        r.append(np.sum((begin <= v) * (v < end) * (s == 'R')))
    return np.mean(r)


def select_shooting_point(pathensemble, pool, bins, densities,
                          populations=None, t0=[0., 0.]):
    """
    Parameters
    ----------
    pathensemble
    pool: selection pool (without equilibrium)
    bins: foliate the space according to the model's RC (logit comm)
    population: how to bias in the adaptation bins
    t0: minimum time of the equilibrium simulations that can be selected
    
    Returns
    -------
    shooting_point: mdtraj object
    pool_index: for substitution (in case producing an accepted path)
    t0: current time
    """
    
    shots, equilibriumA, equilibriumB = scorporate_pathensembles(pathensemble)
    equilibrium = equilibriumA + equilibriumB
    
    # statistics
    write(f'selection pool')
    for fname in pool.shooting_trajectory_filenames:
        write(f'{pool.directory}/{fname}')
    
    # bias by densities and populations populations
    if not np.sum(densities):
        _densities = np.ones(len(bins) - 1)
    else:
        _densities = densities.copy()
    _densities[_densities == 0.] = 1e-9
    _densities /= np.sum(_densities)
    write(f'bins                  {bins}')
    write(f'densities             {_densities}')
    if populations is None:
        populations = np.ones(len(_densities))
    write(f'populations           {populations}')
    _densities *= populations
    _densities /= np.sum(_densities)
    correction = 1 / np.concatenate([[np.inf], _densities, [np.inf]])
    write(f'density correction by {correction[1:-1]}')
    
    # selecting from pool
    write(f'    going to selection pool')
    values = pathensemble.values_function(
        np.concatenate(pool.descriptors(internal=True)))
    states = np.concatenate(pool.states(internal=True))
    histogram = np.histogram(values, bins)[0]
    positions = np.concatenate(pool.trajectory_positions(internal=True))
    files = np.concatenate(pool.trajectory_filenames(internal=True))
    lengths = pool.internal_lengths
    indices = np.repeat(np.arange(len(pool)), lengths)  # pool indices
    write(f'the selection pool yielded {len(values)} candidate points')
    write(f'    hist: {histogram}')
    
    # correction
    weights = np.repeat(1 / lengths, lengths)
    weights *= correction[np.digitize(values, bins)]
    if not np.sum(weights):  # recover from unpleasant situation
        weights[np.argmin(np.abs(values))] = 1.
    weights /= np.sum(weights)
    histogram = np.histogram(values, bins, weights=weights)[0]
    write(f'correct.: {histogram}')
    
    # select point
    i = np.random.choice(len(values), p=weights)
    fname = files[i]
    position = positions[i]
    index = indices[i]
    value = values[i]
    write(f'selecting shooting point {fname}, {position}')
    write(f'   pool position: {index} ({pool[index].trajectory_files[0]})')
    k = np.digitize(value, bins) - 1
    write(f'   value: {value:.2f}, bin {k}, state: {states[i]}')
    selection_bias = correction[k + 1]
    if not selection_bias:
        selection_bias = np.inf
    write(f'selection bias {selection_bias}')

    # can you override?
    # try selecting from equilibrium: load data
    new_t0 = list(t0)
    if len(equilibriumA.pathensembles) and equilibriumA.nframes:
        keepersA = (equilibriumA.frame_states == 'R') * (
                    equilibriumA.frame_simulation_times >= t0[0])
    else:
        keepersA = np.zeros(0, dtype=bool)
    nA = np.sum(keepersA)
    if len(equilibriumB.pathensembles) and equilibriumB.nframes:
        keepersB = (equilibriumB.frame_states == 'R') * (
                    equilibriumB.frame_simulation_times >= t0[1])
    else:
        keepersB = np.zeros(0, dtype=bool)
    nB = np.sum(keepersB)
    eq_size = nA + nB
    keepers = np.append(keepersA, keepersB)
    if eq_size:
        values = pathensemble.values_function(
                    equilibrium.frame_descriptors[keepers])
        states = equilibrium.frame_states[keepers]
    else:
        values = np.zeros(0)
        states = np.zeros(0, dtype='<U1')
    positions = equilibrium.frame_trajectory_positions[keepers]
    files = np.array(equilibrium.trajectory_files)[
        equilibrium.frame_trajectory_indices[keepers]]
    histogram = np.histogram(values, bins)[0]
    write(f'equilibrium data yielded {eq_size} candidate points '
          f'({time.time() - t0[0]:.0f} s from A, '
          f'{time.time() - t0[1]:.0f} s) from B')
    write(f'    hist: {histogram}')
        
    # try selecting from equilibrium: are there candidates?
    if 0 <= k < len(bins) - 1:
        weights = (values >= bins[k]) * (values <= bins[k + 1])
    elif k == len(bins) - 1:
        weights = (values > bins[-1]) * (values <= value)
    else:
        weights = (values < bins[0]) * (values >= value)
    eq_candidates = np.where(weights)[0]
    write(f'   {len(eq_candidates)} candidates in bin')
    
    # success
    if len(eq_candidates): 
        i = np.random.choice(eq_candidates)
        fname = files[i]
        position = positions[i]
        value = values[i]
        write(f'   overriding with shooting point {fname}, {position}')
        write(f'   value: {value:.2f}, bin {k}, state: {states[i]}')
        
        # update time
        if i < nA:
            new_t0[0] = np.max(equilibriumA.frame_simulation_times)
        else:
            new_t0[1] = np.max(equilibriumB.frame_simulation_times)
    
    try:
        return (extract_frame(fname, position, '../run.gro'),
                selection_bias, index, new_t0)
    except:
        write('Attention! Frame extraction failed. Attempting a new one')
        return select_shooting_point(
            pathensemble, pool, bins, densities, populations, t0)


###############################################################################
#### ANALYSIS #################################################################
###############################################################################


def crop_pathensemble(pathensemble, step_number):    
    shots, equilibriumA, equilibriumB = scorporate_pathensembles(pathensemble)
    if step_number is None:
        keepers = None
        t1 = np.inf
    else:
        completion_times = shots.completion_times
        keepers = np.argsort(completion_times)[:step_number]
        t1 = completion_times[keepers][-1]
    shots = shots[keepers]
    equilibriumA = equilibriumA.crop(tmax=t1)
    equilibriumB = equilibriumB.crop(tmax=t1)
    return shots + equilibriumA + equilibriumB


def initialize_reference_pathensemble(
    states_function, descriptors_function,
    directory='.', topology='run.gro', xy_traj='xy.xtc',
    weights=None):
    reference_pe = PathEnsemble(directory, topology,
        states_function, descriptors_function)
    reference_pe.append(xy_traj, verbose=True)
    frame_indices =  np.zeros(reference_pe.nframes * 3, dtype=int)
    frame_indices[1::3] = np.arange(reference_pe.nframes)
    reference_pe.frame_states[0] = 'Z'
    reference_pe._PathEnsemble__frame_indices = frame_indices
    reference_pe._PathEnsemble__lengths = np.ones(
        reference_pe.nframes, dtype=int) * 3
    reference_pe._PathEnsemble__shooting_indices = np.zeros(
        reference_pe.nframes, dtype=int)
    if weights:
        reference_pe._PathEnsemble__weights = nd.load(weights).ravel()
    else:
        reference_pe._PathEnsemble__weights = np.ones(reference_pe.nframes)
    reference_pe._PathEnsemble__are_accepted = np.ones(
        reference_pe.nframes, dtype=bool)  
    return reference_pe


def estimate_transition_rates_from_equilibrium(equilibrium, dt=1.):
    kAB0 = []
    kBA0 = []
    kAB0_max = []
    kBA0_max = []
    kAB0_min = []
    kBA0_min = []
    TP_lengths = []
    TP_lengths_max = []
    TP_lengths_min = []
    times0 = []
    total_tAB = []
    total_tBA = []
    current_lengths = np.zeros(0)

    transitions = equilibrium[equilibrium.are_transitions]
    shooting_trajectory_indices = equilibrium.shooting_trajectory_indices
    tr_shooting_trajectory_indices = transitions.shooting_trajectory_indices    
    
    trajectory_indices = []
    for trajectory_index in shooting_trajectory_indices:
        if trajectory_index not in trajectory_indices:
            trajectory_indices.append(trajectory_index)

    initial_times = equilibrium.initial_times
    final_times = equilibrium.final_times
    if len(trajectory_indices) == len(transitions):
        pathensemble = [transitions]
        final_times = [transitions.final_times[-1]]
    else:
        pathensemble = [
            transitions[tr_shooting_trajectory_indices == trajectory_index]
            for trajectory_index in trajectory_indices]
        final_times = [final_times[
            shooting_trajectory_indices == trajectory_index][-1] -
            initial_times[
            shooting_trajectory_indices == trajectory_index][0]
            for trajectory_index in trajectory_indices]
    
    for equilibrium, base in zip(pathensemble, padcumsum(final_times)):
        t = []
        ti = equilibrium.frame_times[0] * dt
        t0 = equilibrium.final_times * dt - ti
        lengths = equilibrium.internal_lengths * dt
        if len(times0):
            times0 += list(t0 + base * dt)
        else:
            times0 = list(t0)
        start_from_A = equilibrium.final_states[0] == 'A'
        for i in range(len(t0)):
            current_lengths = np.append(current_lengths, [lengths[i]])
            t.append(t0[i])
            if start_from_A:
                tAB = np.diff(t)[0::2]  # BA information
                tBA = np.diff(t)[1::2]  # AB information
            else:
                tAB = np.diff(t)[1::2]  # BA information
                tBA = np.diff(t)[0::2]  # AB information
            current_tAB = np.append(total_tAB, tAB)
            current_tBA = np.append(total_tBA, tBA)
            if len(current_tAB):
                kAB = 1 / np.mean(current_tAB)
                temp = []
                for bootstrapping_event in range(1000):
                    k = np.random.choice(len(current_tAB), len(current_tAB))
                    temp.append(1 / np.mean(current_tAB[k]))
                kAB_max = np.quantile(temp, .975)
                kAB_min = np.quantile(temp, .025)
            else:
                kAB = np.nan
                kAB_max = np.nan
                kAB_min = np.nan
            if len(current_tBA):
                kBA = 1 / np.mean(current_tBA)
                temp = []
                for bootstrapping_event in range(1000):
                    k = np.random.choice(len(current_tBA), len(current_tBA))
                    temp.append(1 / np.mean(current_tBA[k]))
                kBA_max = np.quantile(temp, .975)
                kBA_min = np.quantile(temp, .025)
            else:
                kBA = np.nan
                kBA_max = np.nan
                kBA_min = np.nan
            temp = []
            for bootstrapping_event in range(1000):
                k = np.random.choice(len(current_lengths), len(current_lengths))
                temp.append(np.mean(current_lengths[k]))
            TP_lengths.append(np.mean(current_lengths))
            TP_lengths_max.append(np.quantile(temp, .975))
            TP_lengths_min.append(np.quantile(temp, .025))
            kAB0.append(kAB)
            kBA0.append(kBA)
            kAB0_max.append(kAB_max)
            kBA0_max.append(kBA_max)
            kAB0_min.append(kAB_min)
            kBA0_min.append(kBA_min)
        total_tAB = current_tAB
        total_tBA = current_tBA
        
    return (np.array(kAB0), np.array(kBA0),
            np.array(kAB0_max), np.array(kBA0_max),
            np.array(kAB0_min), np.array(kBA0_min),
            np.array(TP_lengths),
            np.array(TP_lengths_max), np.array(TP_lengths_min),
            np.array(times0))


def compute_energies_and_rates(pathensemble,
                               bins=[-np.inf, +np.inf],
                               bootstrapping=0,
                               reweight_while_bootstrapping=False,
                               states='AB',
                               reweight_parameters={},
                               f=None,
                               frames=False,
                               verbose=False):
    # boostrap
    bootstrapping_results = []
    bootstrapping_k = []
    bootstrapping_e = []
    bootstrapping_z = []
    
    for _ in tqdm(range(bootstrapping), position=0, disable=not verbose):
        k = np.random.choice(len(pathensemble), len(pathensemble))
        if reweight_while_bootstrapping:
            E = []
            Z = []
            K = []
            total_weights = pathensemble.weights * 0.
            for state in states:
                w, t1, t2, t3, z, m, e, s, t4 = pathensemble.reweight(
                    state=state,key=k,**reweight_parameters)
                E.append(e)
                Z.append(z)
                old_weights = pathensemble.weights
                weights = pathensemble.weights * 0.
                for i, kk in enumerate(k):
                    weights[kk] += w[i]
                total_weights += weights
                pathensemble.weights = weights
                K.append(1 / pathensemble.project()[0])
                pathensemble.weights = old_weights
            pathensemble.weights = total_weights
            bootstrapping_results.append(
                pathensemble.project(bins=bins, f=f, frames=frames))
            pathensemble.weights = old_weights
            bootstrapping_k.append(K)
            bootstrapping_e.append(E)
            bootstrapping_z.append(Z)
        else:
            bootstrapping_k.append([np.nan])
            bootstrapping_e.append([np.nan])
            bootstrapping_z.append([np.nan])
            bootstrapping_results.append(pathensemble.project(
                key=k, bins=bins, f=f, frames=frames))
        bootstrapping_results[-1] /= np.sum(bootstrapping_results[-1])
    bootstrapping_results = np.array(bootstrapping_results)
    bootstrapping_k = np.array(bootstrapping_k)
    bootstrapping_e = np.array(bootstrapping_e, dtype=object)
    bootstrapping_z = np.array(bootstrapping_z, dtype=object)
    
    result = pathensemble.project(bins=bins, f=f, frames=frames)
    result /= np.sum(result)
    return (result, bootstrapping_results,
            bootstrapping_k,
            bootstrapping_e,
            bootstrapping_z)


def plot_2d_energy(X, Y, F, levels,
                   X2=None, Y2=None, P=None,
                   rc_levels=None,
                   rc_labels=True,
                   cmap='magma', clabel='[kT]',
                   rotate_clabel=True,
                   xA=None, yA=None,
                   xB=None, yB=None,
                   rA=None, rB=None,
                   wrmse=0.):
    figure, ax = plt.subplots(1, 1, figsize=(3, 2.5))
    if X2 is not None:
        drop = len(X) // (len(X2) * 2)
    else:
        drop = 0
    print(drop)
    
    X = X[drop:len(X)-drop,
                   drop:len(X[0])-drop]
    Y = Y[drop:len(Y)-drop,
                   drop:len(Y[0])-drop]
    F = F[drop:len(F)-drop,
                   drop:len(F[0])-drop]
    plt.contourf(X, Y, F, levels=levels, cmap=cmap, zorder=40)
    ax.set_aspect(abs((X[-1, -1] - X[0, 0]) / (Y[-1, -1] - Y[0, 0])))
    plt.subplots_adjust(left=0.16, bottom=0.1, right=0.8, top=1.04)
    
    if rotate_clabel:
        c = plt.colorbar(fraction=0.0452)
        plt.text(X[0,0] + (X[-1,-1] - X[0,0]) * (
            1.25 + .05 * (levels[-1] >= 10.)),
                 Y[0,0] + (Y[-1,-1] - Y[0,0]) * .94, clabel)
    else:
        plt.colorbar(fraction=0.0452,label=clabel)
    
    if xA is not None and yA is not None:
        plt.text(xA, yA, 'A',
                 ha='center',
                 va='center',
                 fontsize=20,
                 color='black',
                 path_effects=[pe.withStroke(linewidth=3, foreground="w")],
                 fontweight=750,
                 zorder=60)
    if yA is not None and yB is not None:
        plt.text(xB, yB, 'B',
                 ha='center',
                 va='center',
                 fontsize=20,
                 color='black',
                 path_effects=[pe.withStroke(linewidth=3, foreground="w")],
                 fontweight=750,
                 zorder=60)
    
    if P is not None:
        if X2 is None:
            X2 = X
            Y2 = Y
            P = P[drop:len(X)-drop,
                  drop:len(X[0])-drop]
        if len(rc_levels) > 8:
            plt.contour(X2,Y2,P,zorder=50,
                        colors='#cccccc', alpha=.84, linewidths=1.36,
                         levels=rc_levels[::2])
            c = plt.contour(X2,Y2,P,zorder=50,
                          colors='#cccccc', alpha=.84, linewidths=1.36,
                         levels=rc_levels[1::2])
        else:
            c = plt.contour(X2,Y2,P,zorder=50,
                          colors='#cccccc', alpha=.84, linewidths=1.36,
                         levels=rc_levels)
        if rc_labels:
            f=plt.clabel(c, colors='black')
            plt.setp(f, path_effects=[pe.withStroke(linewidth=3, foreground="w")])
        
    return figure, ax


def plot_1d_energy_profile(pathensemble,
                           reference,
                           nbins=100,
                           vmin=-np.inf,
                           vmax=+np.inf,
                           bootstrapping=0,
                           pathensemble_bootstrapping=500,
                           reference_bootstrapping=50,
                           reweight_while_bootstrapping=True,
                           states='AB',
                           reweight_parameters={},
                           offset=np.nan,
                           max_error=1.,
                           verbose=False):
    
    vmin = np.max([-30, vmin, np.min(np.concatenate(reference.values(reference.weights > 0)))])
    vmax = np.min([+30, vmax, np.max(np.concatenate(reference.values(reference.weights > 0)))])
    bins = np.linspace(vmin, vmax, nbins + 1)
    values = (bins[:-1] + bins[1:]) / 2
    
    # reference
    result, bootstrapping_results, *_ = (
        compute_energies_and_rates(
            reference, bins,
            bootstrapping=reference_bootstrapping,
            reweight_while_bootstrapping=False,
            verbose=verbose))
    
    F0 = -np.log(result)
    if reference_bootstrapping:
        F0_bootstrapping = -np.log(bootstrapping_results)
        F0_min = F0 - np.std(F0_bootstrapping, axis=0)
        F0_min = np.quantile(F0_bootstrapping, 0.025, axis=0)
        F0_max = F0 + np.std(F0_bootstrapping, axis=0)
        F0_max = np.quantile(F0_bootstrapping, 0.975, axis=0)
    else:
        F0_min = F0
        F0_max = F0

    # pe estimate
    result, bootstrapping_results, *_ = (
        compute_energies_and_rates(
            pathensemble, bins,
            bootstrapping=pathensemble_bootstrapping,
            reweight_while_bootstrapping=reweight_while_bootstrapping,
            states=states,
            reweight_parameters=reweight_parameters,
            verbose=verbose))
    
    F = -np.log(result)
    if pathensemble_bootstrapping:
        F_bootstrapping = -np.log(bootstrapping_results)
        F_min = F - np.std(F_bootstrapping, axis=0)
        F_min = np.quantile(F_bootstrapping, 0.025, axis=0)
        F_max = F + np.std(F_bootstrapping, axis=0)
        F_max = np.quantile(F_bootstrapping, 0.975, axis=0)
        k = ~np.isnan(F_max)
    else:
        F_min = F
        F_max = F
        k = ~np.isinf(F)
    
    values = values[k]
    F = F[k]
    F0 = F0[k]
    F_min = F_min[k]
    F0_min = F0_min[k]
    F0_max = F0_max[k]
    F_max = F_max[k]
        
    if not np.isnan(offset):
        center = np.argmin(np.abs(values-offset))
        F -= F0[center]
        F_min -= F0[center]
        F_max -= F0[center]
        F0_min -= F0[center]
        F0_max -= F0[center]
        F0 -= F0[center]
    
    # plot
    figure, (ax1, ax2) = plt.subplots(2,1,
                                      figsize=(4,2.7),
                                      sharex=True,
                                      gridspec_kw={'height_ratios': [3, 1]})
    color2 = 'forestgreen'
    color = plt.get_cmap('Dark2')(0)

    # uncertanties
    if np.sum(np.abs(F0_max - F0_min)):
        ax1.fill_between(values, F0_min, F0_max, color='black', alpha=.25)
        ax2.fill_between(values, F0_min-F0, F0_max-F0, color='black', alpha=.25)
    if np.sum(np.abs(F_max - F_min)):
        ax1.fill_between(values, F_min, F_max, color=color2, alpha=.4)
        ax2.fill_between(values, F_min-F0, F_max-F0, color=color2, alpha=.4)
    
    # estimates
    ax1.plot(values, F, '.', color=color, markersize=2.5)
    ax1.plot(values, F, '-', color=color, lw=2.5)
    ax1.plot(values, F0, ':', color='black')

    # errors
    ax2.plot(values, F - F0, '.', color=color, markersize=2.5)
    ax2.plot(values, F - F0, color=color, lw=2.5)
    ax2.plot(values, values * 0, ':', color='black')
    
    # fix axis
    ax1.grid()
    ax2.grid()
    ax2.set_xlabel('Reaction coordinate $\lambda$')
    ax1.set_ylabel('Free energy [$k_BT$]')
    ax2.set_ylabel('Est$-$true')
    if max_error >= 1.5:
        ax2.set_yticks([-1, 0., 1], ['$-$1','0','+1'])
    elif max_error >= .75:
        ax2.set_yticks([-.5, 0., .5], ['$-$0.5','0.0','+0.5'])
    elif max_error >= .25:
        ax2.set_yticks([-.2, 0., .2], ['$-$0.2','0.0','+0.2'])
    ax2.set_ylim(-max_error, max_error)
    plt.minorticks_off()
    plt.subplots_adjust(left=.2102,
                        bottom=.1958,
                        right=.9148,
                        top=.9373,
                        hspace=0)
        
    return figure, (ax1, ax2)


def project_on_grid(pathensemble, X, Y, f=lambda x:x, frames=False):
    Z = pathensemble.project([X[0, :], Y[:, 0]], f=f, frames=frames)
    Z /= np.sum(Z)
    return Z


def plot_2d_committor_estimate_vs_reference(grid_X, grid_Y, grid_V,
                                            grid_committor_estimate,
                                            grid_committor_relaxation,
                                            lambdaA,
                                            lambdaB,
                                            xA, yA, xB, yB, radius,
                                            potential_energy_levels,
                                            grid_committor_levels,
                                            exact_levels=None,
                                            error_threshold=.125,
                                            logit=False,
                                            rescale_error=True):
    figure = plt.figure(figsize=(4/1.12, 3/1.08))
    if logit is False:
        error = (expit(grid_committor_estimate) - 
                 expit(grid_committor_relaxation))
    else:
        error = grid_committor_estimate - grid_committor_relaxation
    
    if rescale_error:
        error /= (grid_committor_relaxation *
                  (1 - grid_committor_relaxation) * 4) ** .5
    error[np.isinf(error)] = np.nan
    error[error >= +error_threshold] = + error_threshold - 1e-9
    error[error <= -error_threshold] = - error_threshold + 1e-9
    error[grid_V > potential_energy_levels[-1]] = np.nan    
    # contours
    plt.gca().set_aspect('equal')
    plt.contourf(grid_X,
                 grid_Y,
                 error,
                 levels=np.linspace(-error_threshold, error_threshold, 11),
                 cmap='RdYlGn')
    plt.colorbar(fraction=0.0452)

    if logit:
        plt.text(grid_X[-1,-1] * 1.33, grid_Y[-1,-1] * 1.05,
                 '$\lambda-\\mathrm{logit}(p_B)$', ha='right')
    else:
        plt.text(grid_X[-1,-1] * 1.33, grid_Y[-1,-1] * 1.05,
                 '$\\mathrm{expit}(\lambda)-p_B$', ha='right')
    
    plt.contour(grid_X,
                grid_Y,
                grid_V,
                levels=potential_energy_levels,
                colors='#a0a0a0',
                linewidths=1,
                alpha=.64)
    
    # validity region
    grid_validity_region = grid_X * 0.
    grid_validity_region[grid_committor_estimate < lambdaA] = 1.
    grid_validity_region[grid_committor_estimate > lambdaB] = 1.
    plt.contourf(grid_X,
                 grid_Y,
                 grid_validity_region,
                 levels=[.5, 1],
                 colors='black',
                 alpha=0.25)
    
    # states
    circleA = plt.Circle((xA, yA),
                         radius,
                         ec='black',
                         fc=(1,1,1,1),
                         zorder=20)
    circleB = plt.Circle((xB, yB),
                         radius,
                         ec='black',
                         fc=(1,1,1,1),
                         zorder=20)
    plt.text(xA, yA, 'A',
             ha='center',
             va='center',
             fontsize=20,
             color='black',
             fontweight=750,
             zorder=22)
    plt.text(xB, yB, 'B',
             ha='center',
             va='center',
             fontsize=20,
             color='black',
             fontweight=750,
             zorder=22)
    ax = plt.gca()
    ax.add_patch(circleA)
    ax.add_patch(circleB)

    if exact_levels is None:
        exact_levels = grid_committor_levels
    plt.contour(grid_X, grid_Y, grid_committor_relaxation, zorder=-10,
                 colors='#cccccc', levels=exact_levels)
    plt.contour(grid_X, grid_Y, grid_committor_estimate, zorder=-10,
                 colors='#333333', levels=grid_committor_levels)
    
    plt.xlim(grid_X[0,0],grid_X[-1,-1])
    plt.ylim(grid_Y[0,0],grid_Y[-1,-1])
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_aspect(abs((xlim[-1]-xlim[0])/(ylim[-1]-ylim[0])))
    figure.subplots_adjust(
        left=.12,
        bottom=.15,
        right=.8,
        top=.92,
        hspace=0)
    figure.set_size_inches(4/1.12, 3/1.08)
    return figure, ax


def create_equilibrium_tpe():
    # TODO much easier if saved as xtc files. Keep like this for now.
    equilibrium = tps[:0]
    trajs = [f'equilibrium/{file}' for file in sorted(os.listdir('equilibrium'))
            if len(file) == 20 and file[:10] == 'transition'
            and file[-4:] == '.npy']
    for traj in tqdm(trajs[:4000], position=0):
        t = np.load(traj)
        equilibrium._update(
            trajectory_files = equilibrium.trajectory_files + [f'{len(equilibrium)}'],
            frame_trajectory_indices = np.append(
                equilibrium.frame_trajectory_indices,
                np.repeat(len(equilibrium), len(t))),
            frame_trajectory_positions = np.append(
                equilibrium.frame_trajectory_positions,
                np.arange(len(t))),
            frame_times = np.append(
                equilibrium.frame_times,
                np.arange(len(t))),
            frame_simulation_times = np.append(
                equilibrium.frame_simulation_times,
                np.arange(len(t))),
            frame_states = np.append(
                equilibrium.frame_states,
                ['A'] + ['R'] * (len(t) - 2) + ['B']),
            frame_descriptors = np.append(
                equilibrium.frame_descriptors,
                t, axis=0) if equilibrium.nframes else t,
            frame_values = np.append(
                equilibrium.frame_values,
                np.zeros(len(t))),
            frame_indices = np.append(
                equilibrium._PathEnsemble__frame_indices,
                np.arange(len(t)) + equilibrium.nframes),
            lengths = np.append(equilibrium.lengths, [len(t)]),
            weights = np.append(equilibrium.weights, [1.]),
            shooting_indices = np.append(equilibrium.shooting_indices, [0]),
            are_accepted = np.append(equilibrium.are_accepted, [True]))
        equilibrium.save('equilibrium/pe.h5')


def initialize_results(n=4000):
    return {'step_numbers': np.zeros(n, dtype=int) + np.nan,
            'kAB': np.zeros(n) + np.nan,
            'kBA': np.zeros(n) + np.nan,
            'kAB_max': np.zeros(n) + np.nan,
            'kAB_min': np.zeros(n) + np.nan,
            'kBA_max': np.zeros(n) + np.nan,
            'kBA_min': np.zeros(n) + np.nan,
            'times': np.zeros(n) + np.nan,
            'timesA': np.zeros(n) + np.nan,
            'timesB': np.zeros(n) + np.nan,
            'timesS': np.zeros(n) + np.nan,
            'timesT': np.zeros(n) + np.nan,
            'TP_number': np.zeros(n, dtype=int) + np.nan,
            'TP_length': np.zeros(n) + np.nan,
            'TP_length_min': np.zeros(n) + np.nan,
            'TP_length_max': np.zeros(n) + np.nan,
            'channel_differences': np.zeros(n) + np.nan}


def compute_average_tps_lenghts(tps, dt=1.):
    TP_length = []
    TP_length_max = []
    TP_length_min = []
    lengths = tps.internal_lengths * dt
    weights = tps.weights
    for i in tqdm(range(len(tps)), position=0):
        TP_length.append(
            np.average(lengths[:i + 1], weights=weights[:i + 1]))
        temp = []
        for bootstrapping_event in range(1000):
            k = np.random.choice(i + 1, i + 1)
            temp.append(np.average(lengths[k], weights=weights[k]))
        TP_length_max.append(np.quantile(temp, .975))
        TP_length_min.append(np.quantile(temp, .025))
    return (np.array(TP_length),
            np.array(TP_length_max),
            np.array(TP_length_min))

