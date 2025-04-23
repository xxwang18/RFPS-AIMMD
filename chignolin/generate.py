#####
##### TODO POLISH ONE SINGLE JOBID + REMOVE UNESSENTIAL
#####
"""
AIMMD Launcher Script

This script is responsible for launching an AIMMD simulation, either on a Slurm-based cluster or a local workstation. 

Main Features:
1. **Argument Parsing**:
   - Parses command-line arguments specifying the project directory, number of 
     steps, computational nodes, and Slurm options.
   - Supports an optional dependency argument to wait for a job to finish before 
     starting.

2. **Dependency Handling**:
   - If a dependency job ID is provided, the script waits for it to complete 
     before proceeding.
   - If running on a Slurm cluster, the dependency is handled using `afterany:<jobid>`.
   - If running locally, it checks for an existing process and waits until it 
     terminates.

3. **Project Directory Setup**:
   - Creates necessary directories (`shots`, `equilibriumA`, `equilibriumB`).
   - Removes old status files (`proceed.txt`, `jobids.txt`).

4. **Logging**:
   - Writes a log file (`manager.log`) to track execution progress.
   - Logs Slurm configuration options if applicable.

5. **Job Submission**:
   - If running on a Slurm cluster:
     - Reads Slurm options from a configuration file.
     - Generates and submits Slurm job scripts dynamically.
     - Uses `srun` to execute different components (shooters, equilibrators, manager).
   - If running on a local workstation:
     - Uses `nohup` to start jobs in the background.
     - Tracks process IDs.

6. **Job Execution**:
   - Assigns computational nodes to different tasks:
     - `shooter.py` for transition path sampling.
     - `equilibrium.py` for equilibrium simulations in state A and state B.
     - `manager.py` for coordinating the overall simulation.

7. **Finalizing**:
   - Logs all job IDs for reference.
   - Writes the simulation parameters to `manager.log`.
   - Prints instructions for manually submitting the job if needed.

"""


import os, argparse
from time import sleep
from textwrap import wrap
from datetime import datetime

PYTHON = 'python3'

# functions
def write(text, path=None, wrap_text=False):
    if wrap_text:
        text = "\n".join(wrap(text, 80,
            break_long_words=False, replace_whitespace=False))
    text = text.replace('"',"'")
    
    if path is None:
        os.system(f'''echo "{text}"''')
    else:
        os.system(f'''echo "{text}" >> {path}''')

# go!
parser = argparse.ArgumentParser(description='AIMMD launcher')
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
parser.add_argument('-s', '--slurm', type=str, default='',
    help='running on cluster with slurm options defined here')
parser.add_argument('-d', '--dependency', type=str,
    help='wait for job to terminate before starting')
args = parser.parse_args()
directory = args.directory
nsteps = args.nsteps
n = args.n
nA = args.nA
nB = args.nB
slurm = args.slurm
dependency = args.dependency
if dependency is not None:
    if slurm and dependency.isdigit():
        dependency = f'-d afterany:{dependency}' 
        '''
            slurm_command = f"sbatch {dependency} {slurm} my_script.sh"
            subprocess.run(slurm_command, shell=True)
        '''
    else:
        dependency = int(dependency)
        while True:
            try:
                os.kill(dependency, 0)  # Check if the process is running
                sleep(1)
            except OSError:
                break  # Process has finished
else:
    dependency = ''


# create original
for worker_id in range(n):
    os.system(f'mkdir {directory}/shots{worker_id}')
os.system(f'mkdir {directory}/equilibriumA')
os.system(f'mkdir {directory}/equilibriumB')
os.system(f'rm {directory}/proceed.txt')
os.system(f'rm {directory}/jobids.txt')

# load and copy params
logfile = f'{directory}/manager.log'
t0 = str(datetime.now())[:19]
write(f'''
######################## AIMMD RUN {t0} ########################
''', logfile)

# submit jobs
jobids = []
if len(slurm):  # on cluster
    slurm_options = open(slurm).read()
    exec(slurm_options)
    
    write(f'''
slurm options _________________________________________________________________
{slurm_options}
_______________________________________________________________________________
''', logfile)
    
    def submit_job(command):
        global jobids
        ID = os.popen(command).read()
        jobids.append(ID.split()[-1])

    ntasks = 1
    for line in slurm_options.split('\n'):
        if line[:16] == '#SBATCH --ntasks':
            ntasks = int(line.split('=')[1].split()[0])

    shooter_worker_id = 0
    equiliA_worker_id = 0
    equiliB_worker_id = 0
    total_tasks = 0
    finished = False
    fname = f'.{directory}_job.sh'
    while True:

        # reinitialize job
        if total_tasks == 0:
            file = open(fname, 'w')
            file.write(f'#!/bin/bash -x\n')
            file.write(f'#SBATCH --job-name={directory}')
            file.write(f'{workers_options}\n')
                
        if shooter_worker_id < n:  # shooter
            log = f'{directory}/shots{shooter_worker_id}/log'
            file.write(f'srun --exclusive --ntasks=1 '
                f'{PYTHON} shooter.py {directory} '
                f'{shooter_worker_id} -s >> {log} 2>&1 &\n')
            shooter_worker_id += 1
            total_tasks += 1
        elif equiliA_worker_id < nA:  # equiliA
            log = f'{directory}/equilibriumA/{equiliA_worker_id}.log'
            file.write(f'srun --exclusive --ntasks=1 '
                f'{PYTHON} equilibrium.py {directory} '
                f'{equiliA_worker_id} {nA} A -s >> {log} 2>&1 &\n')
            equiliA_worker_id += 1
            total_tasks += 1
        elif equiliB_worker_id < nB:  # equiliB
            log = f'{directory}/equilibriumB/{equiliB_worker_id}.log'
            file.write(f'srun --exclusive --ntasks=1 '
                f'{PYTHON} equilibrium.py {directory} '
                f'{equiliB_worker_id} {nB} B -s >> {log} 2>&1 &\n')
            equiliB_worker_id += 1
            total_tasks += 1
        else:  # manager
            log = f'{directory}/manager.log'
            file.write(f'srun --exclusive --ntasks=1 '
                f'{PYTHON} manager.py {directory} {nsteps} '
                f'{n} {nA} {nB} -s >> {log} 2>&1 &\n')
            total_tasks += 1
            finished = True
        
        # submit
        if total_tasks >= ntasks or finished:
            file.write('wait\n')
            file.close()
            #submit_job(f'sbatch {dependency} {fname}')
            write(f'\n\n{open(fname).read()}\n\n')
            total_tasks = 0
        
        if finished:
            break
    
    # TODO CORE HERE!
    write('####################################')
    write('DONE! Please submit:')
    write(f'sbatch {fname}')
    write('####################################')
    write('')

else:  # on workstation
    def submit_job(command):
        global jobids
        ID = str(int(os.popen(f'nohup {command} & echo $!').read()))
        jobids.append(ID.split()[-1])
    
    # shooter
    for worker_id in range(n):
        log = f'{directory}/shots{worker_id}/log'
        submit_job(f'python shooter.py {directory} {worker_id} '
                   f'>>{log} 2>&1')
    
    # equilibriumA
    for worker_id in range(nA):
        log = f'{directory}/equilibriumA/{worker_id}.log'
        submit_job(f'python equilibrium.py {directory} {worker_id} {nA} A '
                   f'>>{log} 2>&1')
    
    # equilibriumB
    for worker_id in range(nB):
        log = f'{directory}/equilibriumB/{worker_id}.log'
        submit_job(f'python equilibrium.py {directory} {worker_id} {nB} B '
                   f'>>{log} 2>&1')
    
    # manager
    log = f'{directory}/manager.log'
    submit_job(f'python manager.py {directory} {nsteps} {n} {nA} {nB} '
               f'>>{log} 2>&1')

write(f'''
params.py _____________________________________________________________________
{open(f'{directory}/params.py').read()}
_______________________________________________________________________________
''', logfile)

write(f'''
AIMMD run paramers ____________________________________________________________
directory = {directory}
nsteps = {nsteps}
n = {n}
nA = {nA}
nB = {nB}
slurm = {slurm}
dependency = {dependency}
_______________________________________________________________________________
''', logfile)

# write jobids
jobids = ' '.join(jobids)
write(f'''JOBIDS: {jobids}''', logfile, wrap_text=True)
write('''
Handling to manager ___________________________________________________________
''', logfile)
