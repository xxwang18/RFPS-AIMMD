# sbatch options
workers_options = """
#SBATCH --partition=booster
#SBATCH --account=aimmdmemb
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=1
#SBATCH --mail-type=FAIL
#SBATCH --time=22:59:00
"""
manager_options = workers_options
