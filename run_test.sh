#!/bin/bash
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --job-name=nowcast_test
#SBATCH --output=/users/antoumos/nowcast/logs/test_%j.out
#SBATCH --error=/users/antoumos/nowcast/logs/test_%j.err

# Activate correct environment
export PATH=/users/antoumos/.conda/envs/pysteps/bin:$PATH

# Confirm GPU is visible
nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Run test script
python /store_new/mch/msclim/antoumos/R/develop/NOWPRECIP/new_project/nowcast_03_train.py