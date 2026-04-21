#!/bin/bash
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --job-name=unet_test
#SBATCH --output=/store_new/mch/msclim/antoumos/R/develop/NOWPRECIP/new_project/logs/test_%j.out
#SBATCH --error=/store_new/mch/msclim/antoumos/R/develop/NOWPRECIP/new_project/logs/test_%j.err

# Activate correct environment
export PATH=/users/antoumos/.conda/envs/pysteps/bin:$PATH

# Confirm GPU is visible
nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Run test script
python /store_new/mch/msclim/antoumos/R/develop/NOWPRECIP/new_project/nowcast_05_test.py