#!/bin/bash
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --job-name=unet_train
#SBATCH --output=/store_new/mch/msclim/antoumos/R/develop/NOWPRECIP/new_project/logs/train_%j.out
#SBATCH --error=/store_new/mch/msclim/antoumos/R/develop/NOWPRECIP/new_project/logs/train_%j.err

export PATH=/users/antoumos/.conda/envs/pysteps/bin:$PATH

nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

python /store_new/mch/msclim/antoumos/R/develop/NOWPRECIP/new_project/nowcast_04_train_all.py