#!/bin/bash
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --job-name=unet_test
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err

export PATH=$HOME/.conda/envs/pysteps/bin:$PATH

nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

python nowcast_05_test.py