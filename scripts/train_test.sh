#!/bin/bash
#SBATCH -N 1
#SBATCH -C knl
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --time=00:03:00
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1

##SBATCH --mail-type=ALL
#SBATCH --output=%j-%x.%u.out
#SBATCH --job-name=gta5cityscapes

module load cgpu pytorch/1.7.1
## conda activate env_name

srun python ../trainval.py -e gta5_seg -sb ./results/gta5cityscapes/ -d ./data/Segmentation/ -r 1 -da 0 -ds 1 -ss None -ne 5
