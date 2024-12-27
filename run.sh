#!/bin/bash
#SBATCH --job-name=medical-chatbot
#SBATCH --partition=localhost
#SBATCH --time=23:59:00
#SBATCH --output=slurm_output.txt
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=40000M

python src/training/train.py
echo "Test job" 
echo "Job finished"
exit 0