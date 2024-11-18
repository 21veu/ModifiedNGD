#!/bin/bash
#SBATCH --job-name=optic_W
#SBATCH --output=log/output_%j_test1_u=100.txt
#SBATCH --error=log/error_%j.txt
#SBATCH --ntasks=1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4090:2
#SBATCH --time=01:00:00
#SBATCH --partition=GPU

sleep 5s 
