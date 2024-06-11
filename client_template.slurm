#!/bin/bash
#SBATCH -J 21yui_client
#SBATCH -p kshdexclu04
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task 8
#SBATCH --gres=dcu:4
#SBATCH --mem 90G
#SBATCH --exclude=e04r1n03
##SBATCH --ntasks-per-node=1


date
export PYTHONPATH=/public/home/ssct004t/project/21yui/ng
module switch compiler/rocm/4.0.1
mpirun python mlp.py 
date
