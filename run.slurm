#!/bin/bash
#SBATCH -J Optics_yui
#SBATCH -p GPU
#SBATCH --mem=64G
#SBATCH -N 2
#SBATCH -n 4
#SBATCH --nodelist=node[1-2]

mkdir -p ./log

# sleep 30s
# sh client.sh
mpirun -n 66 --hostfile hostfile --mca btl_tcp_if_include ib0 python forward_trace_main.py
