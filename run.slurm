#!/bin/bash
#SBATCH -J yui_train
#SBATCH -p GPU
#SBATCH --mem=0
#SBATCH -N 1
#SBATCH --nodelist=node1

# conda init
# conda activate ng

mkdir -p ./log

mydate=$(date +"%Y_%m_%d_%H_%M_%S")

device=cuda
dataname=synthetic
n_epochs=100
batch_size=256
perturbmode=original
sigma=5
u=0.7


CUDA_VISIBLE_DEVICES=1 python train.py \
--date $mydate \
--device $device \
--data_name $dataname \
--n_epochs $n_epochs \
--batch_size $batch_size \
--perturb_mode $perturbmode \
--sigma $sigma \
--u $u \
--MNGD \
>./log/aaatest$dataname$perturbmode$sigma$mydate.out 2>&1 
 
