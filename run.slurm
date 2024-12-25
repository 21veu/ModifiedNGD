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
n_epochs=1000
lr=0.01
batch_size=1024
perturbmode=original
sigma=2
u=2

if [ "$perturbmode" == "noise" ];then
    title="Test_${dataname}_${perturbmode}_sigma${sigma}"
elif [ "$perturbmode" == "condition" ]; then
    title="Test_${dataname}_${perturbmode}_u${u}"
else
    title="Test_${dataname}_${perturbmode}"
fi



CUDA_VISIBLE_DEVICES=0 python train.py \
--date $mydate \
--device $device \
--data_name $dataname \
--n_epochs $n_epochs \
--lr $lr \
--batch_size $batch_size \
--perturb_mode $perturbmode \
--sigma $sigma \
--u $u \
--NGD \
>./log/"${title}_${mydate}".out 2>&1 

