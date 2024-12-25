#!/bin/bash
mkdir -p ./log

mydate=$(date +"%Y_%m_%d_%H_%M_%S")

device=cuda
dataname=htru2
n_epochs=300
lr=0.1
batch_size=128
perturb_mode=original
sigma=1

# conda init
# conda activate ng

python train.py \
--date $mydate \
--device $device \
--data_name $dataname \
--n_epochs $n_epochs \
--lr $lr \
--batch_size $batch_size \
--perturb_mode $perturb_mode \
--sigma $sigma \
--MNGD \
>./log/$dataname_$perturb_mode_$sigma_$mydate.out 2>&1 
 

