#!/bin/bash
n=66
title='_semi_HN_05_14_D1em1_Dmax1em0_r01_w1em4'
perturb_rate=1.1


mpirun -n $n --hostfile hostfile --mca btl_tcp_if_include ib0 python script_BFGS.py \
--title $title \

>./log/Fields$SGD_indicator$title.out 2>&1 
 

