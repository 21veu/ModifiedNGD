mkdir -p ./log

n=2
CUDA_VISIBLE_DEVICES=$n python train_H.py >./log/htru2_o$n.out 2>&1 &

# sh ./main.sh >./log/$$.out 2>&1 &
# ps -ef|grep 27057
# less ./log/a.out 