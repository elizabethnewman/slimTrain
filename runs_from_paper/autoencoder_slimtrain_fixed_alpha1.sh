#!/bin/sh
export PYTHONPATH=".."

num_epochs=50
alpha1=1e-10
num_train=50000
num_val=10000
num_test=10000
log_interval=10
batch_size=32
mem_depth=5
seed=20

for alpha2 in 1e-10 1e-1 1e0
do
    python run_autoencoder_mnist_slimtrain.py --num-epoch $num_epochs --batch-size $batch_size --num-train $num_train --num-val $num_val --num-test $num_test --log-interval $log_interval --alpha1 $alpha1 --alpha2 $alpha2 --mem-depth $mem_depth --sum-lambda $alpha2 --seed $seed --save
done


