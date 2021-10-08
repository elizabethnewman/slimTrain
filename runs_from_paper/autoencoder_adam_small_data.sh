#!/bin/sh
export PYTHONPATH=".."

num_epochs=50
alpha1=1e-10
alpha2=1e-10
num_train=50000
num_val=10000
num_test=10000
log_interval=10
batch_size=32

for seed in 0 10 20 30 40 50 60 70 80 90
do
    python run_autoencoder_mnist_adam.py --num-epoch $num_epochs --batch-size $batch_size --num-train $num_train --num-val $num_val --num-test $num_test --log-interval $log_interval --alpha1 $alpha1 --alpha2 $alpha2 --seed $seed --save
done


