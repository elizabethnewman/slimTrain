#!/bin/sh
export PYTHONPATH=".."

num_epochs=50
alpha=1e-10
num_val=10000
num_test=10000
log_interval=10
batch_size=32
mem_depth=5

for num_train in 128 256 512 1024 2048 4096 8192 16384
do
  for seed in 0 10 20 30 40 50 60 70 80 90
  do
      python run_autoencoder_mnist_slimtrain.py --num-epoch $num_epochs --batch-size $batch_size --num-train $num_train --num-val $num_val --num-test $num_test --log-interval $log_interval --alpha1 $alpha --alpha2 $alpha --mem-depth $mem_depth --sum-lambda $alpha --seed $seed --save
  done
done


