#!/usr/bin/env sh
if [ "$TORCH_HOME" = "" ]; then
  echo "Must set TORCH_HOME envoriment variable for data dir saving"
  exit 1
else
  echo "TORCH_HOME : $TORCH_HOME"
fi
model=caffe_cifar
dataset=cifar10
epochs=180

$PYTHON main.py $TORCH_HOME/cifar.python \
        --dataset ${dataset} --arch ${model} --save_path ./snapshots/${dataset}_${model}_${epochs} --epochs ${epochs} \
        --schedule 120 150 --gammas 0.1 0.1 --learning_rate 0.01 --decay 0.004 --batch_size 128 --workers 2 --ngpu 1
