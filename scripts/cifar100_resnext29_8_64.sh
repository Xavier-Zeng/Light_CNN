#!/usr/bin/env sh
if [ "$TORCH_HOME" = "" ]; then
  echo "Must set TORCH_HOME envoriment variable for data dir saving"
  exit 1
else
  echo "TORCH_HOME : $TORCH_HOME"
fi
model=resnext29_8_64
dataset=cifar100
epochs=300

$PYTHON main.py $TORCH_HOME/cifar.python \
	--dataset ${dataset} --arch ${model} --save_path ./snapshots/${dataset}_${model}_${epochs} --epochs ${epochs} --learning_rate 0.1 \
	--schedule 150 225 --gammas 0.1 0.1 --batch_size 64 --workers 16 --ngpu 8
