#!/usr/bin/env sh
if [ "$#" -ne 1 ] ;then                      
  echo "Input illegal number of parameters " $#
  echo "Need 1 parameters for the model name"
  exit 1        
fi
if [ "$TORCH_HOME" = "" ]; then
  echo "Must set TORCH_HOME envoriment variable for data dir saving"
  exit 1
else
  echo "TORCH_HOME : $TORCH_HOME"
fi
model=$1

$PYTHON imagenet_train.py $TORCH_HOME/ILSVRC2012 --arch ${model} -j 36 \
		 --save_dir ./snapshots/ImageNet/${model}
