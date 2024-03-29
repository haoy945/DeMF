#!/usr/bin/env bash

CONFIG=configs/baseline/votenet.py  # path to the config file
GPUS=8
PORT=29500

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/../train.py $CONFIG --launcher pytorch ${@:3}
