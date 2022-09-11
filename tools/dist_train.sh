#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=29500

python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    ./tools/train.py $CONFIG  \
    --launcher pytorch ${@:3}
