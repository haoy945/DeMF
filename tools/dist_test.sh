#!/usr/bin/env bash


CONFIG=$1
CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=29501

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/test.py --config $CONFIG --checkpoint $CHECKPOINT --eval mAP \
    --launcher pytorch ${@:4}
