#!/bin/bash

port=$(python get_free_port.py)
GPU=2

alias exp="python -m torch.distributed.launch --master_port=${port} --nproc_per_node=${GPU} tools/train_incremental.py --ist"
shopt -s expand_aliases

# FIRST STEP
# python -m torch.distributed.launch --nproc_per_node=${GPU} tools/train_first_step.py -c configs/OD_cfg/e2e_faster_rcnn_R_50_C4_4x.yaml

task=15-5
exp -t ${task} -n MMA --rpn --uce --dist_type uce --cls 1 --mask 0.
exp -t ${task} -n MMA_plus --rpn --uce --dist_type uce --cls 1 --mask 0.5

task=19-1
exp -t ${task} -n MMA --rpn --uce --dist_type uce --cls 1 --mask 0.
exp -t ${task} -n MMA_plus --rpn --uce --dist_type uce --cls 1 --mask 0.5


