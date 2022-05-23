#!/bin/bash

port=$(python get_free_port.py)
GPU=2

alias exp="python -m torch.distributed.launch --master_port=${port} --nproc_per_node=${GPU} tools/train_incremental.py"
shopt -s expand_aliases

# FIRST STEP
# python -m torch.distributed.launch --nproc_per_node=${GPU} tools/train_first_step.py -c configs/OD_cfg/e2e_faster_rcnn_R_50_C4_4x.yaml

# INCREMENTAL STEPS
#task=10-2
#for s in {4..5}; do
#  exp -t ${task} -n MMA --rpn --uce --dist_type uce --cls 1 -s $s
#  echo Done
#done
##
#task=15-1
#for s in {4..5}; do
##  exp -t ${task} -n ILOD -s $s --cls 1.
#  exp -t ${task} -n MMA --rpn --uce --dist_type uce --cls 1 -s $s
#  echo Done
#done
#
#task=10-5
#for s in 1 2; do
#  exp -t ${task} -n MMA --rpn --uce --dist_type uce --cls 0.5 -s $s
#  echo Done
#done

task=10-1
for s in {1..10}; do
  exp -t ${task} -n MMA --rpn --uce --dist_type uce --cls 1 -s $s
done