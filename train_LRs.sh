#!/bin/bash
set -e

DIST_URL='auto'
DATASET=scannet
CONFIG_BASE=semseg-litept-sidra-efficient-lr
EXP_NAME_BASE="diss/FT-sonata-litept-full"
MACHINES=1
GPUS=4
WEIGHT="pedrosidra/diss/3pqn3zzl-model_last:v19"

export PYTHONPATH=/workspaces/Sculpting/Pointcept
# export CUDA_VISIBLE_DEVICES=0,1,2,3

# TRAIN_COMMAND="scripts/train.sh -g 4 -d scannet -c pretrain-sonatalike-spunet -n diss/pretrain-sonatalike-normals"
# bash $TRAIN_COMMAND


for LR in 1 5 10 20 100
do
    CONFIG="${CONFIG_BASE}${LR}"
    EXP_NAME="${EXP_NAME_BASE}${LR}"
    TRAIN_COMMAND="scripts/train.sh -g $GPUS -d $DATASET  -c $CONFIG -n $EXP_NAME -w $WEIGHT"
    bash $TRAIN_COMMAND
done
