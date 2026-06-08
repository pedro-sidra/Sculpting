#!/bin/bash
set -e

DIST_URL='auto'
DATASET=scannet
CONFIG_BASE=semseg-spunet-sidra-efficient-lr
EXP_NAME_BASE="diss/FT-spunet-enc-sonata-lr"
MACHINES=1
GPUS=4
WEIGHT="pedrosidra/debug/leq4cgk7-model_last:v19"

export PYTHONPATH=/workspaces/Sculpting/Pointcept

# TRAIN_COMMAND="scripts/train.sh -g 8 -d scannet -c pretrain-sonata-litept -n debug/pretrain-sonata-litept"
# bash $TRAIN_COMMAND


for LR in 1 5 10 20 100
do
    CONFIG="${CONFIG_BASE}${LR}"
    EXP_NAME="${EXP_NAME_BASE}${LR}"
    TRAIN_COMMAND="scripts/train.sh -g $GPUS -d $DATASET  -c $CONFIG -n $EXP_NAME -w $WEIGHT"
    bash $TRAIN_COMMAND
done
