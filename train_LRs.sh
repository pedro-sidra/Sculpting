#!/bin/bash
set -e

DIST_URL='auto'
DATASET=scannet
CONFIG_BASE=semseg-spunet-sidra-efficient-lr
EXP_NAME_BASE="diss/FT_trimming_01"
WEIGHT="pedrosidra/diss/zpb1wjgw-model_last:v19"
MACHINES=1
GPUS=2

export PYTHONPATH=/workspaces/Sculpting/Pointcept
export GLOO_SOCKET_IFNAME=enp36s0f1
export NCCL_SOCKET_IFNAME=enp36s0f1

for LR in 1 5 10 20 100
do
    CONFIG="${CONFIG_BASE}${LR}"
    EXP_NAME="${EXP_NAME_BASE}${LR}"
    TRAIN_COMMAND="scripts/train.sh -g $GPUS -d $DATASET  -c $CONFIG -n $EXP_NAME -w $WEIGHT"
    bash $TRAIN_COMMAND
done
