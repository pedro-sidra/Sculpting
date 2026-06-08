#!/bin/bash
set -e

DIST_URL='auto'
DATASET=diss
CONFIG_BASE=pretrain-AM-spunet-sculpting
EXP_NAME_BASE="diss/pretrain-AM-spunet-sculpting"
WEIGHT=""
MACHINES=1
GPUS=2

export PYTHONPATH=/workspaces/Sculpting/Pointcept

TRAIN_COMMAND="scripts/train.sh -g 2 -d diss -c pretrain-AM-spunet-trimming-scheduling -n diss/pretrain-AM-spunet-trimming-scheduling"
bash $TRAIN_COMMAND

CONFIG="${CONFIG_BASE}75"
EXP_NAME="${EXP_NAME_BASE}_beta75"
TRAIN_COMMAND="scripts/train.sh -g $GPUS -d $DATASET  -c $CONFIG -n $EXP_NAME -w $WEIGHT"
bash $TRAIN_COMMAND

CONFIG_BASE=pretrain-AM-spunet-trimming
EXP_NAME_BASE="diss/pretrain-AM-spunet-trimming"
CONFIG="${CONFIG_BASE}"
EXP_NAME="${EXP_NAME_BASE}"
TRAIN_COMMAND="scripts/train.sh -g $GPUS -d $DATASET  -c $CONFIG -n $EXP_NAME -w $WEIGHT"
bash $TRAIN_COMMAND
