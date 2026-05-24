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

CONFIG="${CONFIG_BASE}"
EXP_NAME="${EXP_NAME_BASE}"
TRAIN_COMMAND="scripts/train.sh -g $GPUS -d $DATASET  -c $CONFIG -n $EXP_NAME -w $WEIGHT"
bash $TRAIN_COMMAND

CONFIG="${CONFIG_BASE}01"
EXP_NAME="${EXP_NAME_BASE}_beta01"
TRAIN_COMMAND="scripts/train.sh -g $GPUS -d $DATASET  -c $CONFIG -n $EXP_NAME -w $WEIGHT"
bash $TRAIN_COMMAND

CONFIG="${CONFIG_BASE}25"
EXP_NAME="${EXP_NAME_BASE}_beta25"
TRAIN_COMMAND="scripts/train.sh -g $GPUS -d $DATASET  -c $CONFIG -n $EXP_NAME -w $WEIGHT"
bash $TRAIN_COMMAND