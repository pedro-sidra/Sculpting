#!/bin/sh
set -e

cd $(dirname $(dirname "$0")) || exit
ROOT_DIR=$(pwd)
PYTHON=python

TRAIN_CODE=train.py

DATASET=scannet
CONFIG="None"
EXP_NAME="None"
WEIGHT="None"
RESUME=false
ARTIFACT="None"
NUM_GPU=None
NUM_MACHINE=1
DIST_URL="auto"

# fuser -k $MASTER_PORT/tcp


while getopts "p:d:c:n:w:g:m:r:a" opt; do
  case $opt in
    p)
      PYTHON=$OPTARG
      ;;
    d)
      DATASET=$OPTARG
      ;;
    c)
      CONFIG=$OPTARG
      ;;
    n)
      EXP_NAME=$OPTARG
      ;;
    w)
      WEIGHT=$OPTARG
      ;;
    r)
      RESUME=$OPTARG
      ;;
    g)
      NUM_GPU=$OPTARG
      ;;
    m)
      NUM_MACHINE=$OPTARG
      ;;
    a)
      ARTIFACT=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG"
      ;;
  esac
done

if [ "${EXP_NAME}" = 'None' ]
then
  EXP_NAME=$CONFIG
fi

# if [ "${NUM_GPU}" = 'None' ]
# then
#   NUM_GPU=`$PYTHON -c 'import torch; print(torch.cuda.device_count())'`
# fi

if [ "${ARTIFACT}" != 'None' ]
then
  wandb artifact get $ARTIFACT
  ln -sbf ~+/artifacts/$(basename -- $ARTIFACT) data/$DATASET #|| echo "WARNING: not using dataset artifact $ARTIFACT because file data/$DATASET exists"
fi

echo "Experiment name: $EXP_NAME"
echo "Python interpreter dir: $PYTHON"
echo "Dataset: $DATASET"
echo "Config: $CONFIG"
echo "GPU Num: $NUM_GPU"
echo "Machine Num: $NUM_MACHINE"

# if [ -n "$DIST_URL" ]; then
#   DIST_URL="auto"
# fi
# if [ -n "$SLURM_NODELIST" ]; then
#   # MASTER_HOSTNAME=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
#   # MASTER_ADDR=$(getent hosts "$MASTER_ADDR" | awk '{ print $1 }')
#   MASTER_PORT=$((23450 + 0x$(echo -n "${DATASET}/${EXP_NAME}" | md5sum | cut -c 1-4 | awk '{print $1}') % 6))
#   DIST_URL=tcp://$MASTER_ADDR:$MASTER_PORT
# fi

# echo $DIST_URL $SLURM_PROCID $SLURM_JOBID $SLURM_NODEID > /workspaces/Pointcept/worker_log
echo "Dist URL: $DIST_URL"
echo "MASTER_PORT"=$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE
echo "MASTER_ADDR="$MASTER_ADDR
echo "nodeid=$SLURM_NODEID"

EXP_DIR=exp/${DATASET}/${EXP_NAME}
MODEL_DIR=${EXP_DIR}/model
CODE_DIR=${EXP_DIR}/code
CONFIG_DIR=configs/${DATASET}/${CONFIG}.py


echo " =========> CREATE EXP DIR <========="
echo "Experiment dir: $ROOT_DIR/$EXP_DIR"
if [ "${RESUME}" = true ] && [ -d "$EXP_DIR" ]
then
  CONFIG_DIR=${EXP_DIR}/config.py
  WEIGHT=$MODEL_DIR/model_last.pth
else
  RESUME=false
  mkdir -p "$MODEL_DIR" "$CODE_DIR"
  cp -r scripts tools "$CODE_DIR"
fi

echo "Loading config in:" $CONFIG_DIR
# export PYTHONPATH=./$CODE_DIR
echo "Running code in: $CODE_DIR"


echo " =========> RUN TASK <========="

if [ "${WEIGHT}" = "None" ]
then
     $PYTHON "$PWD"/tools/$TRAIN_CODE \
    --config-file "$CONFIG_DIR" \
    --num-gpus "$NUM_GPU" \
    --num-machines "$NUM_MACHINE" \
    --machine-rank ${SLURM_NODEID:-0} \
    --dist-url ${DIST_URL} \
    --options save_path="$EXP_DIR"
else
     $PYTHON "$PWD"/tools/$TRAIN_CODE \
    --config-file "$CONFIG_DIR" \
    --num-gpus "$NUM_GPU" \
    --num-machines "$NUM_MACHINE" \
    --machine-rank ${SLURM_NODEID:-0} \
    --dist-url ${DIST_URL} \
    --options save_path="$EXP_DIR" resume="$RESUME" weight="$WEIGHT"
fi
