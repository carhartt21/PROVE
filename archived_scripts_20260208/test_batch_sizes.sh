#!/bin/bash
# Experiment with different batch sizes for MapillaryVistas testing

CHECKPOINT="${AWARE_DATA_ROOT}/WEIGHTS/baseline/mapillaryvistas/deeplabv3plus_r50/iter_80000.pth"
CONFIG="${AWARE_DATA_ROOT}/WEIGHTS/baseline/mapillaryvistas/deeplabv3plus_r50/training_config.py"
BASE_OUTPUT="${AWARE_DATA_ROOT}/WEIGHTS/baseline/mapillaryvistas/deeplabv3plus_r50/test_batch_exp"

BATCH_SIZE=$1
JOB_NAME="mapillary_batch_${BATCH_SIZE}"

bsub -J "$JOB_NAME" \
     -q BatchGPU \
     -R "rusage[mem=24000,ngpus_physical=1]" \
     -R "select[ngpus>0]" \
     -gpu "num=1:mode=exclusive_process" \
     -W 360 \
     -o "${HOME}/repositories/PROVE/logs/${JOB_NAME}_%J.log" \
     -e "${HOME}/repositories/PROVE/logs/${JOB_NAME}_%J.err" \
     "cd ${HOME}/repositories/PROVE && source ~/.bashrc && conda activate prove && python fine_grained_test.py --config $CONFIG --checkpoint $CHECKPOINT --output-dir ${BASE_OUTPUT}_batch${BATCH_SIZE} --dataset MapillaryVistas --batch-size $BATCH_SIZE"

echo "Submitted job for batch_size=$BATCH_SIZE"
