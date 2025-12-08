#!/bin/bash

# Activate virtual environment
source venv/bin/activate

$DATASET = acdc
$MODEL = deeplabv3plus_r50
$STRATEGY = baseline
# Set job name
jobname="train_${DATASET}_${MODEL}_${STRATEGY}"

# Submit job to LSF
bsub -gpu "num=1:mode=exclusive_process:gmem=24G" \
    -q BatchGPU \
    -R "span[hosts=1]" \
    -n 8 \
    -oo "lsf_${jobname}_%J_gpu.log" \
    -eo "lsf_${jobname}_%J_gpu.err" \
    -L /bin/bash \
    -J "${jobname}" \
    "./train_unified.sh single --dataset ${DATASET} --model ${MODEL} --strategy ${STRATEGY}"