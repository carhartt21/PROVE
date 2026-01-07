#!/bin/bash
# Submit training jobs for gen_flux_kontext MapillaryVistas only
# Only MapillaryVistas has data (45,162 images), BDD10k and IDD-AW have 0 images

cd /home/mima2416/repositories/PROVE

# Common settings
QUEUE="BatchGPU"
GPU_TYPE="A100"
WALLTIME="24:00"
MEM="16G"

count=0

# Arrays of models
MODELS=("deeplabv3plus_r50" "deeplabv3plus_r50_clear_day" "pspnet_r50" "pspnet_r50_clear_day" "segformer_mit-b5" "segformer_mit-b5_clear_day")

echo "=========================================="
echo "gen_flux_kontext MapillaryVistas training jobs..."
echo "=========================================="
STRATEGY="gen_flux_kontext"
DATASET="MapillaryVistas"
for MODEL in "${MODELS[@]}"; do
    JOB_NAME="prove_${DATASET}_${MODEL}_${STRATEGY}"
    bsub -q $QUEUE -gpu "num=1:j_exclusive=yes:gmem=${MEM}:gmodel=${GPU_TYPE}" \
         -W $WALLTIME \
         -J "$JOB_NAME" \
         -o "logs/${JOB_NAME}_%J.log" \
         -e "logs/${JOB_NAME}_%J.err" \
         "source ~/.bashrc && conda activate prove && python unified_training.py --strategy $STRATEGY --datasets $DATASET --architectures $MODEL --work-dir /scratch/aaa_exchange/AWARE/WEIGHTS"
    echo "Submitted: $JOB_NAME"
    ((count++))
done

echo ""
echo "=========================================="
echo "All $count gen_flux_kontext MapillaryVistas training jobs submitted!"
echo "Use 'bjobs' to check job status."
