#!/bin/bash
# Submit additional training jobs for gen_albumentations_weather and gen_cyclediffusion
# - gen_albumentations_weather: IDD-AW (6 models) + MapillaryVistas (6 models) = 12 jobs
# - gen_cyclediffusion: IDD-AW (6 models) = 6 jobs
# Total: 18 jobs

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
echo "gen_albumentations_weather IDD-AW training jobs..."
echo "=========================================="
STRATEGY="gen_albumentations_weather"
DATASET="IDD-AW"
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
echo "gen_albumentations_weather MapillaryVistas training jobs..."
echo "=========================================="
STRATEGY="gen_albumentations_weather"
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
echo "gen_cyclediffusion IDD-AW training jobs..."
echo "=========================================="
STRATEGY="gen_cyclediffusion"
DATASET="IDD-AW"
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
echo "All $count additional training jobs submitted!"
echo "Use 'bjobs' to check job status."
