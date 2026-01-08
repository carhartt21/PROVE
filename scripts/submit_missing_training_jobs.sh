#!/bin/bash
# Submit training jobs for missing BDD10k and full strategies
# - gen_albumentations_weather: BDD10k (6 models)
# - gen_cyclediffusion: BDD10k (6 models)
# - gen_step1x_v1p2: All datasets (18 models)

cd /home/mima2416/repositories/PROVE

# Common settings
QUEUE="BatchGPU"
# GPU_TYPE removed - any GPU is acceptable
WALLTIME="24:00"
MEM="16G"

count=0

# Arrays of models
MODELS=("deeplabv3plus_r50" "deeplabv3plus_r50_clear_day" "pspnet_r50" "pspnet_r50_clear_day" "segformer_mit-b5" "segformer_mit-b5_clear_day")

echo "=========================================="
echo "Submitting gen_albumentations_weather BDD10k training jobs..."
echo "=========================================="
STRATEGY="gen_albumentations_weather"
DATASET="BDD10k"
for MODEL in "${MODELS[@]}"; do
    JOB_NAME="prove_${DATASET}_${MODEL}_${STRATEGY}"
    bsub -q $QUEUE -gpu "num=1:j_exclusive=yes:gmem=${MEM}" \
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
echo "Submitting gen_cyclediffusion BDD10k training jobs..."
echo "=========================================="
STRATEGY="gen_cyclediffusion"
DATASET="BDD10k"
for MODEL in "${MODELS[@]}"; do
    JOB_NAME="prove_${DATASET}_${MODEL}_${STRATEGY}"
    bsub -q $QUEUE -gpu "num=1:j_exclusive=yes:gmem=${MEM}" \
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
echo "Submitting gen_step1x_v1p2 all datasets training jobs..."
echo "=========================================="
STRATEGY="gen_step1x_v1p2"
DATASETS=("BDD10k" "IDD-AW" "MapillaryVistas")
for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        JOB_NAME="prove_${DATASET}_${MODEL}_${STRATEGY}"
        bsub -q $QUEUE -gpu "num=1:j_exclusive=yes:gmem=${MEM}" \
             -W $WALLTIME \
             -J "$JOB_NAME" \
             -o "logs/${JOB_NAME}_%J.log" \
             -e "logs/${JOB_NAME}_%J.err" \
             "source ~/.bashrc && conda activate prove && python unified_training.py --strategy $STRATEGY --datasets $DATASET --architectures $MODEL --work-dir /scratch/aaa_exchange/AWARE/WEIGHTS"
        echo "Submitted: $JOB_NAME"
        ((count++))
    done
done

echo ""
echo "=========================================="
echo "All $count training jobs submitted!"
echo "Use 'bjobs' to check job status."
