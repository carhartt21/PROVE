#!/bin/bash
# Submit training jobs for gen_VisualCloze strategy
# This strategy has all required data (BDD10k, IDD-AW, MapillaryVistas) but no trained models

cd /home/mima2416/repositories/PROVE

# Common settings
STRATEGY="gen_VisualCloze"
QUEUE="BatchGPU"
# GPU_TYPE removed - any GPU is acceptable
WALLTIME="24:00"
MEM="16G"

echo "Submitting training jobs for gen_VisualCloze..."
echo "Dataset images available:"
echo "  - BDD10k: 14,220"
echo "  - IDD-AW: 23,082"
echo "  - MapillaryVistas: 5,026"
echo ""

# Arrays of datasets and models
DATASETS=("BDD10k" "IDD-AW" "MapillaryVistas")
MODELS=("deeplabv3plus_r50" "deeplabv3plus_r50_clear_day" "pspnet_r50" "pspnet_r50_clear_day" "segformer_mit-b5" "segformer_mit-b5_clear_day")

count=0
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
echo "All $count gen_VisualCloze training jobs submitted!"
echo "Use 'bjobs' to check job status."
