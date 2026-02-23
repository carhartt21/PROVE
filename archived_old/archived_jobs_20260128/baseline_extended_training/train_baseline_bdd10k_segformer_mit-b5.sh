#!/bin/bash
#BSUB -J ext_baseline_bdd10k_segformer
#BSUB -o ${HOME}/repositories/PROVE/logs/ext_baseline_bdd10k_segformer_%J.out
#BSUB -e ${HOME}/repositories/PROVE/logs/ext_baseline_bdd10k_segformer_%J.err
#BSUB -q BatchGPU
#BSUB -W 48:00
#BSUB -n 4
#BSUB -R "rusage[mem=32000]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process:gmem=24G"

# Baseline Extended Training: BDD10k / segformer_mit-b5
# Stage 2 (no domain filter - all domains)
# Resume from iter_80000 → train to iter_320000
# Saves checkpoints every 10k iterations

source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate prove

cd ${HOME}/repositories/PROVE

# Configuration
DATASET="BDD10k"
MODEL="segformer_mit-b5"
STRATEGY="baseline"
OUTPUT_DIR="${AWARE_DATA_ROOT}/WEIGHTS_EXTENDED/baseline/bdd10k/segformer_mit-b5"
RESUME_CKPT="${AWARE_DATA_ROOT}/WEIGHTS_STAGE_2/baseline/bdd10k/segformer_mit-b5/iter_80000.pth"
MAX_ITERS=320000

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Baseline Extended Training"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo "Strategy: $STRATEGY"
echo "Output: $OUTPUT_DIR"
echo "Resume from: $RESUME_CKPT"
echo "Max iterations: $MAX_ITERS"
echo "=========================================="

# Run training with resume
# Stage 2 = no domain filter (train on all domains)
python unified_training.py \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --strategy "$STRATEGY" \
    --work-dir "$OUTPUT_DIR" \
    --resume-from "$RESUME_CKPT" \
    --max-iters "$MAX_ITERS"

echo "Training complete!"
