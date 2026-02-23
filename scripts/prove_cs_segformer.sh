#!/bin/bash
#BSUB -J prove_cs_segformer_b3_bs16
#BSUB -q BatchGPU
#BSUB -o ${AWARE_DATA_ROOT}/WEIGHTS_CITYSCAPES/baseline/cityscapes/segformer_mit-b3/train_%J.out
#BSUB -e ${AWARE_DATA_ROOT}/WEIGHTS_CITYSCAPES/baseline/cityscapes/segformer_mit-b3/train_%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=48000]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process:gmem=24G"
#BSUB -W 12:00

source /home/chge7185/.bashrc
mamba activate prove
cd /home/chge7185/repositories/PROVE

echo "=============================================="
echo "PROVE Training: Cityscapes / segformer_mit-b3 / baseline"
echo "Batch size: 16 (PROVE default)"
echo "Max iterations: 20000 (equivalent to ~320k samples)"
echo "Checkpoint/Validation interval: 2000"
echo "FIX APPLIED: Validation uses full resolution (2048x1024)"
echo "=============================================="

# Create output directory
mkdir -p ${AWARE_DATA_ROOT}/WEIGHTS_CITYSCAPES/baseline/cityscapes/segformer_mit-b3

# Run training with custom iterations
python unified_training.py \
    --dataset Cityscapes \
    --model segformer_mit-b3 \
    --strategy baseline \
    --max-iters 20000 \
    --checkpoint-interval 2000 \
    --eval-interval 2000 \
    --work-dir "${AWARE_DATA_ROOT}/WEIGHTS_CITYSCAPES/baseline/cityscapes/segformer_mit-b3"

echo "Training completed"
