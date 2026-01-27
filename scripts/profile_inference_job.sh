#!/bin/bash
#BSUB -q BatchGPU
#BSUB -n 4
#BSUB -R "rusage[mem=32000] span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J profile_inference
#BSUB -o /home/mima2416/repositories/PROVE/logs/profile_inference_%J.out
#BSUB -e /home/mima2416/repositories/PROVE/logs/profile_inference_%J.err
#BSUB -W 1:00

# Profile inference on MapillaryVistas and BDD10k to compare timing

cd /home/mima2416/repositories/PROVE
source ~/.bashrc
conda activate prove

echo "=========================================="
echo "GPU Inference Profiling"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "=========================================="

# Find a baseline checkpoint for profiling
CKPT_MV="/scratch/aaa_exchange/AWARE/WEIGHTS/baseline/mapillaryvistas/deeplabv3plus_r50/iter_80000.pth"
CKPT_BDD="/scratch/aaa_exchange/AWARE/WEIGHTS/baseline/bdd10k/deeplabv3plus_r50/iter_80000.pth"

echo ""
echo "=========================================="
echo "Profiling MapillaryVistas (66 classes)"
echo "=========================================="
python tools/profile_full_inference.py \
    --checkpoint "$CKPT_MV" \
    --dataset MapillaryVistas \
    --num-images 100 \
    --batch-size 4

echo ""
echo "=========================================="
echo "Profiling BDD10k (19 classes)"
echo "=========================================="
python tools/profile_full_inference.py \
    --checkpoint "$CKPT_BDD" \
    --dataset BDD10k \
    --num-images 100 \
    --batch-size 4

echo ""
echo "=========================================="
echo "Profiling complete!"
echo "Date: $(date)"
echo "=========================================="
