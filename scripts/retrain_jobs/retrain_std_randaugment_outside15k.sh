#!/bin/bash
#BSUB -J retrain_std_randaugment_outside15k
#BSUB -o /home/mima2416/repositories/PROVE/logs/retrain/retrain_std_randaugment_outside15k_%J.out
#BSUB -e /home/mima2416/repositories/PROVE/logs/retrain/retrain_std_randaugment_outside15k_%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=16000]"
#BSUB -gpu "num=1:mode=exclusive_process:gmem=20G"
#BSUB -W 12:00
#BSUB -q BatchGPU

# Activate environment
source ~/.bashrc
conda activate prove

cd /home/mima2416/repositories/PROVE

echo "========================================"
echo "Retraining job: retrain_std_randaugment_outside15k"
echo "Strategy: std_randaugment"
echo "Dataset: outside15k"
echo "Started: $(date)"
echo "========================================"


echo "----------------------------------------"
echo "Training: outside15k/deeplabv3plus_r50"
echo "Strategy: std_randaugment"
echo "Real/Gen Ratio: 1.0"
echo "Started: $(date)"
echo "----------------------------------------"

# Train model
python unified_training.py \
    --dataset OUTSIDE15k \
    --model deeplabv3plus_r50 \
    --strategy std_randaugment \
    --real-gen-ratio 1.0 \
    --domain-filter clear_day \
    --max-iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/std_randaugment/outside15k_cd/deeplabv3plus_r50/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --checkpoint /scratch/aaa_exchange/AWARE/WEIGHTS/std_randaugment/outside15k_cd/deeplabv3plus_r50/iter_80000.pth \
        --config /scratch/aaa_exchange/AWARE/WEIGHTS/std_randaugment/outside15k_cd/deeplabv3plus_r50/training_config.py \
        --dataset OUTSIDE15k \
        --output-dir /scratch/aaa_exchange/AWARE/WEIGHTS/std_randaugment/outside15k_cd/deeplabv3plus_r50/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: outside15k/deeplabv3plus_r50 at $(date)"


echo "----------------------------------------"
echo "Training: outside15k/pspnet_r50"
echo "Strategy: std_randaugment"
echo "Real/Gen Ratio: 1.0"
echo "Started: $(date)"
echo "----------------------------------------"

# Train model
python unified_training.py \
    --dataset OUTSIDE15k \
    --model pspnet_r50 \
    --strategy std_randaugment \
    --real-gen-ratio 1.0 \
    --domain-filter clear_day \
    --max-iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/std_randaugment/outside15k_cd/pspnet_r50/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --checkpoint /scratch/aaa_exchange/AWARE/WEIGHTS/std_randaugment/outside15k_cd/pspnet_r50/iter_80000.pth \
        --config /scratch/aaa_exchange/AWARE/WEIGHTS/std_randaugment/outside15k_cd/pspnet_r50/training_config.py \
        --dataset OUTSIDE15k \
        --output-dir /scratch/aaa_exchange/AWARE/WEIGHTS/std_randaugment/outside15k_cd/pspnet_r50/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: outside15k/pspnet_r50 at $(date)"


echo "----------------------------------------"
echo "Training: outside15k/segformer_mit-b5"
echo "Strategy: std_randaugment"
echo "Real/Gen Ratio: 1.0"
echo "Started: $(date)"
echo "----------------------------------------"

# Train model
python unified_training.py \
    --dataset OUTSIDE15k \
    --model segformer_mit-b5 \
    --strategy std_randaugment \
    --real-gen-ratio 1.0 \
    --domain-filter clear_day \
    --max-iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/std_randaugment/outside15k_cd/segformer_mit-b5/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --checkpoint /scratch/aaa_exchange/AWARE/WEIGHTS/std_randaugment/outside15k_cd/segformer_mit-b5/iter_80000.pth \
        --config /scratch/aaa_exchange/AWARE/WEIGHTS/std_randaugment/outside15k_cd/segformer_mit-b5/training_config.py \
        --dataset OUTSIDE15k \
        --output-dir /scratch/aaa_exchange/AWARE/WEIGHTS/std_randaugment/outside15k_cd/segformer_mit-b5/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: outside15k/segformer_mit-b5 at $(date)"


echo "========================================"
echo "Job completed: $(date)"
echo "========================================"
