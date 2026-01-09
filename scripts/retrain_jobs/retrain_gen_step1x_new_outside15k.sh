#!/bin/bash
#BSUB -J retrain_gen_step1x_new_outside15k
#BSUB -o /home/mima2416/repositories/PROVE/logs/retrain/retrain_gen_step1x_new_outside15k_%J.out
#BSUB -e /home/mima2416/repositories/PROVE/logs/retrain/retrain_gen_step1x_new_outside15k_%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=16000]"
#BSUB -gpu "num=1:mode=exclusive_process:gmem=20G"
#BSUB -W 4:00
#BSUB -q BatchGPU

# Activate environment
source ~/.bashrc
conda activate prove

cd /home/mima2416/repositories/PROVE

echo "========================================"
echo "Retraining job: retrain_gen_step1x_new_outside15k"
echo "Strategy: gen_step1x_new"
echo "Dataset: outside15k"
echo "Started: $(date)"
echo "========================================"


echo "----------------------------------------"
echo "Training: outside15k/deeplabv3plus_r50_ratio0p50"
echo "Strategy: gen_step1x_new"
echo "Real/Gen Ratio: 0.5"
echo "Started: $(date)"
echo "----------------------------------------"

# Train model
python unified_training.py \
    --dataset OUTSIDE15k \
    --model deeplabv3plus_r50 \
    --strategy gen_step1x_new \
    --real-gen-ratio 0.5 \
    --domain-filter clear_day \
    --max-iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_step1x_new/outside15k_cd/deeplabv3plus_r50_ratio0p50/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --checkpoint /scratch/aaa_exchange/AWARE/WEIGHTS/gen_step1x_new/outside15k_cd/deeplabv3plus_r50_ratio0p50/iter_80000.pth \
        --config /scratch/aaa_exchange/AWARE/WEIGHTS/gen_step1x_new/outside15k_cd/deeplabv3plus_r50_ratio0p50/training_config.py \
        --dataset OUTSIDE15k \
        --output-dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_step1x_new/outside15k_cd/deeplabv3plus_r50_ratio0p50/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: outside15k/deeplabv3plus_r50_ratio0p50 at $(date)"


echo "----------------------------------------"
echo "Training: outside15k/pspnet_r50_ratio0p50"
echo "Strategy: gen_step1x_new"
echo "Real/Gen Ratio: 0.5"
echo "Started: $(date)"
echo "----------------------------------------"

# Train model
python unified_training.py \
    --dataset OUTSIDE15k \
    --model pspnet_r50 \
    --strategy gen_step1x_new \
    --real-gen-ratio 0.5 \
    --domain-filter clear_day \
    --max-iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_step1x_new/outside15k_cd/pspnet_r50_ratio0p50/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --checkpoint /scratch/aaa_exchange/AWARE/WEIGHTS/gen_step1x_new/outside15k_cd/pspnet_r50_ratio0p50/iter_80000.pth \
        --config /scratch/aaa_exchange/AWARE/WEIGHTS/gen_step1x_new/outside15k_cd/pspnet_r50_ratio0p50/training_config.py \
        --dataset OUTSIDE15k \
        --output-dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_step1x_new/outside15k_cd/pspnet_r50_ratio0p50/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: outside15k/pspnet_r50_ratio0p50 at $(date)"


echo "----------------------------------------"
echo "Training: outside15k/segformer_mit-b5_ratio0p50"
echo "Strategy: gen_step1x_new"
echo "Real/Gen Ratio: 0.5"
echo "Started: $(date)"
echo "----------------------------------------"

# Train model
python unified_training.py \
    --dataset OUTSIDE15k \
    --model segformer_mit-b5 \
    --strategy gen_step1x_new \
    --real-gen-ratio 0.5 \
    --domain-filter clear_day \
    --max-iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_step1x_new/outside15k_cd/segformer_mit-b5_ratio0p50/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --checkpoint /scratch/aaa_exchange/AWARE/WEIGHTS/gen_step1x_new/outside15k_cd/segformer_mit-b5_ratio0p50/iter_80000.pth \
        --config /scratch/aaa_exchange/AWARE/WEIGHTS/gen_step1x_new/outside15k_cd/segformer_mit-b5_ratio0p50/training_config.py \
        --dataset OUTSIDE15k \
        --output-dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_step1x_new/outside15k_cd/segformer_mit-b5_ratio0p50/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: outside15k/segformer_mit-b5_ratio0p50 at $(date)"


echo "========================================"
echo "Job completed: $(date)"
echo "========================================"
