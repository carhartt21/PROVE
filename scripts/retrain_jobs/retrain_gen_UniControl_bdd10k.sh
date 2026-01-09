#!/bin/bash
#BSUB -J retrain_gen_UniControl_bdd10k
#BSUB -o /home/mima2416/repositories/PROVE/logs/retrain/retrain_gen_UniControl_bdd10k_%J.out
#BSUB -e /home/mima2416/repositories/PROVE/logs/retrain/retrain_gen_UniControl_bdd10k_%J.err
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
echo "Retraining job: retrain_gen_UniControl_bdd10k"
echo "Strategy: gen_UniControl"
echo "Dataset: bdd10k"
echo "Started: $(date)"
echo "========================================"


echo "----------------------------------------"
echo "Training: bdd10k/deeplabv3plus_r50_ratio0p50"
echo "Strategy: gen_UniControl"
echo "Real/Gen Ratio: 0.5"
echo "Started: $(date)"
echo "----------------------------------------"

# Train model
python unified_training.py \
    --dataset BDD10k \
    --model deeplabv3plus_r50 \
    --strategy gen_UniControl \
    --real-gen-ratio 0.5 \
    --domain-filter clear_day \
    --max-iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/bdd10k_cd/deeplabv3plus_r50_ratio0p50/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --checkpoint /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/bdd10k_cd/deeplabv3plus_r50_ratio0p50/iter_80000.pth \
        --config /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/bdd10k_cd/deeplabv3plus_r50_ratio0p50/training_config.py \
        --dataset BDD10k \
        --output-dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/bdd10k_cd/deeplabv3plus_r50_ratio0p50/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: bdd10k/deeplabv3plus_r50_ratio0p50 at $(date)"


echo "----------------------------------------"
echo "Training: bdd10k/pspnet_r50_ratio0p50"
echo "Strategy: gen_UniControl"
echo "Real/Gen Ratio: 0.5"
echo "Started: $(date)"
echo "----------------------------------------"

# Train model
python unified_training.py \
    --dataset BDD10k \
    --model pspnet_r50 \
    --strategy gen_UniControl \
    --real-gen-ratio 0.5 \
    --domain-filter clear_day \
    --max-iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/bdd10k_cd/pspnet_r50_ratio0p50/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --checkpoint /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/bdd10k_cd/pspnet_r50_ratio0p50/iter_80000.pth \
        --config /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/bdd10k_cd/pspnet_r50_ratio0p50/training_config.py \
        --dataset BDD10k \
        --output-dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/bdd10k_cd/pspnet_r50_ratio0p50/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: bdd10k/pspnet_r50_ratio0p50 at $(date)"


echo "----------------------------------------"
echo "Training: bdd10k/segformer_mit-b5_ratio0p50"
echo "Strategy: gen_UniControl"
echo "Real/Gen Ratio: 0.5"
echo "Started: $(date)"
echo "----------------------------------------"

# Train model
python unified_training.py \
    --dataset BDD10k \
    --model segformer_mit-b5 \
    --strategy gen_UniControl \
    --real-gen-ratio 0.5 \
    --domain-filter clear_day \
    --max-iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/bdd10k_cd/segformer_mit-b5_ratio0p50/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --checkpoint /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/bdd10k_cd/segformer_mit-b5_ratio0p50/iter_80000.pth \
        --config /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/bdd10k_cd/segformer_mit-b5_ratio0p50/training_config.py \
        --dataset BDD10k \
        --output-dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/bdd10k_cd/segformer_mit-b5_ratio0p50/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: bdd10k/segformer_mit-b5_ratio0p50 at $(date)"


echo "========================================"
echo "Job completed: $(date)"
echo "========================================"
