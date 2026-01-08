#!/bin/bash
#BSUB -J retrain_gen_UniControl
#BSUB -o /home/mima2416/repositories/PROVE/logs/retrain/retrain_gen_UniControl_%J.out
#BSUB -e /home/mima2416/repositories/PROVE/logs/retrain/retrain_gen_UniControl_%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=16000]"
#BSUB -gpu "num=1:mode=exclusive_process:gmem=20G"
#BSUB -W 72:00
#BSUB -q BatchGPU

# Activate environment
source ~/.bashrc
conda activate prove

cd /home/mima2416/repositories/PROVE

echo "========================================"
echo "Retraining job: retrain_gen_UniControl"
echo "Strategy: gen_UniControl"
echo "Started: $(date)"
echo "========================================"

# Process each model configuration sequentially

echo "----------------------------------------"
echo "Training: bdd10k/deeplabv3plus_r50"
echo "Strategy: gen_UniControl"
echo "Started: $(date)"
echo "----------------------------------------"

# Remove old checkpoints to force retraining
rm -f /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/bdd10k/deeplabv3plus_r50/iter_*.pth 2>/dev/null
rm -f /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/bdd10k/deeplabv3plus_r50/last_checkpoint 2>/dev/null

# Train model
python unified_training.py \
    --dataset BDD10k \
    --model deeplabv3plus_r50 \
    --strategy gen_UniControl \
    \
    --max_iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/bdd10k/deeplabv3plus_r50/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --weights_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/bdd10k/deeplabv3plus_r50/iter_80000.pth \
        --config_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/bdd10k/deeplabv3plus_r50/training_config.py \
        --dataset BDD10k \
        --output_dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/bdd10k/deeplabv3plus_r50/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: bdd10k/deeplabv3plus_r50 at $(date)"


echo "----------------------------------------"
echo "Training: bdd10k/deeplabv3plus_r50_clear_day"
echo "Strategy: gen_UniControl"
echo "Started: $(date)"
echo "----------------------------------------"

# Remove old checkpoints to force retraining
rm -f /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/bdd10k/deeplabv3plus_r50_clear_day/iter_*.pth 2>/dev/null
rm -f /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/bdd10k/deeplabv3plus_r50_clear_day/last_checkpoint 2>/dev/null

# Train model
python unified_training.py \
    --dataset BDD10k \
    --model deeplabv3plus_r50 \
    --strategy gen_UniControl \
    --domain_filter clear_day\
    --max_iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/bdd10k/deeplabv3plus_r50_clear_day/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --weights_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/bdd10k/deeplabv3plus_r50_clear_day/iter_80000.pth \
        --config_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/bdd10k/deeplabv3plus_r50_clear_day/training_config.py \
        --dataset BDD10k \
        --output_dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/bdd10k/deeplabv3plus_r50_clear_day/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: bdd10k/deeplabv3plus_r50_clear_day at $(date)"


echo "----------------------------------------"
echo "Training: bdd10k/pspnet_r50"
echo "Strategy: gen_UniControl"
echo "Started: $(date)"
echo "----------------------------------------"

# Remove old checkpoints to force retraining
rm -f /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/bdd10k/pspnet_r50/iter_*.pth 2>/dev/null
rm -f /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/bdd10k/pspnet_r50/last_checkpoint 2>/dev/null

# Train model
python unified_training.py \
    --dataset BDD10k \
    --model pspnet_r50 \
    --strategy gen_UniControl \
    \
    --max_iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/bdd10k/pspnet_r50/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --weights_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/bdd10k/pspnet_r50/iter_80000.pth \
        --config_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/bdd10k/pspnet_r50/training_config.py \
        --dataset BDD10k \
        --output_dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/bdd10k/pspnet_r50/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: bdd10k/pspnet_r50 at $(date)"


echo "----------------------------------------"
echo "Training: bdd10k/pspnet_r50_clear_day"
echo "Strategy: gen_UniControl"
echo "Started: $(date)"
echo "----------------------------------------"

# Remove old checkpoints to force retraining
rm -f /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/bdd10k/pspnet_r50_clear_day/iter_*.pth 2>/dev/null
rm -f /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/bdd10k/pspnet_r50_clear_day/last_checkpoint 2>/dev/null

# Train model
python unified_training.py \
    --dataset BDD10k \
    --model pspnet_r50 \
    --strategy gen_UniControl \
    --domain_filter clear_day\
    --max_iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/bdd10k/pspnet_r50_clear_day/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --weights_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/bdd10k/pspnet_r50_clear_day/iter_80000.pth \
        --config_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/bdd10k/pspnet_r50_clear_day/training_config.py \
        --dataset BDD10k \
        --output_dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/bdd10k/pspnet_r50_clear_day/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: bdd10k/pspnet_r50_clear_day at $(date)"


echo "----------------------------------------"
echo "Training: bdd10k/segformer_mit-b5"
echo "Strategy: gen_UniControl"
echo "Started: $(date)"
echo "----------------------------------------"

# Remove old checkpoints to force retraining
rm -f /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/bdd10k/segformer_mit-b5/iter_*.pth 2>/dev/null
rm -f /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/bdd10k/segformer_mit-b5/last_checkpoint 2>/dev/null

# Train model
python unified_training.py \
    --dataset BDD10k \
    --model segformer_mit-b5 \
    --strategy gen_UniControl \
    \
    --max_iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/bdd10k/segformer_mit-b5/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --weights_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/bdd10k/segformer_mit-b5/iter_80000.pth \
        --config_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/bdd10k/segformer_mit-b5/training_config.py \
        --dataset BDD10k \
        --output_dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/bdd10k/segformer_mit-b5/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: bdd10k/segformer_mit-b5 at $(date)"


echo "----------------------------------------"
echo "Training: bdd10k/segformer_mit-b5_clear_day"
echo "Strategy: gen_UniControl"
echo "Started: $(date)"
echo "----------------------------------------"

# Remove old checkpoints to force retraining
rm -f /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/bdd10k/segformer_mit-b5_clear_day/iter_*.pth 2>/dev/null
rm -f /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/bdd10k/segformer_mit-b5_clear_day/last_checkpoint 2>/dev/null

# Train model
python unified_training.py \
    --dataset BDD10k \
    --model segformer_mit-b5 \
    --strategy gen_UniControl \
    --domain_filter clear_day\
    --max_iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/bdd10k/segformer_mit-b5_clear_day/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --weights_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/bdd10k/segformer_mit-b5_clear_day/iter_80000.pth \
        --config_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/bdd10k/segformer_mit-b5_clear_day/training_config.py \
        --dataset BDD10k \
        --output_dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/bdd10k/segformer_mit-b5_clear_day/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: bdd10k/segformer_mit-b5_clear_day at $(date)"


echo "----------------------------------------"
echo "Training: idd-aw/deeplabv3plus_r50"
echo "Strategy: gen_UniControl"
echo "Started: $(date)"
echo "----------------------------------------"

# Remove old checkpoints to force retraining
rm -f /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/idd-aw/deeplabv3plus_r50/iter_*.pth 2>/dev/null
rm -f /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/idd-aw/deeplabv3plus_r50/last_checkpoint 2>/dev/null

# Train model
python unified_training.py \
    --dataset IDD-AW \
    --model deeplabv3plus_r50 \
    --strategy gen_UniControl \
    \
    --max_iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/idd-aw/deeplabv3plus_r50/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --weights_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/idd-aw/deeplabv3plus_r50/iter_80000.pth \
        --config_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/idd-aw/deeplabv3plus_r50/training_config.py \
        --dataset IDD-AW \
        --output_dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/idd-aw/deeplabv3plus_r50/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: idd-aw/deeplabv3plus_r50 at $(date)"


echo "----------------------------------------"
echo "Training: idd-aw/deeplabv3plus_r50_clear_day"
echo "Strategy: gen_UniControl"
echo "Started: $(date)"
echo "----------------------------------------"

# Remove old checkpoints to force retraining
rm -f /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/idd-aw/deeplabv3plus_r50_clear_day/iter_*.pth 2>/dev/null
rm -f /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/idd-aw/deeplabv3plus_r50_clear_day/last_checkpoint 2>/dev/null

# Train model
python unified_training.py \
    --dataset IDD-AW \
    --model deeplabv3plus_r50 \
    --strategy gen_UniControl \
    --domain_filter clear_day\
    --max_iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/idd-aw/deeplabv3plus_r50_clear_day/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --weights_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/idd-aw/deeplabv3plus_r50_clear_day/iter_80000.pth \
        --config_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/idd-aw/deeplabv3plus_r50_clear_day/training_config.py \
        --dataset IDD-AW \
        --output_dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/idd-aw/deeplabv3plus_r50_clear_day/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: idd-aw/deeplabv3plus_r50_clear_day at $(date)"


echo "----------------------------------------"
echo "Training: idd-aw/pspnet_r50"
echo "Strategy: gen_UniControl"
echo "Started: $(date)"
echo "----------------------------------------"

# Remove old checkpoints to force retraining
rm -f /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/idd-aw/pspnet_r50/iter_*.pth 2>/dev/null
rm -f /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/idd-aw/pspnet_r50/last_checkpoint 2>/dev/null

# Train model
python unified_training.py \
    --dataset IDD-AW \
    --model pspnet_r50 \
    --strategy gen_UniControl \
    \
    --max_iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/idd-aw/pspnet_r50/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --weights_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/idd-aw/pspnet_r50/iter_80000.pth \
        --config_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/idd-aw/pspnet_r50/training_config.py \
        --dataset IDD-AW \
        --output_dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/idd-aw/pspnet_r50/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: idd-aw/pspnet_r50 at $(date)"


echo "----------------------------------------"
echo "Training: idd-aw/pspnet_r50_clear_day"
echo "Strategy: gen_UniControl"
echo "Started: $(date)"
echo "----------------------------------------"

# Remove old checkpoints to force retraining
rm -f /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/idd-aw/pspnet_r50_clear_day/iter_*.pth 2>/dev/null
rm -f /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/idd-aw/pspnet_r50_clear_day/last_checkpoint 2>/dev/null

# Train model
python unified_training.py \
    --dataset IDD-AW \
    --model pspnet_r50 \
    --strategy gen_UniControl \
    --domain_filter clear_day\
    --max_iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/idd-aw/pspnet_r50_clear_day/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --weights_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/idd-aw/pspnet_r50_clear_day/iter_80000.pth \
        --config_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/idd-aw/pspnet_r50_clear_day/training_config.py \
        --dataset IDD-AW \
        --output_dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/idd-aw/pspnet_r50_clear_day/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: idd-aw/pspnet_r50_clear_day at $(date)"


echo "----------------------------------------"
echo "Training: idd-aw/segformer_mit-b5"
echo "Strategy: gen_UniControl"
echo "Started: $(date)"
echo "----------------------------------------"

# Remove old checkpoints to force retraining
rm -f /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/idd-aw/segformer_mit-b5/iter_*.pth 2>/dev/null
rm -f /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/idd-aw/segformer_mit-b5/last_checkpoint 2>/dev/null

# Train model
python unified_training.py \
    --dataset IDD-AW \
    --model segformer_mit-b5 \
    --strategy gen_UniControl \
    \
    --max_iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/idd-aw/segformer_mit-b5/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --weights_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/idd-aw/segformer_mit-b5/iter_80000.pth \
        --config_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/idd-aw/segformer_mit-b5/training_config.py \
        --dataset IDD-AW \
        --output_dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/idd-aw/segformer_mit-b5/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: idd-aw/segformer_mit-b5 at $(date)"


echo "----------------------------------------"
echo "Training: idd-aw/segformer_mit-b5_clear_day"
echo "Strategy: gen_UniControl"
echo "Started: $(date)"
echo "----------------------------------------"

# Remove old checkpoints to force retraining
rm -f /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/idd-aw/segformer_mit-b5_clear_day/iter_*.pth 2>/dev/null
rm -f /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/idd-aw/segformer_mit-b5_clear_day/last_checkpoint 2>/dev/null

# Train model
python unified_training.py \
    --dataset IDD-AW \
    --model segformer_mit-b5 \
    --strategy gen_UniControl \
    --domain_filter clear_day\
    --max_iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/idd-aw/segformer_mit-b5_clear_day/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --weights_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/idd-aw/segformer_mit-b5_clear_day/iter_80000.pth \
        --config_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/idd-aw/segformer_mit-b5_clear_day/training_config.py \
        --dataset IDD-AW \
        --output_dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/idd-aw/segformer_mit-b5_clear_day/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: idd-aw/segformer_mit-b5_clear_day at $(date)"


echo "----------------------------------------"
echo "Training: mapillaryvistas/deeplabv3plus_r50"
echo "Strategy: gen_UniControl"
echo "Started: $(date)"
echo "----------------------------------------"

# Remove old checkpoints to force retraining
rm -f /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/mapillaryvistas/deeplabv3plus_r50/iter_*.pth 2>/dev/null
rm -f /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/mapillaryvistas/deeplabv3plus_r50/last_checkpoint 2>/dev/null

# Train model
python unified_training.py \
    --dataset MapillaryVistas \
    --model deeplabv3plus_r50 \
    --strategy gen_UniControl \
    \
    --max_iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/mapillaryvistas/deeplabv3plus_r50/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --weights_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/mapillaryvistas/deeplabv3plus_r50/iter_80000.pth \
        --config_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/mapillaryvistas/deeplabv3plus_r50/training_config.py \
        --dataset MapillaryVistas \
        --output_dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/mapillaryvistas/deeplabv3plus_r50/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: mapillaryvistas/deeplabv3plus_r50 at $(date)"


echo "----------------------------------------"
echo "Training: mapillaryvistas/deeplabv3plus_r50_clear_day"
echo "Strategy: gen_UniControl"
echo "Started: $(date)"
echo "----------------------------------------"

# Remove old checkpoints to force retraining
rm -f /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/mapillaryvistas/deeplabv3plus_r50_clear_day/iter_*.pth 2>/dev/null
rm -f /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/mapillaryvistas/deeplabv3plus_r50_clear_day/last_checkpoint 2>/dev/null

# Train model
python unified_training.py \
    --dataset MapillaryVistas \
    --model deeplabv3plus_r50 \
    --strategy gen_UniControl \
    --domain_filter clear_day\
    --max_iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/mapillaryvistas/deeplabv3plus_r50_clear_day/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --weights_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/mapillaryvistas/deeplabv3plus_r50_clear_day/iter_80000.pth \
        --config_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/mapillaryvistas/deeplabv3plus_r50_clear_day/training_config.py \
        --dataset MapillaryVistas \
        --output_dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/mapillaryvistas/deeplabv3plus_r50_clear_day/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: mapillaryvistas/deeplabv3plus_r50_clear_day at $(date)"


echo "----------------------------------------"
echo "Training: mapillaryvistas/pspnet_r50"
echo "Strategy: gen_UniControl"
echo "Started: $(date)"
echo "----------------------------------------"

# Remove old checkpoints to force retraining
rm -f /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/mapillaryvistas/pspnet_r50/iter_*.pth 2>/dev/null
rm -f /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/mapillaryvistas/pspnet_r50/last_checkpoint 2>/dev/null

# Train model
python unified_training.py \
    --dataset MapillaryVistas \
    --model pspnet_r50 \
    --strategy gen_UniControl \
    \
    --max_iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/mapillaryvistas/pspnet_r50/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --weights_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/mapillaryvistas/pspnet_r50/iter_80000.pth \
        --config_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/mapillaryvistas/pspnet_r50/training_config.py \
        --dataset MapillaryVistas \
        --output_dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/mapillaryvistas/pspnet_r50/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: mapillaryvistas/pspnet_r50 at $(date)"


echo "----------------------------------------"
echo "Training: mapillaryvistas/pspnet_r50_clear_day"
echo "Strategy: gen_UniControl"
echo "Started: $(date)"
echo "----------------------------------------"

# Remove old checkpoints to force retraining
rm -f /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/mapillaryvistas/pspnet_r50_clear_day/iter_*.pth 2>/dev/null
rm -f /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/mapillaryvistas/pspnet_r50_clear_day/last_checkpoint 2>/dev/null

# Train model
python unified_training.py \
    --dataset MapillaryVistas \
    --model pspnet_r50 \
    --strategy gen_UniControl \
    --domain_filter clear_day\
    --max_iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/mapillaryvistas/pspnet_r50_clear_day/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --weights_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/mapillaryvistas/pspnet_r50_clear_day/iter_80000.pth \
        --config_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/mapillaryvistas/pspnet_r50_clear_day/training_config.py \
        --dataset MapillaryVistas \
        --output_dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/mapillaryvistas/pspnet_r50_clear_day/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: mapillaryvistas/pspnet_r50_clear_day at $(date)"


echo "----------------------------------------"
echo "Training: mapillaryvistas/segformer_mit-b5"
echo "Strategy: gen_UniControl"
echo "Started: $(date)"
echo "----------------------------------------"

# Remove old checkpoints to force retraining
rm -f /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/mapillaryvistas/segformer_mit-b5/iter_*.pth 2>/dev/null
rm -f /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/mapillaryvistas/segformer_mit-b5/last_checkpoint 2>/dev/null

# Train model
python unified_training.py \
    --dataset MapillaryVistas \
    --model segformer_mit-b5 \
    --strategy gen_UniControl \
    \
    --max_iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/mapillaryvistas/segformer_mit-b5/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --weights_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/mapillaryvistas/segformer_mit-b5/iter_80000.pth \
        --config_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/mapillaryvistas/segformer_mit-b5/training_config.py \
        --dataset MapillaryVistas \
        --output_dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/mapillaryvistas/segformer_mit-b5/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: mapillaryvistas/segformer_mit-b5 at $(date)"


echo "----------------------------------------"
echo "Training: mapillaryvistas/segformer_mit-b5_clear_day"
echo "Strategy: gen_UniControl"
echo "Started: $(date)"
echo "----------------------------------------"

# Remove old checkpoints to force retraining
rm -f /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/mapillaryvistas/segformer_mit-b5_clear_day/iter_*.pth 2>/dev/null
rm -f /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/mapillaryvistas/segformer_mit-b5_clear_day/last_checkpoint 2>/dev/null

# Train model
python unified_training.py \
    --dataset MapillaryVistas \
    --model segformer_mit-b5 \
    --strategy gen_UniControl \
    --domain_filter clear_day\
    --max_iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/mapillaryvistas/segformer_mit-b5_clear_day/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --weights_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/mapillaryvistas/segformer_mit-b5_clear_day/iter_80000.pth \
        --config_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/mapillaryvistas/segformer_mit-b5_clear_day/training_config.py \
        --dataset MapillaryVistas \
        --output_dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_UniControl/mapillaryvistas/segformer_mit-b5_clear_day/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: mapillaryvistas/segformer_mit-b5_clear_day at $(date)"


echo "========================================"
echo "Job completed: $(date)"
echo "========================================"
