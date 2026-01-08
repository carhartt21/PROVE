#!/bin/bash
#BSUB -J retrain_gen_Weather_Effect_Generator
#BSUB -o /home/mima2416/repositories/PROVE/logs/retrain/retrain_gen_Weather_Effect_Generator_%J.out
#BSUB -e /home/mima2416/repositories/PROVE/logs/retrain/retrain_gen_Weather_Effect_Generator_%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=16000]"
#BSUB -gpu "num=1:mode=exclusive_process:gmem=20G"
#BSUB -W 72:00
#BSUB -q BatchGPU

# Activate environment
source ~/.bashrc
conda activate prove

cd /home/mima2416/repositories/PROVE

echo "========================================"
echo "Retraining job: retrain_gen_Weather_Effect_Generator"
echo "Strategy: gen_Weather_Effect_Generator"
echo "Started: $(date)"
echo "========================================"

# Process each model configuration sequentially

echo "----------------------------------------"
echo "Training: bdd10k/deeplabv3plus_r50"
echo "Strategy: gen_Weather_Effect_Generator"
echo "Real/Gen Ratio: 0.5"
echo "Started: $(date)"
echo "----------------------------------------"

# Train model
python unified_training.py \
    --dataset BDD10k \
    --model deeplabv3plus_r50 \
    --strategy gen_Weather_Effect_Generator \
    --real-gen-ratio 0.5 \
     \
    --max-iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/bdd10k/deeplabv3plus_r50/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --weights_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/bdd10k/deeplabv3plus_r50/iter_80000.pth \
        --config_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/bdd10k/deeplabv3plus_r50/training_config.py \
        --dataset BDD10k \
        --output_dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/bdd10k/deeplabv3plus_r50/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: bdd10k/deeplabv3plus_r50 at $(date)"


echo "----------------------------------------"
echo "Training: bdd10k/deeplabv3plus_r50_clear_day"
echo "Strategy: gen_Weather_Effect_Generator"
echo "Real/Gen Ratio: 0.5"
echo "Started: $(date)"
echo "----------------------------------------"

# Train model
python unified_training.py \
    --dataset BDD10k \
    --model deeplabv3plus_r50 \
    --strategy gen_Weather_Effect_Generator \
    --real-gen-ratio 0.5 \
    --domain-filter clear_day \
    --max-iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/bdd10k/deeplabv3plus_r50_clear_day/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --weights_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/bdd10k/deeplabv3plus_r50_clear_day/iter_80000.pth \
        --config_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/bdd10k/deeplabv3plus_r50_clear_day/training_config.py \
        --dataset BDD10k \
        --output_dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/bdd10k/deeplabv3plus_r50_clear_day/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: bdd10k/deeplabv3plus_r50_clear_day at $(date)"


echo "----------------------------------------"
echo "Training: bdd10k/pspnet_r50"
echo "Strategy: gen_Weather_Effect_Generator"
echo "Real/Gen Ratio: 0.5"
echo "Started: $(date)"
echo "----------------------------------------"

# Train model
python unified_training.py \
    --dataset BDD10k \
    --model pspnet_r50 \
    --strategy gen_Weather_Effect_Generator \
    --real-gen-ratio 0.5 \
     \
    --max-iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/bdd10k/pspnet_r50/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --weights_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/bdd10k/pspnet_r50/iter_80000.pth \
        --config_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/bdd10k/pspnet_r50/training_config.py \
        --dataset BDD10k \
        --output_dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/bdd10k/pspnet_r50/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: bdd10k/pspnet_r50 at $(date)"


echo "----------------------------------------"
echo "Training: bdd10k/pspnet_r50_clear_day"
echo "Strategy: gen_Weather_Effect_Generator"
echo "Real/Gen Ratio: 0.5"
echo "Started: $(date)"
echo "----------------------------------------"

# Train model
python unified_training.py \
    --dataset BDD10k \
    --model pspnet_r50 \
    --strategy gen_Weather_Effect_Generator \
    --real-gen-ratio 0.5 \
    --domain-filter clear_day \
    --max-iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/bdd10k/pspnet_r50_clear_day/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --weights_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/bdd10k/pspnet_r50_clear_day/iter_80000.pth \
        --config_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/bdd10k/pspnet_r50_clear_day/training_config.py \
        --dataset BDD10k \
        --output_dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/bdd10k/pspnet_r50_clear_day/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: bdd10k/pspnet_r50_clear_day at $(date)"


echo "----------------------------------------"
echo "Training: bdd10k/segformer_mit-b5"
echo "Strategy: gen_Weather_Effect_Generator"
echo "Real/Gen Ratio: 0.5"
echo "Started: $(date)"
echo "----------------------------------------"

# Train model
python unified_training.py \
    --dataset BDD10k \
    --model segformer_mit-b5 \
    --strategy gen_Weather_Effect_Generator \
    --real-gen-ratio 0.5 \
     \
    --max-iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/bdd10k/segformer_mit-b5/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --weights_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/bdd10k/segformer_mit-b5/iter_80000.pth \
        --config_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/bdd10k/segformer_mit-b5/training_config.py \
        --dataset BDD10k \
        --output_dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/bdd10k/segformer_mit-b5/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: bdd10k/segformer_mit-b5 at $(date)"


echo "----------------------------------------"
echo "Training: bdd10k/segformer_mit-b5_clear_day"
echo "Strategy: gen_Weather_Effect_Generator"
echo "Real/Gen Ratio: 0.5"
echo "Started: $(date)"
echo "----------------------------------------"

# Train model
python unified_training.py \
    --dataset BDD10k \
    --model segformer_mit-b5 \
    --strategy gen_Weather_Effect_Generator \
    --real-gen-ratio 0.5 \
    --domain-filter clear_day \
    --max-iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/bdd10k/segformer_mit-b5_clear_day/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --weights_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/bdd10k/segformer_mit-b5_clear_day/iter_80000.pth \
        --config_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/bdd10k/segformer_mit-b5_clear_day/training_config.py \
        --dataset BDD10k \
        --output_dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/bdd10k/segformer_mit-b5_clear_day/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: bdd10k/segformer_mit-b5_clear_day at $(date)"


echo "----------------------------------------"
echo "Training: idd-aw/deeplabv3plus_r50"
echo "Strategy: gen_Weather_Effect_Generator"
echo "Real/Gen Ratio: 0.5"
echo "Started: $(date)"
echo "----------------------------------------"

# Train model
python unified_training.py \
    --dataset IDD-AW \
    --model deeplabv3plus_r50 \
    --strategy gen_Weather_Effect_Generator \
    --real-gen-ratio 0.5 \
     \
    --max-iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/idd-aw/deeplabv3plus_r50/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --weights_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/idd-aw/deeplabv3plus_r50/iter_80000.pth \
        --config_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/idd-aw/deeplabv3plus_r50/training_config.py \
        --dataset IDD-AW \
        --output_dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/idd-aw/deeplabv3plus_r50/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: idd-aw/deeplabv3plus_r50 at $(date)"


echo "----------------------------------------"
echo "Training: idd-aw/deeplabv3plus_r50_clear_day"
echo "Strategy: gen_Weather_Effect_Generator"
echo "Real/Gen Ratio: 0.5"
echo "Started: $(date)"
echo "----------------------------------------"

# Train model
python unified_training.py \
    --dataset IDD-AW \
    --model deeplabv3plus_r50 \
    --strategy gen_Weather_Effect_Generator \
    --real-gen-ratio 0.5 \
    --domain-filter clear_day \
    --max-iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/idd-aw/deeplabv3plus_r50_clear_day/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --weights_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/idd-aw/deeplabv3plus_r50_clear_day/iter_80000.pth \
        --config_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/idd-aw/deeplabv3plus_r50_clear_day/training_config.py \
        --dataset IDD-AW \
        --output_dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/idd-aw/deeplabv3plus_r50_clear_day/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: idd-aw/deeplabv3plus_r50_clear_day at $(date)"


echo "----------------------------------------"
echo "Training: idd-aw/pspnet_r50"
echo "Strategy: gen_Weather_Effect_Generator"
echo "Real/Gen Ratio: 0.5"
echo "Started: $(date)"
echo "----------------------------------------"

# Train model
python unified_training.py \
    --dataset IDD-AW \
    --model pspnet_r50 \
    --strategy gen_Weather_Effect_Generator \
    --real-gen-ratio 0.5 \
     \
    --max-iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/idd-aw/pspnet_r50/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --weights_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/idd-aw/pspnet_r50/iter_80000.pth \
        --config_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/idd-aw/pspnet_r50/training_config.py \
        --dataset IDD-AW \
        --output_dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/idd-aw/pspnet_r50/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: idd-aw/pspnet_r50 at $(date)"


echo "----------------------------------------"
echo "Training: idd-aw/pspnet_r50_clear_day"
echo "Strategy: gen_Weather_Effect_Generator"
echo "Real/Gen Ratio: 0.5"
echo "Started: $(date)"
echo "----------------------------------------"

# Train model
python unified_training.py \
    --dataset IDD-AW \
    --model pspnet_r50 \
    --strategy gen_Weather_Effect_Generator \
    --real-gen-ratio 0.5 \
    --domain-filter clear_day \
    --max-iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/idd-aw/pspnet_r50_clear_day/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --weights_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/idd-aw/pspnet_r50_clear_day/iter_80000.pth \
        --config_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/idd-aw/pspnet_r50_clear_day/training_config.py \
        --dataset IDD-AW \
        --output_dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/idd-aw/pspnet_r50_clear_day/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: idd-aw/pspnet_r50_clear_day at $(date)"


echo "----------------------------------------"
echo "Training: idd-aw/segformer_mit-b5"
echo "Strategy: gen_Weather_Effect_Generator"
echo "Real/Gen Ratio: 0.5"
echo "Started: $(date)"
echo "----------------------------------------"

# Train model
python unified_training.py \
    --dataset IDD-AW \
    --model segformer_mit-b5 \
    --strategy gen_Weather_Effect_Generator \
    --real-gen-ratio 0.5 \
     \
    --max-iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/idd-aw/segformer_mit-b5/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --weights_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/idd-aw/segformer_mit-b5/iter_80000.pth \
        --config_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/idd-aw/segformer_mit-b5/training_config.py \
        --dataset IDD-AW \
        --output_dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/idd-aw/segformer_mit-b5/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: idd-aw/segformer_mit-b5 at $(date)"


echo "----------------------------------------"
echo "Training: idd-aw/segformer_mit-b5_clear_day"
echo "Strategy: gen_Weather_Effect_Generator"
echo "Real/Gen Ratio: 0.5"
echo "Started: $(date)"
echo "----------------------------------------"

# Train model
python unified_training.py \
    --dataset IDD-AW \
    --model segformer_mit-b5 \
    --strategy gen_Weather_Effect_Generator \
    --real-gen-ratio 0.5 \
    --domain-filter clear_day \
    --max-iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/idd-aw/segformer_mit-b5_clear_day/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --weights_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/idd-aw/segformer_mit-b5_clear_day/iter_80000.pth \
        --config_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/idd-aw/segformer_mit-b5_clear_day/training_config.py \
        --dataset IDD-AW \
        --output_dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/idd-aw/segformer_mit-b5_clear_day/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: idd-aw/segformer_mit-b5_clear_day at $(date)"


echo "----------------------------------------"
echo "Training: mapillaryvistas/deeplabv3plus_r50"
echo "Strategy: gen_Weather_Effect_Generator"
echo "Real/Gen Ratio: 0.5"
echo "Started: $(date)"
echo "----------------------------------------"

# Train model
python unified_training.py \
    --dataset MapillaryVistas \
    --model deeplabv3plus_r50 \
    --strategy gen_Weather_Effect_Generator \
    --real-gen-ratio 0.5 \
     \
    --max-iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/mapillaryvistas/deeplabv3plus_r50/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --weights_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/mapillaryvistas/deeplabv3plus_r50/iter_80000.pth \
        --config_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/mapillaryvistas/deeplabv3plus_r50/training_config.py \
        --dataset MapillaryVistas \
        --output_dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/mapillaryvistas/deeplabv3plus_r50/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: mapillaryvistas/deeplabv3plus_r50 at $(date)"


echo "----------------------------------------"
echo "Training: mapillaryvistas/deeplabv3plus_r50_clear_day"
echo "Strategy: gen_Weather_Effect_Generator"
echo "Real/Gen Ratio: 0.5"
echo "Started: $(date)"
echo "----------------------------------------"

# Train model
python unified_training.py \
    --dataset MapillaryVistas \
    --model deeplabv3plus_r50 \
    --strategy gen_Weather_Effect_Generator \
    --real-gen-ratio 0.5 \
    --domain-filter clear_day \
    --max-iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/mapillaryvistas/deeplabv3plus_r50_clear_day/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --weights_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/mapillaryvistas/deeplabv3plus_r50_clear_day/iter_80000.pth \
        --config_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/mapillaryvistas/deeplabv3plus_r50_clear_day/training_config.py \
        --dataset MapillaryVistas \
        --output_dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/mapillaryvistas/deeplabv3plus_r50_clear_day/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: mapillaryvistas/deeplabv3plus_r50_clear_day at $(date)"


echo "----------------------------------------"
echo "Training: mapillaryvistas/pspnet_r50"
echo "Strategy: gen_Weather_Effect_Generator"
echo "Real/Gen Ratio: 0.5"
echo "Started: $(date)"
echo "----------------------------------------"

# Train model
python unified_training.py \
    --dataset MapillaryVistas \
    --model pspnet_r50 \
    --strategy gen_Weather_Effect_Generator \
    --real-gen-ratio 0.5 \
     \
    --max-iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/mapillaryvistas/pspnet_r50/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --weights_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/mapillaryvistas/pspnet_r50/iter_80000.pth \
        --config_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/mapillaryvistas/pspnet_r50/training_config.py \
        --dataset MapillaryVistas \
        --output_dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/mapillaryvistas/pspnet_r50/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: mapillaryvistas/pspnet_r50 at $(date)"


echo "----------------------------------------"
echo "Training: mapillaryvistas/pspnet_r50_clear_day"
echo "Strategy: gen_Weather_Effect_Generator"
echo "Real/Gen Ratio: 0.5"
echo "Started: $(date)"
echo "----------------------------------------"

# Train model
python unified_training.py \
    --dataset MapillaryVistas \
    --model pspnet_r50 \
    --strategy gen_Weather_Effect_Generator \
    --real-gen-ratio 0.5 \
    --domain-filter clear_day \
    --max-iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/mapillaryvistas/pspnet_r50_clear_day/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --weights_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/mapillaryvistas/pspnet_r50_clear_day/iter_80000.pth \
        --config_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/mapillaryvistas/pspnet_r50_clear_day/training_config.py \
        --dataset MapillaryVistas \
        --output_dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/mapillaryvistas/pspnet_r50_clear_day/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: mapillaryvistas/pspnet_r50_clear_day at $(date)"


echo "----------------------------------------"
echo "Training: mapillaryvistas/segformer_mit-b5"
echo "Strategy: gen_Weather_Effect_Generator"
echo "Real/Gen Ratio: 0.5"
echo "Started: $(date)"
echo "----------------------------------------"

# Train model
python unified_training.py \
    --dataset MapillaryVistas \
    --model segformer_mit-b5 \
    --strategy gen_Weather_Effect_Generator \
    --real-gen-ratio 0.5 \
     \
    --max-iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/mapillaryvistas/segformer_mit-b5/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --weights_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/mapillaryvistas/segformer_mit-b5/iter_80000.pth \
        --config_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/mapillaryvistas/segformer_mit-b5/training_config.py \
        --dataset MapillaryVistas \
        --output_dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/mapillaryvistas/segformer_mit-b5/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: mapillaryvistas/segformer_mit-b5 at $(date)"


echo "----------------------------------------"
echo "Training: mapillaryvistas/segformer_mit-b5_clear_day"
echo "Strategy: gen_Weather_Effect_Generator"
echo "Real/Gen Ratio: 0.5"
echo "Started: $(date)"
echo "----------------------------------------"

# Train model
python unified_training.py \
    --dataset MapillaryVistas \
    --model segformer_mit-b5 \
    --strategy gen_Weather_Effect_Generator \
    --real-gen-ratio 0.5 \
    --domain-filter clear_day \
    --max-iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/mapillaryvistas/segformer_mit-b5_clear_day/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --weights_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/mapillaryvistas/segformer_mit-b5_clear_day/iter_80000.pth \
        --config_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/mapillaryvistas/segformer_mit-b5_clear_day/training_config.py \
        --dataset MapillaryVistas \
        --output_dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/mapillaryvistas/segformer_mit-b5_clear_day/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: mapillaryvistas/segformer_mit-b5_clear_day at $(date)"


echo "----------------------------------------"
echo "Training: outside15k/deeplabv3plus_r50"
echo "Strategy: gen_Weather_Effect_Generator"
echo "Real/Gen Ratio: 0.5"
echo "Started: $(date)"
echo "----------------------------------------"

# Train model
python unified_training.py \
    --dataset OUTSIDE15k \
    --model deeplabv3plus_r50 \
    --strategy gen_Weather_Effect_Generator \
    --real-gen-ratio 0.5 \
     \
    --max-iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/outside15k/deeplabv3plus_r50/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --weights_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/outside15k/deeplabv3plus_r50/iter_80000.pth \
        --config_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/outside15k/deeplabv3plus_r50/training_config.py \
        --dataset OUTSIDE15k \
        --output_dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/outside15k/deeplabv3plus_r50/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: outside15k/deeplabv3plus_r50 at $(date)"


echo "----------------------------------------"
echo "Training: outside15k/deeplabv3plus_r50_clear_day"
echo "Strategy: gen_Weather_Effect_Generator"
echo "Real/Gen Ratio: 0.5"
echo "Started: $(date)"
echo "----------------------------------------"

# Train model
python unified_training.py \
    --dataset OUTSIDE15k \
    --model deeplabv3plus_r50 \
    --strategy gen_Weather_Effect_Generator \
    --real-gen-ratio 0.5 \
    --domain-filter clear_day \
    --max-iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/outside15k/deeplabv3plus_r50_clear_day/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --weights_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/outside15k/deeplabv3plus_r50_clear_day/iter_80000.pth \
        --config_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/outside15k/deeplabv3plus_r50_clear_day/training_config.py \
        --dataset OUTSIDE15k \
        --output_dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/outside15k/deeplabv3plus_r50_clear_day/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: outside15k/deeplabv3plus_r50_clear_day at $(date)"


echo "----------------------------------------"
echo "Training: outside15k/pspnet_r50"
echo "Strategy: gen_Weather_Effect_Generator"
echo "Real/Gen Ratio: 0.5"
echo "Started: $(date)"
echo "----------------------------------------"

# Train model
python unified_training.py \
    --dataset OUTSIDE15k \
    --model pspnet_r50 \
    --strategy gen_Weather_Effect_Generator \
    --real-gen-ratio 0.5 \
     \
    --max-iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/outside15k/pspnet_r50/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --weights_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/outside15k/pspnet_r50/iter_80000.pth \
        --config_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/outside15k/pspnet_r50/training_config.py \
        --dataset OUTSIDE15k \
        --output_dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/outside15k/pspnet_r50/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: outside15k/pspnet_r50 at $(date)"


echo "----------------------------------------"
echo "Training: outside15k/pspnet_r50_clear_day"
echo "Strategy: gen_Weather_Effect_Generator"
echo "Real/Gen Ratio: 0.5"
echo "Started: $(date)"
echo "----------------------------------------"

# Train model
python unified_training.py \
    --dataset OUTSIDE15k \
    --model pspnet_r50 \
    --strategy gen_Weather_Effect_Generator \
    --real-gen-ratio 0.5 \
    --domain-filter clear_day \
    --max-iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/outside15k/pspnet_r50_clear_day/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --weights_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/outside15k/pspnet_r50_clear_day/iter_80000.pth \
        --config_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/outside15k/pspnet_r50_clear_day/training_config.py \
        --dataset OUTSIDE15k \
        --output_dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/outside15k/pspnet_r50_clear_day/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: outside15k/pspnet_r50_clear_day at $(date)"


echo "----------------------------------------"
echo "Training: outside15k/segformer_mit-b5"
echo "Strategy: gen_Weather_Effect_Generator"
echo "Real/Gen Ratio: 0.5"
echo "Started: $(date)"
echo "----------------------------------------"

# Train model
python unified_training.py \
    --dataset OUTSIDE15k \
    --model segformer_mit-b5 \
    --strategy gen_Weather_Effect_Generator \
    --real-gen-ratio 0.5 \
     \
    --max-iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/outside15k/segformer_mit-b5/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --weights_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/outside15k/segformer_mit-b5/iter_80000.pth \
        --config_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/outside15k/segformer_mit-b5/training_config.py \
        --dataset OUTSIDE15k \
        --output_dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/outside15k/segformer_mit-b5/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: outside15k/segformer_mit-b5 at $(date)"


echo "----------------------------------------"
echo "Training: outside15k/segformer_mit-b5_clear_day"
echo "Strategy: gen_Weather_Effect_Generator"
echo "Real/Gen Ratio: 0.5"
echo "Started: $(date)"
echo "----------------------------------------"

# Train model
python unified_training.py \
    --dataset OUTSIDE15k \
    --model segformer_mit-b5 \
    --strategy gen_Weather_Effect_Generator \
    --real-gen-ratio 0.5 \
    --domain-filter clear_day \
    --max-iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/outside15k/segformer_mit-b5_clear_day/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --weights_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/outside15k/segformer_mit-b5_clear_day/iter_80000.pth \
        --config_path /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/outside15k/segformer_mit-b5_clear_day/training_config.py \
        --dataset OUTSIDE15k \
        --output_dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_Weather_Effect_Generator/outside15k/segformer_mit-b5_clear_day/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: outside15k/segformer_mit-b5_clear_day at $(date)"


echo "========================================"
echo "Job completed: $(date)"
echo "========================================"
