#!/bin/bash
#BSUB -J retrain_gen_LANIT_mapillaryvistas
#BSUB -o /home/mima2416/repositories/PROVE/logs/retrain/retrain_gen_LANIT_mapillaryvistas_%J.out
#BSUB -e /home/mima2416/repositories/PROVE/logs/retrain/retrain_gen_LANIT_mapillaryvistas_%J.err
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
echo "Retraining job: retrain_gen_LANIT_mapillaryvistas"
echo "Strategy: gen_LANIT"
echo "Dataset: mapillaryvistas"
echo "Started: $(date)"
echo "========================================"


echo "----------------------------------------"
echo "Training: mapillaryvistas/deeplabv3plus_r50_ratio0p50"
echo "Strategy: gen_LANIT"
echo "Real/Gen Ratio: 0.5"
echo "Started: $(date)"
echo "----------------------------------------"

# Train model
python unified_training.py \
    --dataset MapillaryVistas \
    --model deeplabv3plus_r50 \
    --strategy gen_LANIT \
    --real-gen-ratio 0.5 \
    --domain-filter clear_day \
    --max-iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_LANIT/mapillaryvistas_cd/deeplabv3plus_r50_ratio0p50/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --checkpoint /scratch/aaa_exchange/AWARE/WEIGHTS/gen_LANIT/mapillaryvistas_cd/deeplabv3plus_r50_ratio0p50/iter_80000.pth \
        --config /scratch/aaa_exchange/AWARE/WEIGHTS/gen_LANIT/mapillaryvistas_cd/deeplabv3plus_r50_ratio0p50/training_config.py \
        --dataset MapillaryVistas \
        --output-dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_LANIT/mapillaryvistas_cd/deeplabv3plus_r50_ratio0p50/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: mapillaryvistas/deeplabv3plus_r50_ratio0p50 at $(date)"


echo "----------------------------------------"
echo "Training: mapillaryvistas/pspnet_r50_ratio0p50"
echo "Strategy: gen_LANIT"
echo "Real/Gen Ratio: 0.5"
echo "Started: $(date)"
echo "----------------------------------------"

# Train model
python unified_training.py \
    --dataset MapillaryVistas \
    --model pspnet_r50 \
    --strategy gen_LANIT \
    --real-gen-ratio 0.5 \
    --domain-filter clear_day \
    --max-iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_LANIT/mapillaryvistas_cd/pspnet_r50_ratio0p50/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --checkpoint /scratch/aaa_exchange/AWARE/WEIGHTS/gen_LANIT/mapillaryvistas_cd/pspnet_r50_ratio0p50/iter_80000.pth \
        --config /scratch/aaa_exchange/AWARE/WEIGHTS/gen_LANIT/mapillaryvistas_cd/pspnet_r50_ratio0p50/training_config.py \
        --dataset MapillaryVistas \
        --output-dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_LANIT/mapillaryvistas_cd/pspnet_r50_ratio0p50/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: mapillaryvistas/pspnet_r50_ratio0p50 at $(date)"


echo "----------------------------------------"
echo "Training: mapillaryvistas/segformer_mit-b5_ratio0p50"
echo "Strategy: gen_LANIT"
echo "Real/Gen Ratio: 0.5"
echo "Started: $(date)"
echo "----------------------------------------"

# Train model
python unified_training.py \
    --dataset MapillaryVistas \
    --model segformer_mit-b5 \
    --strategy gen_LANIT \
    --real-gen-ratio 0.5 \
    --domain-filter clear_day \
    --max-iters 80000

# Test model
if [ -f "/scratch/aaa_exchange/AWARE/WEIGHTS/gen_LANIT/mapillaryvistas_cd/segformer_mit-b5_ratio0p50/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \
        --checkpoint /scratch/aaa_exchange/AWARE/WEIGHTS/gen_LANIT/mapillaryvistas_cd/segformer_mit-b5_ratio0p50/iter_80000.pth \
        --config /scratch/aaa_exchange/AWARE/WEIGHTS/gen_LANIT/mapillaryvistas_cd/segformer_mit-b5_ratio0p50/training_config.py \
        --dataset MapillaryVistas \
        --output-dir /scratch/aaa_exchange/AWARE/WEIGHTS/gen_LANIT/mapillaryvistas_cd/segformer_mit-b5_ratio0p50/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: mapillaryvistas/segformer_mit-b5_ratio0p50 at $(date)"


echo "========================================"
echo "Job completed: $(date)"
echo "========================================"
