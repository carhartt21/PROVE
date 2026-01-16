#!/bin/bash
#BSUB -J train_gen_Qwen-Image-Edit_outside15k_cd
#BSUB -o /home/mima2416/repositories/PROVE/logs/train_gen_Qwen-Image-Edit_outside15k_cd_%J.out
#BSUB -e /home/mima2416/repositories/PROVE/logs/train_gen_Qwen-Image-Edit_outside15k_cd_%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=16000]"
#BSUB -gpu "num=1:mode=exclusive_process:gmem=20G"
#BSUB -W 24:00
#BSUB -q BatchGPU

# Activate environment
source ~/.bashrc
mamba activate prove

cd /home/mima2416/repositories/PROVE

echo "========================================"
echo "Training job: gen_Qwen-Image-Edit on OUTSIDE15k"
echo "Started: $(date)"
echo "========================================"


echo "----------------------------------------"
echo "Training: OUTSIDE15k/deeplabv3plus_r50"
echo "Strategy: gen_Qwen-Image-Edit"
echo "Real/Gen Ratio: 0.5"
echo "Started: $(date)"
echo "----------------------------------------"

# Pre-flight checks
WEIGHTS_PATH="/scratch/aaa_exchange/AWARE/WEIGHTS/gen_Qwen-Image-Edit/outside15k_cd/deeplabv3plus_r50_ratio0p50"
LOCK_FILE="${WEIGHTS_PATH}/.training_lock"
CHECKPOINT="${WEIGHTS_PATH}/iter_80000.pth"

# Check if final checkpoint already exists
if [ -f "$CHECKPOINT" ]; then
    echo "SKIP: Final checkpoint already exists: $CHECKPOINT"
else
    # Try to create lock file (atomic operation)
    mkdir -p "$WEIGHTS_PATH"
    if ( set -o noclobber; echo "$$" > "$LOCK_FILE" ) 2>/dev/null; then
        trap "rm -f '$LOCK_FILE'" EXIT
        echo "Lock acquired. Starting training..."
        
        # Train model
        python unified_training.py \
            --dataset OUTSIDE15k \
            --model deeplabv3plus_r50 \
            --strategy gen_Qwen-Image-Edit \
            --real-gen-ratio 0.5 \
            --domain-filter clear_day \
            --max-iters 80000
        
        # Remove lock
        rm -f "$LOCK_FILE"
        
        if [ -f "$CHECKPOINT" ]; then
            echo "SUCCESS: Training complete for OUTSIDE15k/deeplabv3plus_r50"
        else
            echo "WARNING: Checkpoint not found after training"
        fi
    else
        echo "SKIP: Another process is training OUTSIDE15k/deeplabv3plus_r50"
    fi
fi

echo "Finished: OUTSIDE15k/deeplabv3plus_r50 at $(date)"

echo "----------------------------------------"
echo "Training: OUTSIDE15k/pspnet_r50"
echo "Strategy: gen_Qwen-Image-Edit"
echo "Real/Gen Ratio: 0.5"
echo "Started: $(date)"
echo "----------------------------------------"

# Pre-flight checks
WEIGHTS_PATH="/scratch/aaa_exchange/AWARE/WEIGHTS/gen_Qwen-Image-Edit/outside15k_cd/pspnet_r50_ratio0p50"
LOCK_FILE="${WEIGHTS_PATH}/.training_lock"
CHECKPOINT="${WEIGHTS_PATH}/iter_80000.pth"

# Check if final checkpoint already exists
if [ -f "$CHECKPOINT" ]; then
    echo "SKIP: Final checkpoint already exists: $CHECKPOINT"
else
    # Try to create lock file (atomic operation)
    mkdir -p "$WEIGHTS_PATH"
    if ( set -o noclobber; echo "$$" > "$LOCK_FILE" ) 2>/dev/null; then
        trap "rm -f '$LOCK_FILE'" EXIT
        echo "Lock acquired. Starting training..."
        
        # Train model
        python unified_training.py \
            --dataset OUTSIDE15k \
            --model pspnet_r50 \
            --strategy gen_Qwen-Image-Edit \
            --real-gen-ratio 0.5 \
            --domain-filter clear_day \
            --max-iters 80000
        
        # Remove lock
        rm -f "$LOCK_FILE"
        
        if [ -f "$CHECKPOINT" ]; then
            echo "SUCCESS: Training complete for OUTSIDE15k/pspnet_r50"
        else
            echo "WARNING: Checkpoint not found after training"
        fi
    else
        echo "SKIP: Another process is training OUTSIDE15k/pspnet_r50"
    fi
fi

echo "Finished: OUTSIDE15k/pspnet_r50 at $(date)"

echo "----------------------------------------"
echo "Training: OUTSIDE15k/segformer_mit-b2"
echo "Strategy: gen_Qwen-Image-Edit"
echo "Real/Gen Ratio: 0.5"
echo "Started: $(date)"
echo "----------------------------------------"

# Pre-flight checks
WEIGHTS_PATH="/scratch/aaa_exchange/AWARE/WEIGHTS/gen_Qwen-Image-Edit/outside15k_cd/segformer_mit-b2_ratio0p50"
LOCK_FILE="${WEIGHTS_PATH}/.training_lock"
CHECKPOINT="${WEIGHTS_PATH}/iter_80000.pth"

# Check if final checkpoint already exists
if [ -f "$CHECKPOINT" ]; then
    echo "SKIP: Final checkpoint already exists: $CHECKPOINT"
else
    # Try to create lock file (atomic operation)
    mkdir -p "$WEIGHTS_PATH"
    if ( set -o noclobber; echo "$$" > "$LOCK_FILE" ) 2>/dev/null; then
        trap "rm -f '$LOCK_FILE'" EXIT
        echo "Lock acquired. Starting training..."
        
        # Train model
        python unified_training.py \
            --dataset OUTSIDE15k \
            --model segformer_mit-b2 \
            --strategy gen_Qwen-Image-Edit \
            --real-gen-ratio 0.5 \
            --domain-filter clear_day \
            --max-iters 80000
        
        # Remove lock
        rm -f "$LOCK_FILE"
        
        if [ -f "$CHECKPOINT" ]; then
            echo "SUCCESS: Training complete for OUTSIDE15k/segformer_mit-b2"
        else
            echo "WARNING: Checkpoint not found after training"
        fi
    else
        echo "SKIP: Another process is training OUTSIDE15k/segformer_mit-b2"
    fi
fi

echo "Finished: OUTSIDE15k/segformer_mit-b2 at $(date)"

echo "========================================"
echo "All training complete"
echo "Finished: $(date)"
echo "========================================"
