#!/bin/bash
#BSUB -J train_gen_LANIT_iddaw_cd
#BSUB -o /home/mima2416/repositories/PROVE/logs/train_gen_LANIT_iddaw_cd_%J.out
#BSUB -e /home/mima2416/repositories/PROVE/logs/train_gen_LANIT_iddaw_cd_%J.err
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
echo "Training job: gen_LANIT on IDD-AW"
echo "Started: $(date)"
echo "========================================"


echo "----------------------------------------"
echo "Training: IDD-AW/deeplabv3plus_r50"
echo "Strategy: gen_LANIT"
echo "Real/Gen Ratio: 0.5"
echo "Started: $(date)"
echo "----------------------------------------"

# Pre-flight checks
WEIGHTS_PATH="/scratch/aaa_exchange/AWARE/WEIGHTS/gen_LANIT/iddaw_cd/deeplabv3plus_r50_ratio0p50"
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
            --dataset IDD-AW \
            --model deeplabv3plus_r50 \
            --strategy gen_LANIT \
            --real-gen-ratio 0.5 \
            --domain-filter clear_day \
            --max-iters 80000
        
        # Remove lock
        rm -f "$LOCK_FILE"
        
        if [ -f "$CHECKPOINT" ]; then
            echo "SUCCESS: Training complete for IDD-AW/deeplabv3plus_r50"
        else
            echo "WARNING: Checkpoint not found after training"
        fi
    else
        echo "SKIP: Another process is training IDD-AW/deeplabv3plus_r50"
    fi
fi

echo "Finished: IDD-AW/deeplabv3plus_r50 at $(date)"

echo "----------------------------------------"
echo "Training: IDD-AW/pspnet_r50"
echo "Strategy: gen_LANIT"
echo "Real/Gen Ratio: 0.5"
echo "Started: $(date)"
echo "----------------------------------------"

# Pre-flight checks
WEIGHTS_PATH="/scratch/aaa_exchange/AWARE/WEIGHTS/gen_LANIT/iddaw_cd/pspnet_r50_ratio0p50"
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
            --dataset IDD-AW \
            --model pspnet_r50 \
            --strategy gen_LANIT \
            --real-gen-ratio 0.5 \
            --domain-filter clear_day \
            --max-iters 80000
        
        # Remove lock
        rm -f "$LOCK_FILE"
        
        if [ -f "$CHECKPOINT" ]; then
            echo "SUCCESS: Training complete for IDD-AW/pspnet_r50"
        else
            echo "WARNING: Checkpoint not found after training"
        fi
    else
        echo "SKIP: Another process is training IDD-AW/pspnet_r50"
    fi
fi

echo "Finished: IDD-AW/pspnet_r50 at $(date)"

echo "----------------------------------------"
echo "Training: IDD-AW/segformer_mit-b2"
echo "Strategy: gen_LANIT"
echo "Real/Gen Ratio: 0.5"
echo "Started: $(date)"
echo "----------------------------------------"

# Pre-flight checks
WEIGHTS_PATH="/scratch/aaa_exchange/AWARE/WEIGHTS/gen_LANIT/iddaw_cd/segformer_mit-b2_ratio0p50"
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
            --dataset IDD-AW \
            --model segformer_mit-b2 \
            --strategy gen_LANIT \
            --real-gen-ratio 0.5 \
            --domain-filter clear_day \
            --max-iters 80000
        
        # Remove lock
        rm -f "$LOCK_FILE"
        
        if [ -f "$CHECKPOINT" ]; then
            echo "SUCCESS: Training complete for IDD-AW/segformer_mit-b2"
        else
            echo "WARNING: Checkpoint not found after training"
        fi
    else
        echo "SKIP: Another process is training IDD-AW/segformer_mit-b2"
    fi
fi

echo "Finished: IDD-AW/segformer_mit-b2 at $(date)"

echo "========================================"
echo "All training complete"
echo "Finished: $(date)"
echo "========================================"
