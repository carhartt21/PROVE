#!/bin/bash
#BSUB -J retrain_gen_LANIT_outside15k
#BSUB -o /home/mima2416/repositories/PROVE/logs/retrain/retrain_gen_LANIT_outside15k_%J.out
#BSUB -e /home/mima2416/repositories/PROVE/logs/retrain/retrain_gen_LANIT_outside15k_%J.err
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
echo "Retraining job: retrain_gen_LANIT_outside15k"
echo "Strategy: gen_LANIT"
echo "Dataset: outside15k"
echo "Started: $(date)"
echo "========================================"


echo "----------------------------------------"
echo "Training: outside15k/deeplabv3plus_r50_ratio0p50"
echo "Strategy: gen_LANIT"
echo "Real/Gen Ratio: 0.5"
echo "Started: $(date)"
echo "----------------------------------------"

# Pre-flight checks
WEIGHTS_PATH="/scratch/aaa_exchange/AWARE/WEIGHTS/gen_LANIT/outside15k_cd/deeplabv3plus_r50_ratio0p50"
LOCK_FILE="${WEIGHTS_PATH}/.training_lock"
CHECKPOINT="${WEIGHTS_PATH}/iter_80000.pth"

# Check if final checkpoint already exists
if [ -f "$CHECKPOINT" ]; then
    echo "SKIP: Final checkpoint already exists: $CHECKPOINT"
    echo "Finished: outside15k/deeplabv3plus_r50_ratio0p50 at $(date)"
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
            --strategy gen_LANIT \
            --real-gen-ratio 0.5 \
            --domain-filter clear_day \
            --max-iters 80000
        
        # Remove lock and test model
        rm -f "$LOCK_FILE"
        
        if [ -f "$CHECKPOINT" ]; then
            echo "Training complete. Running test..."
            python fine_grained_test.py \
                --checkpoint $CHECKPOINT \
                --config ${WEIGHTS_PATH}/training_config.py \
                --dataset OUTSIDE15k \
                --output-dir ${WEIGHTS_PATH}/test_results_detailed
        else
            echo "ERROR: Training failed - checkpoint not found"
        fi
        
        echo "Finished: outside15k/deeplabv3plus_r50_ratio0p50 at $(date)"
    else
        LOCK_PID=$(cat "$LOCK_FILE" 2>/dev/null)
        echo "SKIP: Another job (PID: $LOCK_PID) is already training this model"
        echo "Lock file: $LOCK_FILE"
        echo "Finished: outside15k/deeplabv3plus_r50_ratio0p50 at $(date)"
    fi
fi


echo "----------------------------------------"
echo "Training: outside15k/pspnet_r50_ratio0p50"
echo "Strategy: gen_LANIT"
echo "Real/Gen Ratio: 0.5"
echo "Started: $(date)"
echo "----------------------------------------"

# Pre-flight checks
WEIGHTS_PATH="/scratch/aaa_exchange/AWARE/WEIGHTS/gen_LANIT/outside15k_cd/pspnet_r50_ratio0p50"
LOCK_FILE="${WEIGHTS_PATH}/.training_lock"
CHECKPOINT="${WEIGHTS_PATH}/iter_80000.pth"

# Check if final checkpoint already exists
if [ -f "$CHECKPOINT" ]; then
    echo "SKIP: Final checkpoint already exists: $CHECKPOINT"
    echo "Finished: outside15k/pspnet_r50_ratio0p50 at $(date)"
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
            --strategy gen_LANIT \
            --real-gen-ratio 0.5 \
            --domain-filter clear_day \
            --max-iters 80000
        
        # Remove lock and test model
        rm -f "$LOCK_FILE"
        
        if [ -f "$CHECKPOINT" ]; then
            echo "Training complete. Running test..."
            python fine_grained_test.py \
                --checkpoint $CHECKPOINT \
                --config ${WEIGHTS_PATH}/training_config.py \
                --dataset OUTSIDE15k \
                --output-dir ${WEIGHTS_PATH}/test_results_detailed
        else
            echo "ERROR: Training failed - checkpoint not found"
        fi
        
        echo "Finished: outside15k/pspnet_r50_ratio0p50 at $(date)"
    else
        LOCK_PID=$(cat "$LOCK_FILE" 2>/dev/null)
        echo "SKIP: Another job (PID: $LOCK_PID) is already training this model"
        echo "Lock file: $LOCK_FILE"
        echo "Finished: outside15k/pspnet_r50_ratio0p50 at $(date)"
    fi
fi


echo "----------------------------------------"
echo "Training: outside15k/segformer_mit-b5_ratio0p50"
echo "Strategy: gen_LANIT"
echo "Real/Gen Ratio: 0.5"
echo "Started: $(date)"
echo "----------------------------------------"

# Pre-flight checks
WEIGHTS_PATH="/scratch/aaa_exchange/AWARE/WEIGHTS/gen_LANIT/outside15k_cd/segformer_mit-b5_ratio0p50"
LOCK_FILE="${WEIGHTS_PATH}/.training_lock"
CHECKPOINT="${WEIGHTS_PATH}/iter_80000.pth"

# Check if final checkpoint already exists
if [ -f "$CHECKPOINT" ]; then
    echo "SKIP: Final checkpoint already exists: $CHECKPOINT"
    echo "Finished: outside15k/segformer_mit-b5_ratio0p50 at $(date)"
else
    # Try to create lock file (atomic operation)
    mkdir -p "$WEIGHTS_PATH"
    if ( set -o noclobber; echo "$$" > "$LOCK_FILE" ) 2>/dev/null; then
        trap "rm -f '$LOCK_FILE'" EXIT
        echo "Lock acquired. Starting training..."
        
        # Train model
        python unified_training.py \
            --dataset OUTSIDE15k \
            --model segformer_mit-b5 \
            --strategy gen_LANIT \
            --real-gen-ratio 0.5 \
            --domain-filter clear_day \
            --max-iters 80000
        
        # Remove lock and test model
        rm -f "$LOCK_FILE"
        
        if [ -f "$CHECKPOINT" ]; then
            echo "Training complete. Running test..."
            python fine_grained_test.py \
                --checkpoint $CHECKPOINT \
                --config ${WEIGHTS_PATH}/training_config.py \
                --dataset OUTSIDE15k \
                --output-dir ${WEIGHTS_PATH}/test_results_detailed
        else
            echo "ERROR: Training failed - checkpoint not found"
        fi
        
        echo "Finished: outside15k/segformer_mit-b5_ratio0p50 at $(date)"
    else
        LOCK_PID=$(cat "$LOCK_FILE" 2>/dev/null)
        echo "SKIP: Another job (PID: $LOCK_PID) is already training this model"
        echo "Lock file: $LOCK_FILE"
        echo "Finished: outside15k/segformer_mit-b5_ratio0p50 at $(date)"
    fi
fi


echo "========================================"
echo "Job completed: $(date)"
echo "========================================"
