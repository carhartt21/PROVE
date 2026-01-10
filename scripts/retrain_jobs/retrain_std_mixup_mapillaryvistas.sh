#!/bin/bash
#BSUB -J retrain_std_mixup_mapillaryvistas
#BSUB -o /home/mima2416/repositories/PROVE/logs/retrain/retrain_std_mixup_mapillaryvistas_%J.out
#BSUB -e /home/mima2416/repositories/PROVE/logs/retrain/retrain_std_mixup_mapillaryvistas_%J.err
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
echo "Retraining job: retrain_std_mixup_mapillaryvistas"
echo "Strategy: std_mixup"
echo "Dataset: mapillaryvistas"
echo "Started: $(date)"
echo "========================================"


echo "----------------------------------------"
echo "Training: mapillaryvistas/deeplabv3plus_r50"
echo "Strategy: std_mixup"
echo "Real/Gen Ratio: 1.0"
echo "Started: $(date)"
echo "----------------------------------------"

# Pre-flight checks
WEIGHTS_PATH="/scratch/aaa_exchange/AWARE/WEIGHTS/std_mixup/mapillaryvistas_cd/deeplabv3plus_r50"
LOCK_FILE="${WEIGHTS_PATH}/.training_lock"
CHECKPOINT="${WEIGHTS_PATH}/iter_80000.pth"

# Check if final checkpoint already exists
if [ -f "$CHECKPOINT" ]; then
    echo "SKIP: Final checkpoint already exists: $CHECKPOINT"
    echo "Finished: mapillaryvistas/deeplabv3plus_r50 at $(date)"
else
    # Try to create lock file (atomic operation)
    mkdir -p "$WEIGHTS_PATH"
    if ( set -o noclobber; echo "$$" > "$LOCK_FILE" ) 2>/dev/null; then
        trap "rm -f '$LOCK_FILE'" EXIT
        echo "Lock acquired. Starting training..."
        
        # Train model
        python unified_training.py \
            --dataset MapillaryVistas \
            --model deeplabv3plus_r50 \
            --strategy std_mixup \
            --real-gen-ratio 1.0 \
            --domain-filter clear_day \
            --max-iters 80000
        
        # Remove lock and test model
        rm -f "$LOCK_FILE"
        
        if [ -f "$CHECKPOINT" ]; then
            echo "Training complete. Running test..."
            python fine_grained_test.py \
                --checkpoint $CHECKPOINT \
                --config ${WEIGHTS_PATH}/training_config.py \
                --dataset MapillaryVistas \
                --output-dir ${WEIGHTS_PATH}/test_results_detailed
        else
            echo "ERROR: Training failed - checkpoint not found"
        fi
        
        echo "Finished: mapillaryvistas/deeplabv3plus_r50 at $(date)"
    else
        LOCK_PID=$(cat "$LOCK_FILE" 2>/dev/null)
        echo "SKIP: Another job (PID: $LOCK_PID) is already training this model"
        echo "Lock file: $LOCK_FILE"
        echo "Finished: mapillaryvistas/deeplabv3plus_r50 at $(date)"
    fi
fi


echo "----------------------------------------"
echo "Training: mapillaryvistas/pspnet_r50"
echo "Strategy: std_mixup"
echo "Real/Gen Ratio: 1.0"
echo "Started: $(date)"
echo "----------------------------------------"

# Pre-flight checks
WEIGHTS_PATH="/scratch/aaa_exchange/AWARE/WEIGHTS/std_mixup/mapillaryvistas_cd/pspnet_r50"
LOCK_FILE="${WEIGHTS_PATH}/.training_lock"
CHECKPOINT="${WEIGHTS_PATH}/iter_80000.pth"

# Check if final checkpoint already exists
if [ -f "$CHECKPOINT" ]; then
    echo "SKIP: Final checkpoint already exists: $CHECKPOINT"
    echo "Finished: mapillaryvistas/pspnet_r50 at $(date)"
else
    # Try to create lock file (atomic operation)
    mkdir -p "$WEIGHTS_PATH"
    if ( set -o noclobber; echo "$$" > "$LOCK_FILE" ) 2>/dev/null; then
        trap "rm -f '$LOCK_FILE'" EXIT
        echo "Lock acquired. Starting training..."
        
        # Train model
        python unified_training.py \
            --dataset MapillaryVistas \
            --model pspnet_r50 \
            --strategy std_mixup \
            --real-gen-ratio 1.0 \
            --domain-filter clear_day \
            --max-iters 80000
        
        # Remove lock and test model
        rm -f "$LOCK_FILE"
        
        if [ -f "$CHECKPOINT" ]; then
            echo "Training complete. Running test..."
            python fine_grained_test.py \
                --checkpoint $CHECKPOINT \
                --config ${WEIGHTS_PATH}/training_config.py \
                --dataset MapillaryVistas \
                --output-dir ${WEIGHTS_PATH}/test_results_detailed
        else
            echo "ERROR: Training failed - checkpoint not found"
        fi
        
        echo "Finished: mapillaryvistas/pspnet_r50 at $(date)"
    else
        LOCK_PID=$(cat "$LOCK_FILE" 2>/dev/null)
        echo "SKIP: Another job (PID: $LOCK_PID) is already training this model"
        echo "Lock file: $LOCK_FILE"
        echo "Finished: mapillaryvistas/pspnet_r50 at $(date)"
    fi
fi


echo "----------------------------------------"
echo "Training: mapillaryvistas/segformer_mit-b5"
echo "Strategy: std_mixup"
echo "Real/Gen Ratio: 1.0"
echo "Started: $(date)"
echo "----------------------------------------"

# Pre-flight checks
WEIGHTS_PATH="/scratch/aaa_exchange/AWARE/WEIGHTS/std_mixup/mapillaryvistas_cd/segformer_mit-b5"
LOCK_FILE="${WEIGHTS_PATH}/.training_lock"
CHECKPOINT="${WEIGHTS_PATH}/iter_80000.pth"

# Check if final checkpoint already exists
if [ -f "$CHECKPOINT" ]; then
    echo "SKIP: Final checkpoint already exists: $CHECKPOINT"
    echo "Finished: mapillaryvistas/segformer_mit-b5 at $(date)"
else
    # Try to create lock file (atomic operation)
    mkdir -p "$WEIGHTS_PATH"
    if ( set -o noclobber; echo "$$" > "$LOCK_FILE" ) 2>/dev/null; then
        trap "rm -f '$LOCK_FILE'" EXIT
        echo "Lock acquired. Starting training..."
        
        # Train model
        python unified_training.py \
            --dataset MapillaryVistas \
            --model segformer_mit-b5 \
            --strategy std_mixup \
            --real-gen-ratio 1.0 \
            --domain-filter clear_day \
            --max-iters 80000
        
        # Remove lock and test model
        rm -f "$LOCK_FILE"
        
        if [ -f "$CHECKPOINT" ]; then
            echo "Training complete. Running test..."
            python fine_grained_test.py \
                --checkpoint $CHECKPOINT \
                --config ${WEIGHTS_PATH}/training_config.py \
                --dataset MapillaryVistas \
                --output-dir ${WEIGHTS_PATH}/test_results_detailed
        else
            echo "ERROR: Training failed - checkpoint not found"
        fi
        
        echo "Finished: mapillaryvistas/segformer_mit-b5 at $(date)"
    else
        LOCK_PID=$(cat "$LOCK_FILE" 2>/dev/null)
        echo "SKIP: Another job (PID: $LOCK_PID) is already training this model"
        echo "Lock file: $LOCK_FILE"
        echo "Finished: mapillaryvistas/segformer_mit-b5 at $(date)"
    fi
fi


echo "========================================"
echo "Job completed: $(date)"
echo "========================================"
