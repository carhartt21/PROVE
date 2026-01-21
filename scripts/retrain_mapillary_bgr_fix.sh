#!/bin/bash
# =============================================================================
# Retrain all MapillaryVistas models after BGR/RGB fix
# =============================================================================
# Bug: MapillaryRGBToClassId was treating BGR input as RGB, causing wrong class mappings
# Fix: custom_transforms.py now correctly handles BGR input from mmseg LoadAnnotations
#
# This script submits training+testing jobs for all MapillaryVistas models
# in both Stage 1 (clear_day) and Stage 2 (all domains)
#
# Usage:
#   ./scripts/retrain_mapillary_bgr_fix.sh [--dry-run] [--stage 1|2|both] [--limit N]
#
# Features:
#   - Pre-flight checks: Skip if final checkpoint exists
#   - Training locks: Atomic lock file prevents concurrent training
#   - Safe for multi-machine submission
# =============================================================================

set -e

PROVE_DIR="/home/mima2416/repositories/PROVE"
LOG_DIR="${PROVE_DIR}/logs"
QUEUE="BatchGPU"
GPU_MEM="24G"
GPU_MODE="exclusive_process"
NUM_CPUS=8
WALL_TIME="24:00"
MEM_LIMIT="16000"

DRY_RUN=false
STAGE="both"
LIMIT=0  # 0 = no limit
SUBMITTED=0

# Models
MODELS=("deeplabv3plus_r50" "pspnet_r50" "segformer_mit-b5")

# Strategies - baseline and augmentation strategies
BASELINE_STRATEGIES=("baseline")
STD_STRATEGIES=("std_autoaugment" "std_cutmix" "std_mixup" "std_randaugment" "photometric_distort")

# Generative strategies with ratio 0.5
GEN_STRATEGIES=(
    "gen_albumentations_weather"
    "gen_Attribute_Hallucination"
    "gen_augmenters"
    "gen_automold"
    "gen_CNetSeg"
    "gen_CUT"
    "gen_cyclediffusion"
    "gen_cycleGAN"
    "gen_flux_kontext"
    "gen_Img2Img"
    "gen_IP2P"
    "gen_LANIT"
    "gen_Qwen_Image_Edit"
    "gen_stargan_v2"
    "gen_step1x_new"
    "gen_step1x_v1p2"
    "gen_SUSTechGAN"
    "gen_TSIT"
    "gen_UniControl"
    "gen_VisualCloze"
    "gen_Weather_Effect_Generator"
)

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --stage) STAGE="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run        Show commands without submitting"
            echo "  --stage 1|2|both Stage to retrain (default: both)"
            echo "  --limit N        Limit number of jobs to submit (default: no limit)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"

# Generate the job script content with pre-flight checks and lock
generate_job_script() {
    local STRATEGY="$1"
    local MODEL="$2"
    local DOMAIN_FILTER="$3"  # empty for stage 2, "clear_day" for stage 1
    local RATIO="$4"  # for generative strategies
    local WEIGHTS_ROOT="$5"  # WEIGHTS or WEIGHTS_STAGE_2
    
    # Build the weights path
    local WEIGHTS_PATH="${WEIGHTS_ROOT}/${STRATEGY}/mapillaryvistas"
    if [ -n "$RATIO" ]; then
        local RATIO_SUFFIX=$(echo "$RATIO" | sed 's/\./_/; s/^0_/0p/')
        WEIGHTS_PATH="${WEIGHTS_PATH}/${MODEL}_ratio${RATIO_SUFFIX}"
    else
        WEIGHTS_PATH="${WEIGHTS_PATH}/${MODEL}"
    fi
    
    # Build training command args
    local TRAIN_ARGS="--dataset MapillaryVistas --model ${MODEL} --strategy ${STRATEGY}"
    if [ -n "$DOMAIN_FILTER" ]; then
        TRAIN_ARGS="${TRAIN_ARGS} --domain-filter ${DOMAIN_FILTER}"
    fi
    if [ -n "$RATIO" ]; then
        TRAIN_ARGS="${TRAIN_ARGS} --real-gen-ratio ${RATIO}"
    fi
    TRAIN_ARGS="${TRAIN_ARGS} --max-iters 80000"
    
    # Generate the script content
    cat << SCRIPT_EOF
#!/bin/bash
#BSUB -J rt_map_${STRATEGY}_${MODEL}
#BSUB -o ${LOG_DIR}/retrain_mapillary_${STRATEGY}_${MODEL}_%J.out
#BSUB -e ${LOG_DIR}/retrain_mapillary_${STRATEGY}_${MODEL}_%J.err
#BSUB -n ${NUM_CPUS}
#BSUB -R "rusage[mem=${MEM_LIMIT}]"
#BSUB -gpu "num=1:mode=${GPU_MODE}:gmem=${GPU_MEM}"
#BSUB -W ${WALL_TIME}
#BSUB -q ${QUEUE}

# Retrain: MapillaryVistas / ${STRATEGY} / ${MODEL}
# Fix: BGR/RGB mismatch in MapillaryRGBToClassId

source ~/.bashrc
mamba activate prove

cd ${PROVE_DIR}

echo "========================================"
echo "Retrain MapillaryVistas (BGR/RGB Fix)"
echo "Strategy: ${STRATEGY}"
echo "Model: ${MODEL}"
echo "Domain Filter: ${DOMAIN_FILTER:-all}"
echo "Ratio: ${RATIO:-N/A}"
echo "Started: \$(date)"
echo "========================================"

# Pre-flight checks
WEIGHTS_PATH="/scratch/aaa_exchange/AWARE/${WEIGHTS_PATH}"
LOCK_FILE="\${WEIGHTS_PATH}/.training_lock"
CHECKPOINT="\${WEIGHTS_PATH}/iter_80000.pth"

# Check if final checkpoint already exists
if [ -f "\$CHECKPOINT" ]; then
    echo "SKIP: Final checkpoint already exists: \$CHECKPOINT"
else
    # Try to create lock file (atomic operation)
    mkdir -p "\$WEIGHTS_PATH"
    if ( set -o noclobber; echo "\$\$" > "\$LOCK_FILE" ) 2>/dev/null; then
        trap "rm -f '\$LOCK_FILE'" EXIT
        echo "Lock acquired. Starting training..."
        
        # Train model
        python unified_training.py ${TRAIN_ARGS}
        
        # Remove lock
        rm -f "\$LOCK_FILE"
        
        if [ -f "\$CHECKPOINT" ]; then
            echo "SUCCESS: Training complete for MapillaryVistas/${STRATEGY}/${MODEL}"
        else
            echo "WARNING: Checkpoint not found after training"
        fi
    else
        echo "SKIP: Another process is training MapillaryVistas/${STRATEGY}/${MODEL}"
    fi
fi

echo "Finished: \$(date)"
SCRIPT_EOF
}

submit_job() {
    local STRATEGY="$1"
    local MODEL="$2"
    local DOMAIN_FILTER="$3"  # empty for stage 2, "clear_day" for stage 1
    local RATIO="$4"  # for generative strategies
    
    # Check limit
    if [ "$LIMIT" -gt 0 ] && [ "$SUBMITTED" -ge "$LIMIT" ]; then
        echo "[LIMIT REACHED] Skipping: $STRATEGY $MODEL $DOMAIN_FILTER"
        return
    fi
    
    # Determine weights root based on domain filter
    local WEIGHTS_ROOT
    if [ -n "$DOMAIN_FILTER" ]; then
        WEIGHTS_ROOT="WEIGHTS"
    else
        WEIGHTS_ROOT="WEIGHTS_STAGE_2"
    fi
    
    # Build job name
    JOB_NAME="rt_map_${STRATEGY}_${MODEL}"
    if [ -n "$DOMAIN_FILTER" ]; then
        JOB_NAME="${JOB_NAME}_cd"
    else
        JOB_NAME="${JOB_NAME}_ad"
    fi
    
    echo "=== ${JOB_NAME} ==="
    echo "Strategy: ${STRATEGY}, Model: ${MODEL}"
    [ -n "$DOMAIN_FILTER" ] && echo "Domain Filter: ${DOMAIN_FILTER}"
    [ -n "$RATIO" ] && echo "Ratio: ${RATIO}"
    
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would submit job with pre-flight checks and lock"
        echo "Weights: ${WEIGHTS_ROOT}/${STRATEGY}/mapillaryvistas/..."
    else
        # Create temp script file
        SCRIPT_FILE=$(mktemp)
        generate_job_script "$STRATEGY" "$MODEL" "$DOMAIN_FILTER" "$RATIO" "$WEIGHTS_ROOT" > "$SCRIPT_FILE"
        
        # Submit the job
        bsub < "$SCRIPT_FILE"
        rm -f "$SCRIPT_FILE"
        echo "Job submitted!"
    fi
    
    ((SUBMITTED++)) || true
    echo ""
}

echo "=========================================="
echo "MapillaryVistas Retraining (BGR/RGB Fix)"
echo "=========================================="
echo "Stage: ${STAGE}"
echo "Dry run: ${DRY_RUN}"
[ "$LIMIT" -gt 0 ] && echo "Limit: ${LIMIT} jobs"
echo ""

# Stage 1: clear_day training
if [ "$STAGE" = "1" ] || [ "$STAGE" = "both" ]; then
    echo "========== STAGE 1 (clear_day) =========="
    
    # Baseline
    for STRATEGY in "${BASELINE_STRATEGIES[@]}"; do
        for MODEL in "${MODELS[@]}"; do
            submit_job "$STRATEGY" "$MODEL" "clear_day" ""
        done
    done
    
    # Standard augmentation
    for STRATEGY in "${STD_STRATEGIES[@]}"; do
        for MODEL in "${MODELS[@]}"; do
            submit_job "$STRATEGY" "$MODEL" "clear_day" ""
        done
    done
    
    # Generative strategies
    for STRATEGY in "${GEN_STRATEGIES[@]}"; do
        for MODEL in "${MODELS[@]}"; do
            submit_job "$STRATEGY" "$MODEL" "clear_day" "0.5"
        done
    done
fi

# Stage 2: all domains training (no domain filter)
if [ "$STAGE" = "2" ] || [ "$STAGE" = "both" ]; then
    echo "========== STAGE 2 (all domains) =========="
    
    # Baseline
    for STRATEGY in "${BASELINE_STRATEGIES[@]}"; do
        for MODEL in "${MODELS[@]}"; do
            submit_job "$STRATEGY" "$MODEL" "" ""
        done
    done
    
    # Standard augmentation
    for STRATEGY in "${STD_STRATEGIES[@]}"; do
        for MODEL in "${MODELS[@]}"; do
            submit_job "$STRATEGY" "$MODEL" "" ""
        done
    done
    
    # Generative strategies
    for STRATEGY in "${GEN_STRATEGIES[@]}"; do
        for MODEL in "${MODELS[@]}"; do
            submit_job "$STRATEGY" "$MODEL" "" "0.5"
        done
    done
fi

echo "=========================================="
echo "Total jobs submitted: ${SUBMITTED}"
echo "=========================================="
