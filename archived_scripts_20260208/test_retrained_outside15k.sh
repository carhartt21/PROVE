#!/bin/bash
# Test jobs for retrained OUTSIDE15k models (to be run after training completes)
# Run this script after all 6 training jobs finish

cd ${HOME}/repositories/PROVE

WEIGHTS_DIR="${AWARE_DATA_ROOT}/WEIGHTS"

# Configuration array: strategy, model  
declare -a CONFIGS=(
    "gen_Qwen_Image_Edit:pspnet_r50"
    "gen_stargan_v2:segformer_mit-b5"
    "gen_step1x_new:pspnet_r50"
    "gen_step1x_new:segformer_mit-b5"
    "gen_step1x_v1p2:pspnet_r50"
    "gen_step1x_v1p2:segformer_mit-b5"
)

echo "=== Submitting test jobs for retrained OUTSIDE15k models ==="
echo ""

for config in "${CONFIGS[@]}"; do
    IFS=':' read -r STRATEGY MODEL <<< "$config"
    
    RATIO_DIR="${MODEL}_ratio0p50"
    MODEL_DIR="${WEIGHTS_DIR}/${STRATEGY}/outside15k/${RATIO_DIR}"
    CHECKPOINT="${MODEL_DIR}/iter_80000.pth"
    CONFIG_FILE="${MODEL_DIR}/training_config.py"
    OUTPUT_DIR="${MODEL_DIR}/test_results_detailed"
    
    # Check if checkpoint exists
    if [ ! -f "$CHECKPOINT" ]; then
        echo "SKIPPING ${STRATEGY}/${MODEL}: checkpoint not found"
        echo "  Expected: $CHECKPOINT"
        continue
    fi
    
    # Create safe job name
    STRAT_SHORT=$(echo $STRATEGY | sed 's/gen_//' | cut -c1-8)
    MODEL_SHORT=$(echo $MODEL | cut -c1-4)
    JOB_NAME="test_${STRAT_SHORT}_${MODEL_SHORT}_out15k"
    LOG_FILE="logs/test_${STRATEGY}_${MODEL}_outside15k"

    
    echo "Submitting: ${STRATEGY}/${MODEL}"
    
    bsub -J "$JOB_NAME" \
        -o "$LOG_FILE".log \
        -e "$LOG_FILE".err \
        -q BatchGPU \
        -n 8 \
        -gpu "num=1:mode=exclusive_process:gmem=16000" \
        -W 1:00 \
        "source ~/.bashrc && conda activate prove && \
         python fine_grained_test.py \
           --config ${CONFIG_FILE} \
           --checkpoint ${CHECKPOINT} \
           --dataset OUTSIDE15k \
           --output-dir ${OUTPUT_DIR}"
    
    echo ""
done

echo "=== Done ==="
