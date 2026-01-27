#!/bin/bash
# Retraining jobs for 6 mismatched OUTSIDE15k Stage 1 models
# These were incorrectly trained with 19 Cityscapes classes, need retraining with 24 native classes
#
# Issue: gen_step1x_new, gen_step1x_v1p2, gen_Qwen_Image_Edit, gen_stargan_v2
#        PSPNet and SegFormer models on OUTSIDE15k
#        Were trained without --use-native-classes flag
#
# After training completes, tests will be submitted automatically via job dependency

cd /home/mima2416/repositories/PROVE

# Configuration array: strategy, model
declare -a CONFIGS=(
    "gen_Qwen_Image_Edit:pspnet_r50"
    "gen_stargan_v2:segformer_mit-b5"
    "gen_step1x_new:pspnet_r50"
    "gen_step1x_new:segformer_mit-b5"
    "gen_step1x_v1p2:pspnet_r50"
    "gen_step1x_v1p2:segformer_mit-b5"
)

RATIO="0.50"
WEIGHTS_DIR="/scratch/aaa_exchange/AWARE/WEIGHTS"

echo "=== Submitting 6 retraining jobs for mismatched OUTSIDE15k models ==="
echo ""

for config in "${CONFIGS[@]}"; do
    IFS=':' read -r STRATEGY MODEL <<< "$config"
    
    # Create safe job name (short and no special chars)
    STRAT_SHORT=$(echo $STRATEGY | sed 's/gen_//' | cut -c1-8)
    MODEL_SHORT=$(echo $MODEL | cut -c1-4)
    JOB_NAME="fix_${STRAT_SHORT}_${MODEL_SHORT}_out15k"
    LOG_FILE="logs/retrain_${STRATEGY}_${MODEL}_outside15k.log"
    
    # Compute output directory
    RATIO_DIR="${MODEL}_ratio0p50"
    OUTPUT_DIR="${WEIGHTS_DIR}/${STRATEGY}/outside15k/${RATIO_DIR}"
    
    echo "Submitting: $STRATEGY / $MODEL"
    echo "  Output: $OUTPUT_DIR"
    
    # Submit training job
    TRAIN_JOB=$(bsub -J "$JOB_NAME" \
        -o "$LOG_FILE" \
        -q gpu \
        -R "select[ngpus>0] rusage[ngpus_physical=1,mem=32000]" \
        -gpu "num=1:mode=exclusive_process" \
        -W 24:00 \
        "source ~/.bashrc && conda activate mmsegmentation && \
         python unified_training.py \
           --dataset OUTSIDE15k \
           --model $MODEL \
           --strategy $STRATEGY \
           --real-gen-ratio $RATIO \
           --domain-filter clear_day \
           --use-native-classes" 2>&1 | grep -oP '\d+')
    
    echo "  Training job ID: $TRAIN_JOB"
    
    # Submit test job with dependency on training job
    TEST_JOB_NAME="test_${STRAT_SHORT}_${MODEL_SHORT}_out15k"
    TEST_LOG="logs/test_${STRATEGY}_${MODEL}_outside15k.log"
    
    TEST_JOB=$(bsub -J "$TEST_JOB_NAME" \
        -w "done($TRAIN_JOB)" \
        -o "$TEST_LOG" \
        -q gpu \
        -R "select[ngpus>0] rusage[ngpus_physical=1,mem=16000]" \
        -gpu "num=1:mode=exclusive_process" \
        -W 1:00 \
        "source ~/.bashrc && conda activate mmsegmentation && \
         python fine_grained_test.py \
           --config ${OUTPUT_DIR}/training_config.py \
           --checkpoint ${OUTPUT_DIR}/iter_80000.pth \
           --dataset OUTSIDE15k \
           --output-dir ${OUTPUT_DIR}/test_results_detailed" 2>&1 | grep -oP '\d+')
    
    echo "  Test job ID: $TEST_JOB (depends on $TRAIN_JOB)"
    echo ""
done

echo "=== Summary ==="
echo "Submitted 6 training jobs with dependent test jobs"
echo "Monitor with: bjobs -w"
