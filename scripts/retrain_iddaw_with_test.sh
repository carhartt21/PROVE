#!/bin/bash
# Retrain IDD-AW models with CORRECT domain-filter clear_day AND auto-run tests
# This script runs both training AND testing in a single job
# Generated: 2026-01-15

PROJECT_ROOT="/home/mima2416/repositories/PROVE"
WEIGHTS_ROOT="/scratch/aaa_exchange/AWARE/WEIGHTS"
LOG_DIR="${PROJECT_ROOT}/logs"

submitted=0

echo "=== Retraining IDD-AW Models (Stage 1 - Clear Day) + Auto Test ==="
echo ""
echo "Each job will:"
echo "  1. Train model with --domain-filter clear_day"
echo "  2. Automatically run fine_grained_test.py on completion"
echo ""

# Helper to check if corrupted backup exists
check_corrupted() {
    local path=$1
    if [ -d "${path}_corrupted_labels_backup" ]; then
        return 0
    fi
    return 1
}

# Function to submit a training+test job
submit_train_test() {
    local strategy=$1
    local model=$2
    local job_name=$3
    local mem=$4
    local gmem=$5
    local time=$6
    local ratio_arg=$7  # Optional: "--real-gen-ratio 0.5" for gen_* strategies
    
    # Determine output path based on strategy and model
    local model_suffix="${model}"
    if [ -n "$ratio_arg" ]; then
        model_suffix="${model}_ratio0p50"
    fi
    local weights_path="${WEIGHTS_ROOT}/${strategy}/idd-aw_cd/${model_suffix}"
    local results_path="${PROJECT_ROOT}/results/${strategy}/idd-aw_cd/${model_suffix}"
    
    # Build the training command
    local train_cmd="python unified_training.py --dataset IDD-AW --model ${model} --strategy ${strategy} --domain-filter clear_day ${ratio_arg} --no-early-stop --max-iters 80000"
    
    # Build the test command (uses configs saved by training)
    local config_path="${weights_path}/configs/*.py"
    local checkpoint_path="${weights_path}/iter_80000.pth"
    local test_cmd="python fine_grained_test.py --config \$(ls ${config_path} | head -1) --checkpoint ${checkpoint_path} --dataset IDD-AW --output-dir ${results_path}"
    
    # Combined command: train then test
    local full_cmd="source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && ${train_cmd} && echo 'Training complete, starting test...' && ${test_cmd}"
    
    echo "Submitting: ${job_name}..."
    bsub -J "${job_name}" \
        -q BatchGPU \
        -R "rusage[mem=${mem}]" \
        -gpu "num=1:gmem=${gmem}" \
        -W ${time} \
        -o "${LOG_DIR}/${job_name}_%J.out" \
        -e "${LOG_DIR}/${job_name}_%J.err" \
        "${full_cmd}"
    ((submitted++))
}

# ===============================
# BASELINE MODELS
# ===============================
echo "=== Baseline Models ==="

if check_corrupted "${WEIGHTS_ROOT}/baseline/idd-aw_cd/deeplabv3plus_r50"; then
    submit_train_test "baseline" "deeplabv3plus_r50" "rt3_baseline_iddaw_dlv3" "16000" "16G" "18:00" ""
fi

if check_corrupted "${WEIGHTS_ROOT}/baseline/idd-aw_cd/pspnet_r50"; then
    submit_train_test "baseline" "pspnet_r50" "rt3_baseline_iddaw_psp" "16000" "16G" "18:00" ""
fi

if check_corrupted "${WEIGHTS_ROOT}/baseline/idd-aw_cd/segformer_mit-b5"; then
    submit_train_test "baseline" "segformer_mit-b5" "rt3_baseline_iddaw_segf" "20000" "24G" "26:00" ""
fi

# ===============================
# GEN_* STRATEGIES (with ratio 0.5)
# ===============================

echo ""
echo "=== gen_Attribute_Hallucination ==="
if check_corrupted "${WEIGHTS_ROOT}/gen_Attribute_Hallucination/idd-aw_cd/deeplabv3plus_r50_ratio0p50"; then
    submit_train_test "gen_Attribute_Hallucination" "deeplabv3plus_r50" "rt3_AttrHall_iddaw_dlv3" "16000" "16G" "18:00" "--real-gen-ratio 0.5"
fi
if check_corrupted "${WEIGHTS_ROOT}/gen_Attribute_Hallucination/idd-aw_cd/pspnet_r50_ratio0p50"; then
    submit_train_test "gen_Attribute_Hallucination" "pspnet_r50" "rt3_AttrHall_iddaw_psp" "16000" "16G" "18:00" "--real-gen-ratio 0.5"
fi
if check_corrupted "${WEIGHTS_ROOT}/gen_Attribute_Hallucination/idd-aw_cd/segformer_mit-b5_ratio0p50"; then
    submit_train_test "gen_Attribute_Hallucination" "segformer_mit-b5" "rt3_AttrHall_iddaw_segf" "20000" "24G" "26:00" "--real-gen-ratio 0.5"
fi

echo ""
echo "=== gen_CNetSeg ==="
if check_corrupted "${WEIGHTS_ROOT}/gen_CNetSeg/idd-aw_cd/deeplabv3plus_r50_ratio0p50"; then
    submit_train_test "gen_CNetSeg" "deeplabv3plus_r50" "rt3_CNetSeg_iddaw_dlv3" "16000" "16G" "18:00" "--real-gen-ratio 0.5"
fi
if check_corrupted "${WEIGHTS_ROOT}/gen_CNetSeg/idd-aw_cd/pspnet_r50_ratio0p50"; then
    submit_train_test "gen_CNetSeg" "pspnet_r50" "rt3_CNetSeg_iddaw_psp" "16000" "16G" "18:00" "--real-gen-ratio 0.5"
fi
if check_corrupted "${WEIGHTS_ROOT}/gen_CNetSeg/idd-aw_cd/segformer_mit-b5_ratio0p50"; then
    submit_train_test "gen_CNetSeg" "segformer_mit-b5" "rt3_CNetSeg_iddaw_segf" "20000" "24G" "26:00" "--real-gen-ratio 0.5"
fi

echo ""
echo "=== gen_CUT ==="
if check_corrupted "${WEIGHTS_ROOT}/gen_CUT/idd-aw_cd/deeplabv3plus_r50_ratio0p50"; then
    submit_train_test "gen_CUT" "deeplabv3plus_r50" "rt3_CUT_iddaw_dlv3" "16000" "16G" "18:00" "--real-gen-ratio 0.5"
fi
if check_corrupted "${WEIGHTS_ROOT}/gen_CUT/idd-aw_cd/pspnet_r50_ratio0p50"; then
    submit_train_test "gen_CUT" "pspnet_r50" "rt3_CUT_iddaw_psp" "16000" "16G" "18:00" "--real-gen-ratio 0.5"
fi
if check_corrupted "${WEIGHTS_ROOT}/gen_CUT/idd-aw_cd/segformer_mit-b5_ratio0p50"; then
    submit_train_test "gen_CUT" "segformer_mit-b5" "rt3_CUT_iddaw_segf" "20000" "24G" "26:00" "--real-gen-ratio 0.5"
fi

echo ""
echo "=== gen_IP2P ==="
if check_corrupted "${WEIGHTS_ROOT}/gen_IP2P/idd-aw_cd/deeplabv3plus_r50_ratio0p50"; then
    submit_train_test "gen_IP2P" "deeplabv3plus_r50" "rt3_IP2P_iddaw_dlv3" "16000" "16G" "18:00" "--real-gen-ratio 0.5"
fi
if check_corrupted "${WEIGHTS_ROOT}/gen_IP2P/idd-aw_cd/pspnet_r50_ratio0p50"; then
    submit_train_test "gen_IP2P" "pspnet_r50" "rt3_IP2P_iddaw_psp" "16000" "16G" "18:00" "--real-gen-ratio 0.5"
fi
if check_corrupted "${WEIGHTS_ROOT}/gen_IP2P/idd-aw_cd/segformer_mit-b5_ratio0p50"; then
    submit_train_test "gen_IP2P" "segformer_mit-b5" "rt3_IP2P_iddaw_segf" "20000" "24G" "26:00" "--real-gen-ratio 0.5"
fi

# ===============================
# STD_* STRATEGIES (no ratio)
# ===============================

echo ""
echo "=== photometric_distort ==="
if check_corrupted "${WEIGHTS_ROOT}/photometric_distort/idd-aw_cd/deeplabv3plus_r50"; then
    submit_train_test "photometric_distort" "deeplabv3plus_r50" "rt3_photom_iddaw_dlv3" "16000" "16G" "18:00" ""
fi
if check_corrupted "${WEIGHTS_ROOT}/photometric_distort/idd-aw_cd/pspnet_r50"; then
    submit_train_test "photometric_distort" "pspnet_r50" "rt3_photom_iddaw_psp" "16000" "16G" "18:00" ""
fi
if check_corrupted "${WEIGHTS_ROOT}/photometric_distort/idd-aw_cd/segformer_mit-b5"; then
    submit_train_test "photometric_distort" "segformer_mit-b5" "rt3_photom_iddaw_segf" "20000" "24G" "26:00" ""
fi

echo ""
echo "=== std_autoaugment ==="
if check_corrupted "${WEIGHTS_ROOT}/std_autoaugment/idd-aw_cd/deeplabv3plus_r50"; then
    submit_train_test "std_autoaugment" "deeplabv3plus_r50" "rt3_autoaug_iddaw_dlv3" "16000" "16G" "18:00" ""
fi
if check_corrupted "${WEIGHTS_ROOT}/std_autoaugment/idd-aw_cd/pspnet_r50"; then
    submit_train_test "std_autoaugment" "pspnet_r50" "rt3_autoaug_iddaw_psp" "16000" "16G" "18:00" ""
fi
if check_corrupted "${WEIGHTS_ROOT}/std_autoaugment/idd-aw_cd/segformer_mit-b5"; then
    submit_train_test "std_autoaugment" "segformer_mit-b5" "rt3_autoaug_iddaw_segf" "20000" "24G" "26:00" ""
fi

echo ""
echo "=== std_cutmix ==="
if check_corrupted "${WEIGHTS_ROOT}/std_cutmix/idd-aw_cd/deeplabv3plus_r50"; then
    submit_train_test "std_cutmix" "deeplabv3plus_r50" "rt3_cutmix_iddaw_dlv3" "16000" "16G" "18:00" ""
fi
if check_corrupted "${WEIGHTS_ROOT}/std_cutmix/idd-aw_cd/pspnet_r50"; then
    submit_train_test "std_cutmix" "pspnet_r50" "rt3_cutmix_iddaw_psp" "16000" "16G" "18:00" ""
fi
if check_corrupted "${WEIGHTS_ROOT}/std_cutmix/idd-aw_cd/segformer_mit-b5"; then
    submit_train_test "std_cutmix" "segformer_mit-b5" "rt3_cutmix_iddaw_segf" "20000" "24G" "26:00" ""
fi

echo ""
echo "=== std_mixup ==="
if check_corrupted "${WEIGHTS_ROOT}/std_mixup/idd-aw_cd/deeplabv3plus_r50"; then
    submit_train_test "std_mixup" "deeplabv3plus_r50" "rt3_mixup_iddaw_dlv3" "16000" "16G" "18:00" ""
fi
if check_corrupted "${WEIGHTS_ROOT}/std_mixup/idd-aw_cd/pspnet_r50"; then
    submit_train_test "std_mixup" "pspnet_r50" "rt3_mixup_iddaw_psp" "16000" "16G" "18:00" ""
fi
if check_corrupted "${WEIGHTS_ROOT}/std_mixup/idd-aw_cd/segformer_mit-b5"; then
    submit_train_test "std_mixup" "segformer_mit-b5" "rt3_mixup_iddaw_segf" "20000" "24G" "26:00" ""
fi

echo ""
echo "=== std_randaugment ==="
if check_corrupted "${WEIGHTS_ROOT}/std_randaugment/idd-aw_cd/deeplabv3plus_r50"; then
    submit_train_test "std_randaugment" "deeplabv3plus_r50" "rt3_randaug_iddaw_dlv3" "16000" "16G" "18:00" ""
fi
if check_corrupted "${WEIGHTS_ROOT}/std_randaugment/idd-aw_cd/pspnet_r50"; then
    submit_train_test "std_randaugment" "pspnet_r50" "rt3_randaug_iddaw_psp" "16000" "16G" "18:00" ""
fi
if check_corrupted "${WEIGHTS_ROOT}/std_randaugment/idd-aw_cd/segformer_mit-b5"; then
    submit_train_test "std_randaugment" "segformer_mit-b5" "rt3_randaug_iddaw_segf" "20000" "24G" "26:00" ""
fi

echo ""
echo "=== Summary ==="
echo "Submitted: $submitted training+test jobs"
echo ""
echo "Each job will:"
echo "  1. Train for 80k iterations (~2-4 hours)"
echo "  2. Automatically run fine_grained_test.py (~10-15 min)"
echo ""
echo "Monitor with: bjobs -w | grep rt3_"
echo "Check logs:   tail -f logs/rt3_*_*.out"
