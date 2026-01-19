#!/bin/bash
# Submit test jobs for IDD-AW models that have checkpoints at iddaw_cd
# These models completed training but tests failed due to path mismatch
# (looking at idd-aw_cd instead of iddaw_cd)
# Generated: 2026-01-15

PROJECT_ROOT="/home/mima2416/repositories/PROVE"
WEIGHTS_ROOT="/scratch/aaa_exchange/AWARE/WEIGHTS"
LOG_DIR="${PROJECT_ROOT}/logs"

submitted=0

echo "=== Submitting IDD-AW Test Jobs (iddaw_cd paths) ==="
echo ""

# Function to submit test job
submit_test() {
    local strategy=$1
    local model=$2
    local job_name=$3
    local mem=$4
    local gmem=$5
    local is_gen=$6  # "gen" if generative strategy, "" otherwise
    
    # Determine paths based on strategy type
    local model_suffix="${model}"
    if [ "$is_gen" = "gen" ]; then
        model_suffix="${model}_ratio0p50"
    fi
    
    # Use iddaw_cd (old naming) where checkpoints actually exist
    local weights_path="${WEIGHTS_ROOT}/${strategy}/iddaw_cd/${model_suffix}"
    local results_path="${PROJECT_ROOT}/results/${strategy}/idd-aw_cd/${model_suffix}"
    
    # Verify checkpoint exists
    if [ ! -f "${weights_path}/iter_80000.pth" ]; then
        echo "SKIP: ${job_name} - checkpoint not found at ${weights_path}"
        return
    fi
    
    local test_cmd="python fine_grained_test.py --config \$(ls ${weights_path}/configs/*.py | head -1) --checkpoint ${weights_path}/iter_80000.pth --dataset IDD-AW --output-dir ${results_path}"
    local full_cmd="source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && ${test_cmd}"
    
    echo "Submitting: ${job_name}..."
    bsub -J "${job_name}" \
        -q BatchGPU \
        -n 4 \
        -R "rusage[mem=${mem}]" \
        -gpu "num=1:gmem=${gmem}" \
        -W 2:00 \
        -o "${LOG_DIR}/${job_name}_%J.out" \
        -e "${LOG_DIR}/${job_name}_%J.err" \
        "${full_cmd}"
    ((submitted++))
}

# ===============================
# BASELINE (3 jobs)
# ===============================
echo "=== baseline ==="
submit_test "baseline" "deeplabv3plus_r50" "test_baseline_iddaw_dlv3" "8000" "8G" ""
submit_test "baseline" "pspnet_r50" "test_baseline_iddaw_psp" "8000" "8G" ""
submit_test "baseline" "segformer_mit-b5" "test_baseline_iddaw_segf" "12000" "12G" ""

# ===============================
# GEN_* STRATEGIES (9 jobs)
# ===============================
echo ""
echo "=== gen_Attribute_Hallucination ==="
submit_test "gen_Attribute_Hallucination" "deeplabv3plus_r50" "test_AttrHall_iddaw_dlv3" "8000" "8G" "gen"
submit_test "gen_Attribute_Hallucination" "pspnet_r50" "test_AttrHall_iddaw_psp" "8000" "8G" "gen"
submit_test "gen_Attribute_Hallucination" "segformer_mit-b5" "test_AttrHall_iddaw_segf" "12000" "12G" "gen"

echo ""
echo "=== gen_CUT ==="
submit_test "gen_CUT" "deeplabv3plus_r50" "test_CUT_iddaw_dlv3" "8000" "8G" "gen"
submit_test "gen_CUT" "pspnet_r50" "test_CUT_iddaw_psp" "8000" "8G" "gen"
submit_test "gen_CUT" "segformer_mit-b5" "test_CUT_iddaw_segf" "12000" "12G" "gen"

echo ""
echo "=== gen_IP2P (DeepLabV3+ from rt3_ still running) ==="
# gen_IP2P dlv3 is handled by rt3_ job that's still running
submit_test "gen_IP2P" "deeplabv3plus_r50" "test_IP2P_iddaw_dlv3" "8000" "8G" "gen"
submit_test "gen_IP2P" "pspnet_r50" "test_IP2P_iddaw_psp" "8000" "8G" "gen"
submit_test "gen_IP2P" "segformer_mit-b5" "test_IP2P_iddaw_segf" "12000" "12G" "gen"

# ===============================
# STD_* STRATEGIES (15 jobs)
# ===============================
echo ""
echo "=== std_autoaugment ==="
submit_test "std_autoaugment" "deeplabv3plus_r50" "test_autoaug_iddaw_dlv3" "8000" "8G" ""
submit_test "std_autoaugment" "pspnet_r50" "test_autoaug_iddaw_psp" "8000" "8G" ""
submit_test "std_autoaugment" "segformer_mit-b5" "test_autoaug_iddaw_segf" "12000" "12G" ""

echo ""
echo "=== std_cutmix ==="
submit_test "std_cutmix" "deeplabv3plus_r50" "test_cutmix_iddaw_dlv3" "8000" "8G" ""
submit_test "std_cutmix" "pspnet_r50" "test_cutmix_iddaw_psp" "8000" "8G" ""
submit_test "std_cutmix" "segformer_mit-b5" "test_cutmix_iddaw_segf" "12000" "12G" ""

echo ""
echo "=== std_mixup ==="
submit_test "std_mixup" "deeplabv3plus_r50" "test_mixup_iddaw_dlv3" "8000" "8G" ""
submit_test "std_mixup" "pspnet_r50" "test_mixup_iddaw_psp" "8000" "8G" ""
submit_test "std_mixup" "segformer_mit-b5" "test_mixup_iddaw_segf" "12000" "12G" ""

echo ""
echo "=== std_randaugment ==="
submit_test "std_randaugment" "deeplabv3plus_r50" "test_randaug_iddaw_dlv3" "8000" "8G" ""
submit_test "std_randaugment" "pspnet_r50" "test_randaug_iddaw_psp" "8000" "8G" ""
submit_test "std_randaugment" "segformer_mit-b5" "test_randaug_iddaw_segf" "12000" "12G" ""

echo ""
echo "=== photometric_distort ==="
# photometric_distort checkpoints are at iddaw_cd
submit_test "photometric_distort" "deeplabv3plus_r50" "test_photom_iddaw_dlv3" "8000" "8G" ""
submit_test "photometric_distort" "pspnet_r50" "test_photom_iddaw_psp" "8000" "8G" ""
submit_test "photometric_distort" "segformer_mit-b5" "test_photom_iddaw_segf" "12000" "12G" ""

echo ""
echo "=== Summary ==="
echo "Submitted ${submitted} test jobs"
echo ""
echo "Monitor with: bjobs | grep test_"
