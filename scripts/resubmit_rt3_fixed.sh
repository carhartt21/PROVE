#!/bin/bash
# Resubmit failed rt3_ IDD-AW retrain jobs with:
# - -n 10 for multiple CPUs
# - Using unified_training.py directly (no train_script.py path issues)
# - Training + Testing in single job
# Generated: 2026-01-15 (Fixed version)

PROJECT_ROOT="/home/mima2416/repositories/PROVE"
WEIGHTS_ROOT="/scratch/aaa_exchange/AWARE/WEIGHTS"
LOG_DIR="${PROJECT_ROOT}/logs"

submitted=0

echo "=== Resubmitting Failed rt3_ Jobs ==="
echo "Using -n 10 for multiple CPUs"
echo ""

# Function to submit a training+test job
submit_job() {
    local strategy=$1
    local model=$2
    local job_name=$3
    local mem=$4
    local gmem=$5
    local time=$6
    local ratio_arg=$7
    
    # Determine output path based on strategy and model
    local model_suffix="${model}"
    if [ -n "$ratio_arg" ]; then
        model_suffix="${model}_ratio0p50"
    fi
    local weights_path="${WEIGHTS_ROOT}/${strategy}/idd-aw_cd/${model_suffix}"
    local results_path="${PROJECT_ROOT}/results/${strategy}/idd-aw_cd/${model_suffix}"
    
    # Build the training command
    local train_cmd="python unified_training.py --dataset IDD-AW --model ${model} --strategy ${strategy} --domain-filter clear_day ${ratio_arg} --no-early-stop --max-iters 80000"
    
    # Build the test command
    local test_cmd="python fine_grained_test.py --config \$(ls ${weights_path}/configs/*.py | head -1) --checkpoint ${weights_path}/iter_80000.pth --dataset IDD-AW --output-dir ${results_path}"
    
    # Combined command: train then test
    local full_cmd="source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && ${train_cmd} && echo 'Training complete, starting test...' && ${test_cmd}"
    
    echo "Submitting: ${job_name} (${strategy} / ${model})..."
    bsub -J "${job_name}" \
        -q BatchGPU \
        -n 10 \
        -R "rusage[mem=${mem}]" \
        -gpu "num=1:gmem=${gmem}" \
        -W ${time} \
        -o "${LOG_DIR}/${job_name}_%J.out" \
        -e "${LOG_DIR}/${job_name}_%J.err" \
        "${full_cmd}"
    ((submitted++))
}

# ===============================
# BASELINE MODELS (3 jobs)
# ===============================
echo ""
echo "=== Baseline Models ==="
submit_job "baseline" "deeplabv3plus_r50" "rt4_baseline_iddaw_dlv3" "16000" "16G" "18:00" ""
submit_job "baseline" "pspnet_r50" "rt4_baseline_iddaw_psp" "16000" "16G" "18:00" ""
submit_job "baseline" "segformer_mit-b5" "rt4_baseline_iddaw_segf" "20000" "24G" "26:00" ""

# ===============================
# GEN_* STRATEGIES (with ratio 0.5) - 12 jobs
# ===============================
echo ""
echo "=== gen_Attribute_Hallucination ==="
submit_job "gen_Attribute_Hallucination" "deeplabv3plus_r50" "rt4_AttrHall_iddaw_dlv3" "16000" "16G" "18:00" "--real-gen-ratio 0.5"
submit_job "gen_Attribute_Hallucination" "pspnet_r50" "rt4_AttrHall_iddaw_psp" "16000" "16G" "18:00" "--real-gen-ratio 0.5"
submit_job "gen_Attribute_Hallucination" "segformer_mit-b5" "rt4_AttrHall_iddaw_segf" "20000" "24G" "26:00" "--real-gen-ratio 0.5"

echo ""
echo "=== gen_ControlNet_seg2image ==="
submit_job "gen_ControlNet_seg2image" "deeplabv3plus_r50" "rt4_CNetSeg_iddaw_dlv3" "16000" "16G" "18:00" "--real-gen-ratio 0.5"
submit_job "gen_ControlNet_seg2image" "pspnet_r50" "rt4_CNetSeg_iddaw_psp" "16000" "16G" "18:00" "--real-gen-ratio 0.5"
submit_job "gen_ControlNet_seg2image" "segformer_mit-b5" "rt4_CNetSeg_iddaw_segf" "20000" "24G" "26:00" "--real-gen-ratio 0.5"

echo ""
echo "=== gen_CUT ==="
submit_job "gen_CUT" "deeplabv3plus_r50" "rt4_CUT_iddaw_dlv3" "16000" "16G" "18:00" "--real-gen-ratio 0.5"
submit_job "gen_CUT" "pspnet_r50" "rt4_CUT_iddaw_psp" "16000" "16G" "18:00" "--real-gen-ratio 0.5"
submit_job "gen_CUT" "segformer_mit-b5" "rt4_CUT_iddaw_segf" "20000" "24G" "26:00" "--real-gen-ratio 0.5"

echo ""
echo "=== gen_InstructPix2Pix ==="
# Note: rt3_IP2P_iddaw_dlv3 (job 9561538) is still running - skip it
# submit_job "gen_InstructPix2Pix" "deeplabv3plus_r50" "rt4_IP2P_iddaw_dlv3" "16000" "16G" "18:00" "--real-gen-ratio 0.5"
submit_job "gen_InstructPix2Pix" "pspnet_r50" "rt4_IP2P_iddaw_psp" "16000" "16G" "18:00" "--real-gen-ratio 0.5"
submit_job "gen_InstructPix2Pix" "segformer_mit-b5" "rt4_IP2P_iddaw_segf" "20000" "24G" "26:00" "--real-gen-ratio 0.5"

# ===============================
# STD_* STRATEGIES (no ratio) - 14 jobs
# ===============================
echo ""
echo "=== std_autoaugment ==="
submit_job "std_autoaugment" "deeplabv3plus_r50" "rt4_autoaug_iddaw_dlv3" "16000" "16G" "18:00" ""
submit_job "std_autoaugment" "pspnet_r50" "rt4_autoaug_iddaw_psp" "16000" "16G" "18:00" ""
submit_job "std_autoaugment" "segformer_mit-b5" "rt4_autoaug_iddaw_segf" "20000" "24G" "26:00" ""

echo ""
echo "=== std_cutmix ==="
submit_job "std_cutmix" "deeplabv3plus_r50" "rt4_cutmix_iddaw_dlv3" "16000" "16G" "18:00" ""
submit_job "std_cutmix" "pspnet_r50" "rt4_cutmix_iddaw_psp" "16000" "16G" "18:00" ""
submit_job "std_cutmix" "segformer_mit-b5" "rt4_cutmix_iddaw_segf" "20000" "24G" "26:00" ""

echo ""
echo "=== std_mixup ==="
submit_job "std_mixup" "deeplabv3plus_r50" "rt4_mixup_iddaw_dlv3" "16000" "16G" "18:00" ""
submit_job "std_mixup" "pspnet_r50" "rt4_mixup_iddaw_psp" "16000" "16G" "18:00" ""
submit_job "std_mixup" "segformer_mit-b5" "rt4_mixup_iddaw_segf" "20000" "24G" "26:00" ""

echo ""
echo "=== std_photometric_distort ==="
submit_job "std_photometric_distort" "deeplabv3plus_r50" "rt4_photom_iddaw_dlv3" "16000" "16G" "18:00" ""
submit_job "std_photometric_distort" "pspnet_r50" "rt4_photom_iddaw_psp" "16000" "16G" "18:00" ""
submit_job "std_photometric_distort" "segformer_mit-b5" "rt4_photom_iddaw_segf" "20000" "24G" "26:00" ""

echo ""
echo "=== std_randaugment ==="
submit_job "std_randaugment" "deeplabv3plus_r50" "rt4_randaug_iddaw_dlv3" "16000" "16G" "18:00" ""
submit_job "std_randaugment" "pspnet_r50" "rt4_randaug_iddaw_psp" "16000" "16G" "18:00" ""
# Note: segformer already succeeded for randaugment (rt3_randaug_iddaw_segf)

echo ""
echo "=== Summary ==="
echo "Submitted ${submitted} jobs"
echo ""
echo "Monitor with: bjobs | grep rt4_"
