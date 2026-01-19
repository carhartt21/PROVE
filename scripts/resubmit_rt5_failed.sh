#!/bin/bash
# Resubmit failed rt4_ IDD-AW jobs with corrections
# - Fixed strategy names: gen_CNetSeg, gen_IP2P, photometric_distort
# - Added --resume for incomplete training jobs
# - Extended wall time for jobs that hit the time limit
# Generated: 2026-01-15

PROJECT_ROOT="/home/mima2416/repositories/PROVE"
WEIGHTS_ROOT="/scratch/aaa_exchange/AWARE/WEIGHTS"
LOG_DIR="${PROJECT_ROOT}/logs"

submitted=0

echo "=== Resubmitting Failed rt4_ Jobs (Round 5) ==="
echo "Fixed strategy names and added resume support"
echo ""

# Function to submit a training+test job with resume support
submit_job() {
    local strategy=$1
    local model=$2
    local job_name=$3
    local mem=$4
    local gmem=$5
    local time=$6
    local ratio_arg=$7
    local needs_resume=$8
    
    # Determine output path based on strategy and model
    local model_suffix="${model}"
    if [ -n "$ratio_arg" ]; then
        model_suffix="${model}_ratio0p50"
    fi
    local weights_path="${WEIGHTS_ROOT}/${strategy}/idd-aw_cd/${model_suffix}"
    local results_path="${PROJECT_ROOT}/results/${strategy}/idd-aw_cd/${model_suffix}"
    
    # Build the training command
    local resume_arg=""
    if [ "$needs_resume" = "true" ]; then
        resume_arg="--resume"
    fi
    local train_cmd="python unified_training.py --dataset IDD-AW --model ${model} --strategy ${strategy} --domain-filter clear_day ${ratio_arg} --no-early-stop --max-iters 80000 ${resume_arg}"
    
    # Build the test command (use latest checkpoint)
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
# STRATEGY NAME FIX JOBS (8 jobs)
# These failed because of wrong strategy names
# ===============================

echo ""
echo "=== gen_CNetSeg (was gen_ControlNet_seg2image) - 3 jobs ==="
submit_job "gen_CNetSeg" "deeplabv3plus_r50" "rt5_CNetSeg_iddaw_dlv3" "16000" "16G" "18:00" "--real-gen-ratio 0.5" "false"
submit_job "gen_CNetSeg" "pspnet_r50" "rt5_CNetSeg_iddaw_psp" "16000" "16G" "18:00" "--real-gen-ratio 0.5" "false"
submit_job "gen_CNetSeg" "segformer_mit-b5" "rt5_CNetSeg_iddaw_segf" "20000" "24G" "26:00" "--real-gen-ratio 0.5" "false"

echo ""
echo "=== gen_IP2P (was gen_InstructPix2Pix) - 2 jobs ==="
submit_job "gen_IP2P" "pspnet_r50" "rt5_IP2P_iddaw_psp" "16000" "16G" "18:00" "--real-gen-ratio 0.5" "false"
submit_job "gen_IP2P" "segformer_mit-b5" "rt5_IP2P_iddaw_segf" "20000" "24G" "26:00" "--real-gen-ratio 0.5" "false"

echo ""
echo "=== photometric_distort (was std_photometric_distort) - 3 jobs ==="
submit_job "photometric_distort" "deeplabv3plus_r50" "rt5_photom_iddaw_dlv3" "16000" "16G" "18:00" "" "false"
submit_job "photometric_distort" "pspnet_r50" "rt5_photom_iddaw_psp" "16000" "16G" "18:00" "" "false"
submit_job "photometric_distort" "segformer_mit-b5" "rt5_photom_iddaw_segf" "20000" "24G" "26:00" "" "false"

# ===============================
# INCOMPLETE TRAINING JOBS (20 jobs)
# These started training but hit the wall time
# Using --resume to continue from last checkpoint
# Extended wall time to 32:00 / 42:00
# ===============================

echo ""
echo "=== BASELINE (needs resume) - 2 jobs ==="
# Note: rt4_baseline_iddaw_segf is still running
submit_job "baseline" "deeplabv3plus_r50" "rt5_baseline_iddaw_dlv3" "16000" "16G" "32:00" "" "true"
submit_job "baseline" "pspnet_r50" "rt5_baseline_iddaw_psp" "16000" "16G" "32:00" "" "true"

echo ""
echo "=== gen_Attribute_Hallucination (needs resume) - 3 jobs ==="
submit_job "gen_Attribute_Hallucination" "deeplabv3plus_r50" "rt5_AttrHall_iddaw_dlv3" "16000" "16G" "32:00" "--real-gen-ratio 0.5" "true"
submit_job "gen_Attribute_Hallucination" "pspnet_r50" "rt5_AttrHall_iddaw_psp" "16000" "16G" "32:00" "--real-gen-ratio 0.5" "true"
submit_job "gen_Attribute_Hallucination" "segformer_mit-b5" "rt5_AttrHall_iddaw_segf" "20000" "24G" "42:00" "--real-gen-ratio 0.5" "true"

echo ""
echo "=== gen_CUT (needs resume) - 3 jobs ==="
submit_job "gen_CUT" "deeplabv3plus_r50" "rt5_CUT_iddaw_dlv3" "16000" "16G" "32:00" "--real-gen-ratio 0.5" "true"
submit_job "gen_CUT" "pspnet_r50" "rt5_CUT_iddaw_psp" "16000" "16G" "32:00" "--real-gen-ratio 0.5" "true"
submit_job "gen_CUT" "segformer_mit-b5" "rt5_CUT_iddaw_segf" "20000" "24G" "42:00" "--real-gen-ratio 0.5" "true"

echo ""
echo "=== std_autoaugment (needs resume) - 3 jobs ==="
submit_job "std_autoaugment" "deeplabv3plus_r50" "rt5_autoaug_iddaw_dlv3" "16000" "16G" "32:00" "" "true"
submit_job "std_autoaugment" "pspnet_r50" "rt5_autoaug_iddaw_psp" "16000" "16G" "32:00" "" "true"
submit_job "std_autoaugment" "segformer_mit-b5" "rt5_autoaug_iddaw_segf" "20000" "24G" "42:00" "" "true"

echo ""
echo "=== std_cutmix (needs resume) - 3 jobs ==="
submit_job "std_cutmix" "deeplabv3plus_r50" "rt5_cutmix_iddaw_dlv3" "16000" "16G" "32:00" "" "true"
submit_job "std_cutmix" "pspnet_r50" "rt5_cutmix_iddaw_psp" "16000" "16G" "32:00" "" "true"
submit_job "std_cutmix" "segformer_mit-b5" "rt5_cutmix_iddaw_segf" "20000" "24G" "42:00" "" "true"

echo ""
echo "=== std_mixup (needs resume) - 3 jobs ==="
submit_job "std_mixup" "deeplabv3plus_r50" "rt5_mixup_iddaw_dlv3" "16000" "16G" "32:00" "" "true"
submit_job "std_mixup" "pspnet_r50" "rt5_mixup_iddaw_psp" "16000" "16G" "32:00" "" "true"
submit_job "std_mixup" "segformer_mit-b5" "rt5_mixup_iddaw_segf" "20000" "24G" "42:00" "" "true"

echo ""
echo "=== std_randaugment (needs resume) - 2 jobs ==="
submit_job "std_randaugment" "deeplabv3plus_r50" "rt5_randaug_iddaw_dlv3" "16000" "16G" "32:00" "" "true"
submit_job "std_randaugment" "pspnet_r50" "rt5_randaug_iddaw_psp" "16000" "16G" "32:00" "" "true"

echo ""
echo "=== Summary ==="
echo "Submitted ${submitted} jobs"
echo ""
echo "Monitor with: bjobs | grep rt5_"
echo ""
echo "To move to head of queue, run:"
echo "  bjobs | grep rt5_ | awk '{print \$1}' | xargs -I{} btop {}"
