#!/bin/bash
# Submit training jobs for the 8 missing Stage 1 configurations
# Fixed issues:
# - Dataset names use proper case (IDD-AW, MapillaryVistas, OUTSIDE15k)
# - No invalid --stage argument
# - Using --domain-filter clear_day correctly
# Generated: 2026-01-16

PROJECT_ROOT="/home/mima2416/repositories/PROVE"
WEIGHTS_ROOT="/scratch/aaa_exchange/AWARE/WEIGHTS"
LOG_DIR="${PROJECT_ROOT}/logs"

submitted=0

echo "=== Submitting Missing Stage 1 Training Jobs ==="
echo ""

# Function to submit training+test job
submit_job() {
    local strategy=$1
    local dataset=$2
    local model=$3
    local job_name=$4
    local mem=$5
    local gmem=$6
    local time=$7
    local ratio_arg=$8
    
    # Determine output path based on dataset
    local dataset_lower=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')
    local model_suffix="${model}"
    if [ -n "$ratio_arg" ]; then
        model_suffix="${model}_ratio0p50"
    fi
    local weights_path="${WEIGHTS_ROOT}/${strategy}/${dataset_lower}_cd/${model_suffix}"
    local results_path="${PROJECT_ROOT}/results/${strategy}/${dataset_lower}_cd/${model_suffix}"
    
    # Build the training command
    local train_cmd="python unified_training.py --dataset ${dataset} --model ${model} --strategy ${strategy} --domain-filter clear_day --no-early-stop --max-iters 80000"
    
    # Build the test command
    local test_cmd="python fine_grained_test.py --config \$(ls ${weights_path}/configs/*.py | head -1) --checkpoint ${weights_path}/iter_80000.pth --dataset ${dataset} --output-dir ${results_path}"
    
    # Combined command
    local full_cmd="source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && ${train_cmd} && echo 'Training complete, starting test...' && ${test_cmd}"
    
    echo "Submitting: ${job_name} (${strategy} / ${dataset} / ${model})..."
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

submit_job "gen_cyclediffusion" "OUTSIDE15k" "pspnet_r50" "rt6_gen_cyclediffusion_outside15k_psp" "16000" "16G" "18:00" ""


echo ""
echo "=== Summary ==="
echo "Submitted ${submitted} jobs"
echo ""
echo "Monitor with: bjobs | grep rt6_"
echo ""
echo "To move to head of queue:"
echo "  bjobs | grep rt6_ | awk '{print \$1}' | xargs -I{} btop {}"
