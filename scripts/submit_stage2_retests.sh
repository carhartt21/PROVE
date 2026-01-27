#!/bin/bash
# Submit retest jobs for Stage 2 configurations that have stale results
# (i.e., tested with Stage 1 checkpoints instead of Stage 2)
#
# These are all BDD10k tests that need to be re-run against Stage 2 checkpoints

PROJECT_ROOT="/home/mima2416/repositories/PROVE"
WEIGHTS_ROOT="/scratch/aaa_exchange/AWARE/WEIGHTS_STAGE_2"
LOG_DIR="${PROJECT_ROOT}/logs"

mkdir -p "${LOG_DIR}"

# Stale configurations to retest (all BDD10k)
declare -a STALE_TESTS=(
    "baseline/bdd10k/deeplabv3plus_r50"
    "baseline/bdd10k/pspnet_r50"
    "baseline/bdd10k/segformer_mit-b5"
    "gen_Attribute_Hallucination/bdd10k/deeplabv3plus_r50_ratio0p50"
    "gen_Attribute_Hallucination/bdd10k/pspnet_r50_ratio0p50"
    "gen_Attribute_Hallucination/bdd10k/segformer_mit-b5_ratio0p50"
    "gen_CUT/bdd10k/deeplabv3plus_r50_ratio0p50"
    "gen_CUT/bdd10k/pspnet_r50_ratio0p50"
    "gen_CUT/bdd10k/segformer_mit-b5_ratio0p50"
    "gen_IP2P/bdd10k/deeplabv3plus_r50_ratio0p50"
    "gen_Img2Img/bdd10k/deeplabv3plus_r50_ratio0p50"
    "gen_Img2Img/bdd10k/pspnet_r50_ratio0p50"
    "gen_LANIT/bdd10k/deeplabv3plus_r50_ratio0p50"
)

echo "Submitting Stage 2 retest jobs for stale configurations..."
echo ""

SUBMITTED=0
SKIPPED=0

for config in "${STALE_TESTS[@]}"; do
    strategy=$(echo "$config" | cut -d'/' -f1)
    dataset=$(echo "$config" | cut -d'/' -f2)
    model=$(echo "$config" | cut -d'/' -f3)
    
    checkpoint_path="${WEIGHTS_ROOT}/${strategy}/${dataset}/${model}/iter_80000.pth"
    config_path="${WEIGHTS_ROOT}/${strategy}/${dataset}/${model}/training_config.py"
    
    if [[ ! -f "${checkpoint_path}" ]]; then
        echo "SKIP: Missing checkpoint for ${strategy}/${dataset}/${model}"
        ((SKIPPED++))
        continue
    fi
    
    # Job name
    job_name="fg2_${strategy:0:10}_${dataset:0:5}_${model:0:10}"
    job_name=$(echo "$job_name" | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9_')
    
    # Output directory
    output_dir="${WEIGHTS_ROOT}/${strategy}/${dataset}/${model}/test_results_detailed"
    
    # Log file
    log_file="${LOG_DIR}/${job_name}.log"
    
    echo "Submitting: ${strategy}/${dataset}/${model}"
    
    bsub -J "${job_name}" \
         -q BatchGPU \
         -W 0:30 \
         -n 10 \
         -gpu "num=1:gmem=16G:mode=shared" \
         -o "${log_file}" \
         "source ~/.bashrc && mamba activate prove && cd ${PROJECT_ROOT} && python fine_grained_test.py --config '${config_path}' --checkpoint '${checkpoint_path}' --dataset BDD10k --output-dir '${output_dir}'"
    
    ((SUBMITTED++))
    sleep 0.5  # Avoid overwhelming the job scheduler
done

echo ""
echo "=========================================="
echo "Stage 2 Retest Submission Complete"
echo "=========================================="
echo "Submitted: ${SUBMITTED}"
echo "Skipped:   ${SKIPPED}"
echo ""
echo "Monitor jobs: bjobs -w | grep fg2_"
