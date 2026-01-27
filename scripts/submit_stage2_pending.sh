#!/bin/bash
# Submit Stage 2 training jobs for pending strategies: gen_cyclediffusion, std_cutmix, std_mixup

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROVE_DIR="${SCRIPT_DIR}/.."
LOG_DIR="${PROVE_DIR}/logs"

DATASETS=("BDD10k" "IDD-AW" "MapillaryVistas" "OUTSIDE15k")
MODELS=("deeplabv3plus_r50" "pspnet_r50" "segformer_mit-b5")

# Parse arguments
DRY_RUN=false
LIMIT=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --limit) LIMIT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"

count=0
submit_job() {
    local strategy=$1
    local dataset=$2
    local model=$3
    local ratio=$4
    
    # Check limit
    if [ -n "$LIMIT" ] && [ $count -ge $LIMIT ]; then
        return 1
    fi
    
    local job_name="tr_${strategy:0:10}_${dataset:0:4}_${model:0:3}"
    local cmd="source ~/.bashrc && mamba activate prove && cd ${PROVE_DIR} && python unified_training.py --dataset ${dataset} --model ${model} --strategy ${strategy}"
    
    # Add ratio for generative strategies
    if [[ "$strategy" == gen_* ]]; then
        cmd="${cmd} --real-gen-ratio ${ratio}"
    fi
    
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would submit: ${strategy}/${dataset}/${model}"
    else
        bsub -J "${job_name}" \
             -q "BatchGPU" \
             -gpu "num=1:mode=shared:gmem=24G" \
             -n 10 \
             -W "24:00" \
             -o "${LOG_DIR}/${job_name}_%J.out" \
             -e "${LOG_DIR}/${job_name}_%J.err" \
             "${cmd}"
        echo "Submitted: ${strategy}/${dataset}/${model}"
    fi
    
    ((count++))
    [ "$DRY_RUN" = false ] && sleep 0.3
    return 0
}

echo "========================================"
echo "Submit Stage 2 Training Jobs"
echo "========================================"
echo "Strategies: gen_cyclediffusion, std_cutmix, std_mixup"
echo "Dry run: $DRY_RUN"
[ -n "$LIMIT" ] && echo "Limit: $LIMIT jobs"
echo ""

# gen_cyclediffusion (generative - needs ratio)
echo "=== gen_cyclediffusion ==="
for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        submit_job "gen_cyclediffusion" "$dataset" "$model" "0.5" || break 2
    done
done

# std_cutmix (standard - no ratio)
echo ""
echo "=== std_cutmix ==="
for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        submit_job "std_cutmix" "$dataset" "$model" "" || break 2
    done
done

# std_mixup (standard - no ratio)
echo ""
echo "=== std_mixup ==="
for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        submit_job "std_mixup" "$dataset" "$model" "" || break 2
    done
done

echo ""
echo "========================================"
echo "Summary: Submitted $count jobs"
echo "========================================"
