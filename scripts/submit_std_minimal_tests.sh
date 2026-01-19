#!/bin/bash
# Submit test jobs for std_minimal strategy models that need testing
# Created: 2026-01-15

WEIGHTS_ROOT="/scratch/aaa_exchange/AWARE/WEIGHTS/std_minimal"
PROVE_DIR="/home/mima2416/repositories/PROVE"
LOG_DIR="${PROVE_DIR}/logs"

# Create log directory
mkdir -p "$LOG_DIR"

# Models needing tests (from analysis)
declare -a MODELS_TO_TEST=(
    "bdd10k_ad/segformer_mit-b5"
    "idd-aw_ad/segformer_mit-b5"
    "mapillaryvistas_ad/deeplabv3plus_r50"
    "mapillaryvistas_ad/pspnet_r50"
    "mapillaryvistas_ad/segformer_mit-b5"
    "outside15k_ad/deeplabv3plus_r50"
    "outside15k_ad/pspnet_r50"
    "outside15k_ad/segformer_mit-b5"
)

echo "Submitting std_minimal test jobs..."
echo "=========================================="

for model_path in "${MODELS_TO_TEST[@]}"; do
    model_dir="${WEIGHTS_ROOT}/${model_path}"
    checkpoint="${model_dir}/iter_80000.pth"
    
    if [ ! -f "$checkpoint" ]; then
        echo "⚠️  SKIP: No checkpoint at $checkpoint"
        continue
    fi
    
    # Find config file (in configs/ subdirectory)
    config_file=$(ls "${model_dir}/configs/"*.py 2>/dev/null | head -1)
    if [ -z "$config_file" ]; then
        # Fallback to training_config.py in root
        config_file="${model_dir}/training_config.py"
    fi
    
    if [ ! -f "$config_file" ]; then
        echo "⚠️  SKIP: No config file found for $model_path"
        continue
    fi
    
    # Parse dataset and model name
    dataset_stage=$(dirname "$model_path")  # e.g., bdd10k_ad
    model_name=$(basename "$model_path")    # e.g., segformer_mit-b5
    
    # Extract base dataset name (without _cd or _ad)
    dataset_base=$(echo "$dataset_stage" | sed 's/_[ca]d$//')
    
    # Map to proper dataset name for fine_grained_test.py
    case "$dataset_base" in
        "bdd10k") dataset="BDD10k" ;;
        "idd-aw") dataset="IDD-AW" ;;
        "mapillaryvistas") dataset="MapillaryVistas" ;;
        "outside15k") dataset="OUTSIDE15k" ;;
        *) dataset="$dataset_base" ;;
    esac
    
    # Output directory for results
    output_dir="results/std_minimal/${dataset_stage}/${model_name}"
    
    job_name="test_std_minimal_${dataset_stage}_${model_name}"
    log_file="${LOG_DIR}/${job_name}.log"
    err_file="${LOG_DIR}/${job_name}.err"
    
    echo "Submitting: $job_name"
    echo "  Config: $config_file"
    echo "  Checkpoint: $checkpoint"
    echo "  Dataset: $dataset"
    echo "  Output: $output_dir"
    
    bsub -J "$job_name" \
         -q BatchGPU \
         -gpu "num=1:mode=shared:gmem=18G" \
         -n 10 \
         -o "$log_file" \
         -e "$err_file" \
         -W 0:30 \
         "source ~/.bashrc && conda activate prove && cd ${PROVE_DIR} && python fine_grained_test.py --config ${config_file} --checkpoint ${checkpoint} --dataset ${dataset} --output-dir ${output_dir}"
    
    echo ""
done

echo "=========================================="
echo "Submitted all test jobs!"
