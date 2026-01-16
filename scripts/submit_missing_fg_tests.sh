#!/bin/bash
# Submit missing fine-grained test jobs for main model configurations
# These are configs with buggy old tests (mIoU < 5%) that need retesting

# Don't use set -e since bsub may return non-zero in some cases

WEIGHTS_ROOT="/scratch/aaa_exchange/AWARE/WEIGHTS"
LOG_DIR="/home/mima2416/repositories/PROVE/logs"
SCRIPT_DIR="/home/mima2416/repositories/PROVE"

# Dataset display name mapping
get_dataset_display() {
    case $1 in
        bdd10k) echo "BDD10k" ;;
        idd-aw|iddaw) echo "IDD-AW" ;;
        mapillaryvistas) echo "MapillaryVistas" ;;
        outside15k) echo "OUTSIDE15k" ;;
        *) echo "$1" ;;
    esac
}

# Get correct dataset directory name (handles iddaw vs idd-aw)
get_dataset_dir() {
    case $1 in
        idd-aw) echo "idd-aw_cd" ;;
        iddaw) echo "iddaw_cd" ;;
        *) echo "${1}_cd" ;;
    esac
}

# Configs needing tests (from analysis)
CONFIGS=(
    "gen_albumentations_weather|mapillaryvistas|deeplabv3plus_r50_ratio0p50"
    "gen_albumentations_weather|mapillaryvistas|pspnet_r50_ratio0p50"
    "gen_Attribute_Hallucination|idd-aw|deeplabv3plus_r50_ratio0p50"
    "gen_Attribute_Hallucination|idd-aw|pspnet_r50_ratio0p50"
    "gen_Attribute_Hallucination|mapillaryvistas|deeplabv3plus_r50_ratio0p50"
    "gen_Attribute_Hallucination|mapillaryvistas|pspnet_r50_ratio0p50"
    "gen_Attribute_Hallucination|mapillaryvistas|segformer_mit-b5_ratio0p50"
    "gen_augmenters|mapillaryvistas|deeplabv3plus_r50_ratio0p50"
    "gen_augmenters|mapillaryvistas|segformer_mit-b5_ratio0p50"
    "gen_automold|mapillaryvistas|deeplabv3plus_r50_ratio0p50"
    "gen_automold|mapillaryvistas|pspnet_r50_ratio0p50"
    "gen_CNetSeg|idd-aw|deeplabv3plus_r50_ratio0p50"
    "gen_CNetSeg|idd-aw|pspnet_r50_ratio0p50"
    "gen_CNetSeg|mapillaryvistas|pspnet_r50_ratio0p50"
    "gen_CNetSeg|mapillaryvistas|segformer_mit-b5_ratio0p50"
    "gen_CUT|idd-aw|deeplabv3plus_r50_ratio0p50"
    "gen_CUT|idd-aw|pspnet_r50_ratio0p50"
    "gen_CUT|idd-aw|segformer_mit-b5_ratio0p50"
    "gen_cyclediffusion|mapillaryvistas|deeplabv3plus_r50_ratio0p50"
    "gen_cyclediffusion|mapillaryvistas|pspnet_r50_ratio0p50"
    "gen_cyclediffusion|mapillaryvistas|segformer_mit-b5_ratio0p50"
    "gen_cycleGAN|mapillaryvistas|deeplabv3plus_r50_ratio0p50"
    "gen_cycleGAN|mapillaryvistas|pspnet_r50_ratio0p50"
    "gen_flux_kontext|iddaw|deeplabv3plus_r50_ratio0p50"
    "gen_flux_kontext|iddaw|pspnet_r50_ratio0p50"
    "gen_flux_kontext|mapillaryvistas|pspnet_r50_ratio0p50"
    "gen_flux_kontext|outside15k|deeplabv3plus_r50_ratio0p50"
    "gen_flux_kontext|outside15k|pspnet_r50_ratio0p50"
    "gen_IP2P|idd-aw|deeplabv3plus_r50_ratio0p50"
    "gen_IP2P|idd-aw|pspnet_r50_ratio0p50"
    "gen_IP2P|mapillaryvistas|pspnet_r50_ratio0p50"
)

echo "=== Submitting ${#CONFIGS[@]} Fine-Grained Test Jobs ==="
echo ""

submitted=0
skipped=0

for config in "${CONFIGS[@]}"; do
    IFS='|' read -r strategy dataset model <<< "$config"
    
    dataset_dir=$(get_dataset_dir "$dataset")
    dataset_display=$(get_dataset_display "$dataset")
    
    weights_dir="${WEIGHTS_ROOT}/${strategy}/${dataset_dir}/${model}"
    config_path="${weights_dir}/training_config.py"
    checkpoint_path="${weights_dir}/iter_80000.pth"
    output_dir="${weights_dir}/test_results_detailed"
    
    # Create short job name
    short_strategy=$(echo "$strategy" | sed 's/gen_/g/' | cut -c1-10)
    short_dataset=$(echo "$dataset" | cut -c1-4)
    short_model=$(echo "$model" | cut -c1-3)
    job_name="fg_${short_strategy}_${short_dataset}_${short_model}"
    
    if [ ! -f "$checkpoint_path" ]; then
        echo "SKIP: Missing checkpoint for $strategy/$dataset/$model"
        ((skipped++))
        continue
    fi
    
    if [ ! -f "$config_path" ]; then
        echo "SKIP: Missing config for $strategy/$dataset/$model"
        ((skipped++))
        continue
    fi
    
    echo "Submitting: $strategy/$dataset/$model"
    
    bsub -J "$job_name" \
        -q BatchGPU \
        -n 4 \
        -R "span[hosts=1]" \
        -R "rusage[mem=8000]" \
        -gpu "num=1:gmem=8G:mode=shared" \
        -W 0:30 \
        -o "${LOG_DIR}/${job_name}_%J.out" \
        -e "${LOG_DIR}/${job_name}_%J.err" \
        "source ~/.bashrc && conda activate prove && cd ${SCRIPT_DIR} && python fine_grained_test.py --config ${config_path} --checkpoint ${checkpoint_path} --dataset ${dataset_display} --output-dir ${output_dir} --batch-size 8"
    
    ((submitted++))
    
    # Small delay to avoid overwhelming the scheduler
    sleep 0.5
done

echo ""
echo "=== Summary ==="
echo "Submitted: $submitted jobs"
echo "Skipped: $skipped jobs"
