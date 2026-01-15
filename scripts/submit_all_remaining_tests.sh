#!/bin/bash
# Submit test jobs for all buggy and missing configurations
# Based on TESTING_COVERAGE.md analysis

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

# Get model directory name
get_model_dir() {
    local model=$1
    local strategy=$2
    case $model in
        "DeepLabV3+"|"DLV3+")
            if [[ "$strategy" == std_* ]] || [[ "$strategy" == "baseline" ]] || [[ "$strategy" == "photometric_distort" ]]; then
                echo "deeplabv3plus_r50"
            else
                echo "deeplabv3plus_r50_ratio0p50"
            fi
            ;;
        "PSPNet"|"PSP")
            if [[ "$strategy" == std_* ]] || [[ "$strategy" == "baseline" ]] || [[ "$strategy" == "photometric_distort" ]]; then
                echo "pspnet_r50"
            else
                echo "pspnet_r50_ratio0p50"
            fi
            ;;
        "SegFormer"|"SF")
            if [[ "$strategy" == std_* ]] || [[ "$strategy" == "baseline" ]] || [[ "$strategy" == "photometric_distort" ]]; then
                echo "segformer_mit-b5"
            else
                echo "segformer_mit-b5_ratio0p50"
            fi
            ;;
        *) echo "$model" ;;
    esac
}

# Configurations to test - format: "strategy|dataset|model_display"
# Buggy configurations (mIoU < 5%)
BUGGY_CONFIGS=(
    "baseline|idd-aw|DeepLabV3+"
    "baseline|idd-aw|SegFormer"
    "gen_Attribute_Hallucination|idd-aw|DeepLabV3+"
    "gen_Attribute_Hallucination|idd-aw|PSPNet"
    "gen_CNetSeg|idd-aw|DeepLabV3+"
    "gen_CNetSeg|idd-aw|PSPNet"
    "gen_CUT|idd-aw|DeepLabV3+"
    "gen_CUT|idd-aw|PSPNet"
    "gen_CUT|idd-aw|SegFormer"
    "gen_IP2P|idd-aw|DeepLabV3+"
    "gen_IP2P|idd-aw|PSPNet"
    "gen_Weather_Effect_Generator|outside15k|DeepLabV3+"
    "gen_cyclediffusion|mapillaryvistas|DeepLabV3+"
    "gen_cyclediffusion|mapillaryvistas|PSPNet"
    "gen_cyclediffusion|mapillaryvistas|SegFormer"
    "gen_stargan_v2|outside15k|SegFormer"
    "gen_step1x_new|outside15k|PSPNet"
    "gen_step1x_new|outside15k|SegFormer"
    "gen_step1x_v1p2|outside15k|PSPNet"
    "gen_step1x_v1p2|outside15k|SegFormer"
    "photometric_distort|outside15k|SegFormer"
    "std_autoaugment|outside15k|PSPNet"
    "std_autoaugment|outside15k|SegFormer"
    "std_cutmix|outside15k|DeepLabV3+"
    "std_cutmix|outside15k|PSPNet"
    "std_cutmix|outside15k|SegFormer"
    "std_mixup|outside15k|DeepLabV3+"
    "std_mixup|outside15k|PSPNet"
    "std_mixup|outside15k|SegFormer"
    "std_randaugment|outside15k|SegFormer"
)

# Missing configurations (no test results)
MISSING_CONFIGS=(
    "baseline|mapillaryvistas|DeepLabV3+"
    "gen_Attribute_Hallucination|mapillaryvistas|DeepLabV3+"
    "gen_Attribute_Hallucination|mapillaryvistas|PSPNet"
    "gen_Attribute_Hallucination|mapillaryvistas|SegFormer"
    "gen_CNetSeg|mapillaryvistas|PSPNet"
    "gen_CNetSeg|mapillaryvistas|SegFormer"
    "gen_IP2P|mapillaryvistas|PSPNet"
    "gen_Qwen_Image_Edit|bdd10k|DeepLabV3+"
    "gen_Qwen_Image_Edit|bdd10k|PSPNet"
    "gen_Qwen_Image_Edit|bdd10k|SegFormer"
    "gen_Weather_Effect_Generator|mapillaryvistas|DeepLabV3+"
    "gen_Weather_Effect_Generator|mapillaryvistas|PSPNet"
    "gen_albumentations_weather|mapillaryvistas|DeepLabV3+"
    "gen_albumentations_weather|mapillaryvistas|PSPNet"
    "gen_augmenters|mapillaryvistas|DeepLabV3+"
    "gen_augmenters|mapillaryvistas|SegFormer"
    "gen_automold|mapillaryvistas|DeepLabV3+"
    "gen_automold|mapillaryvistas|PSPNet"
    "gen_cycleGAN|mapillaryvistas|DeepLabV3+"
    "gen_cycleGAN|mapillaryvistas|PSPNet"
    "gen_flux_kontext|mapillaryvistas|PSPNet"
    "gen_flux_kontext|outside15k|DeepLabV3+"
    "gen_flux_kontext|outside15k|PSPNet"
    "gen_step1x_new|bdd10k|DeepLabV3+"
    "gen_step1x_new|bdd10k|PSPNet"
    "gen_step1x_new|bdd10k|SegFormer"
    "std_minimal|bdd10k|DeepLabV3+"
    "std_minimal|bdd10k|PSPNet"
    "std_minimal|idd-aw|DeepLabV3+"
    "std_minimal|idd-aw|PSPNet"
)

# Combine all configs
ALL_CONFIGS=("${BUGGY_CONFIGS[@]}" "${MISSING_CONFIGS[@]}")

echo "=== Submitting ${#ALL_CONFIGS[@]} Test Jobs ==="
echo "  Buggy: ${#BUGGY_CONFIGS[@]}"
echo "  Missing: ${#MISSING_CONFIGS[@]}"
echo ""

submitted=0
skipped=0

for config in "${ALL_CONFIGS[@]}"; do
    IFS='|' read -r strategy dataset model_display <<< "$config"
    
    # Get actual model directory name
    model=$(get_model_dir "$model_display" "$strategy")
    
    # Handle dataset directory naming
    if [ "$dataset" = "idd-aw" ]; then
        # Try both idd-aw_cd and iddaw_cd
        if [ -d "${WEIGHTS_ROOT}/${strategy}/idd-aw_cd" ]; then
            dataset_dir="idd-aw_cd"
        elif [ -d "${WEIGHTS_ROOT}/${strategy}/iddaw_cd" ]; then
            dataset_dir="iddaw_cd"
        else
            echo "SKIP: No dataset dir for $strategy/idd-aw"
            ((skipped++))
            continue
        fi
    else
        dataset_dir="${dataset}_cd"
    fi
    
    weights_dir="${WEIGHTS_ROOT}/${strategy}/${dataset_dir}/${model}"
    config_path="${weights_dir}/training_config.py"
    checkpoint_path="${weights_dir}/iter_80000.pth"
    output_dir="${weights_dir}/test_results_detailed"
    
    dataset_display=$(get_dataset_display "$dataset")
    
    # Create short job name
    short_strategy=$(echo "$strategy" | sed 's/gen_/g/' | cut -c1-10)
    short_dataset=$(echo "$dataset" | cut -c1-4)
    short_model=$(echo "$model_display" | cut -c1-3)
    job_name="fg_${short_strategy}_${short_dataset}_${short_model}"
    
    if [ ! -f "$checkpoint_path" ]; then
        echo "SKIP: Missing checkpoint for $strategy/$dataset/$model_display"
        ((skipped++))
        continue
    fi
    
    if [ ! -f "$config_path" ]; then
        echo "SKIP: Missing config for $strategy/$dataset/$model_display"
        ((skipped++))
        continue
    fi
    
    echo "Submitting: $strategy/$dataset/$model_display"
    
    bsub -J "$job_name" \
        -q BatchGPU \
        -n 10 \
        -R "span[hosts=1]" \
        -R "rusage[mem=8000]" \
        -gpu "num=1:gmem=20G" \
        -W 5:00 \
        -o "${LOG_DIR}/${job_name}_%J.out" \
        -e "${LOG_DIR}/${job_name}_%J.err" \
        "source ~/.bashrc && conda activate prove && cd ${SCRIPT_DIR} && python fine_grained_test.py --config ${config_path} --checkpoint ${checkpoint_path} --dataset ${dataset_display} --output-dir ${output_dir} --batch-size 8"
    
    ((submitted++))
    
    # Small delay to avoid overwhelming the scheduler
    sleep 0.3
done

echo ""
echo "=== Summary ==="
echo "Submitted: $submitted jobs"
echo "Skipped: $skipped jobs"
