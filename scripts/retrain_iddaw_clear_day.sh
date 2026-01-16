#!/bin/bash
# Retrain IDD-AW models with CORRECT domain-filter clear_day
# Previous retrain_iddaw_corrupted.sh was missing --domain-filter clear_day
# This resulted in models being trained on all domains instead of clear_day only
# Generated: 2026-01-15

PROJECT_ROOT="/home/mima2416/repositories/PROVE"
WEIGHTS_ROOT="/scratch/aaa_exchange/AWARE/WEIGHTS"
LOG_DIR="${PROJECT_ROOT}/logs"

submitted=0

echo "=== Retraining IDD-AW Models (Stage 1 - Clear Day Domain Filter) ==="
echo ""
echo "IMPORTANT: These models MUST use --domain-filter clear_day for Stage 1 training"
echo "The models should be saved to idd-aw_cd directories"
echo ""

# Helper to check if corrupted backup exists
check_corrupted() {
    local path=$1
    if [ -d "${path}_corrupted_labels_backup" ]; then
        return 0
    fi
    return 1
}

# ===============================
# BASELINE MODELS
# ===============================
echo "=== Baseline Models ==="

# baseline/idd-aw_cd/DeepLabV3+
if check_corrupted "${WEIGHTS_ROOT}/baseline/idd-aw_cd/deeplabv3plus_r50"; then
    echo "Submitting: baseline/idd-aw_cd/DeepLabV3+..."
    bsub -J "rtfix2_baseline_iddaw_dlv3" \
        -q BatchGPU \
        -R "rusage[mem=16000]" \
        -gpu "num=1:gmem=16G" \
        -W 16:00 \
        -o "${LOG_DIR}/rtfix2_baseline_iddaw_dlv3_%J.out" \
        -e "${LOG_DIR}/rtfix2_baseline_iddaw_dlv3_%J.err" \
        "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python unified_training.py --dataset IDD-AW --model deeplabv3plus_r50 --strategy baseline --domain-filter clear_day --no-early-stop --max-iters 80000"
    ((submitted++))
fi

# baseline/idd-aw_cd/PSPNet
if check_corrupted "${WEIGHTS_ROOT}/baseline/idd-aw_cd/pspnet_r50"; then
    echo "Submitting: baseline/idd-aw_cd/PSPNet..."
    bsub -J "rtfix2_baseline_iddaw_psp" \
        -q BatchGPU \
        -R "rusage[mem=16000]" \
        -gpu "num=1:gmem=16G" \
        -W 16:00 \
        -o "${LOG_DIR}/rtfix2_baseline_iddaw_psp_%J.out" \
        -e "${LOG_DIR}/rtfix2_baseline_iddaw_psp_%J.err" \
        "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python unified_training.py --dataset IDD-AW --model pspnet_r50 --strategy baseline --domain-filter clear_day --no-early-stop --max-iters 80000"
    ((submitted++))
fi

# baseline/idd-aw_cd/SegFormer
if check_corrupted "${WEIGHTS_ROOT}/baseline/idd-aw_cd/segformer_mit-b5"; then
    echo "Submitting: baseline/idd-aw_cd/SegFormer..."
    bsub -J "rtfix2_baseline_iddaw_segf" \
        -q BatchGPU \
        -R "rusage[mem=20000]" \
        -gpu "num=1:gmem=24G" \
        -W 24:00 \
        -o "${LOG_DIR}/rtfix2_baseline_iddaw_segf_%J.out" \
        -e "${LOG_DIR}/rtfix2_baseline_iddaw_segf_%J.err" \
        "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python unified_training.py --dataset IDD-AW --model segformer_mit-b5 --strategy baseline --domain-filter clear_day --no-early-stop --max-iters 80000"
    ((submitted++))
fi

# ===============================
# GEN_* STRATEGIES
# ===============================

# gen_Attribute_Hallucination
echo ""
echo "=== gen_Attribute_Hallucination ==="
for model in deeplabv3plus_r50 pspnet_r50 segformer_mit-b5; do
    if check_corrupted "${WEIGHTS_ROOT}/gen_Attribute_Hallucination/idd-aw_cd/${model}_ratio0p50"; then
        short_model=$(echo $model | sed 's/deeplabv3plus_r50/dlv3/;s/pspnet_r50/psp/;s/segformer_mit-b5/segf/')
        mem=16000
        gmem=16G
        time=16:00
        if [ "$model" == "segformer_mit-b5" ]; then
            mem=20000
            gmem=24G
            time=24:00
        fi
        echo "Submitting: gen_Attribute_Hallucination/idd-aw_cd/$model..."
        bsub -J "rtfix2_AttrHall_iddaw_${short_model}" \
            -q BatchGPU \
            -R "rusage[mem=${mem}]" \
            -gpu "num=1:gmem=${gmem}" \
            -W ${time} \
            -o "${LOG_DIR}/rtfix2_AttrHall_iddaw_${short_model}_%J.out" \
            -e "${LOG_DIR}/rtfix2_AttrHall_iddaw_${short_model}_%J.err" \
            "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python unified_training.py --dataset IDD-AW --model $model --strategy gen_Attribute_Hallucination --domain-filter clear_day --real-gen-ratio 0.5 --no-early-stop --max-iters 80000"
        ((submitted++))
    fi
done

# gen_CNetSeg
echo ""
echo "=== gen_CNetSeg ==="
for model in deeplabv3plus_r50 pspnet_r50 segformer_mit-b5; do
    if check_corrupted "${WEIGHTS_ROOT}/gen_CNetSeg/idd-aw_cd/${model}_ratio0p50"; then
        short_model=$(echo $model | sed 's/deeplabv3plus_r50/dlv3/;s/pspnet_r50/psp/;s/segformer_mit-b5/segf/')
        mem=16000
        gmem=16G
        time=16:00
        if [ "$model" == "segformer_mit-b5" ]; then
            mem=20000
            gmem=24G
            time=24:00
        fi
        echo "Submitting: gen_CNetSeg/idd-aw_cd/$model..."
        bsub -J "rtfix2_CNetSeg_iddaw_${short_model}" \
            -q BatchGPU \
            -R "rusage[mem=${mem}]" \
            -gpu "num=1:gmem=${gmem}" \
            -W ${time} \
            -o "${LOG_DIR}/rtfix2_CNetSeg_iddaw_${short_model}_%J.out" \
            -e "${LOG_DIR}/rtfix2_CNetSeg_iddaw_${short_model}_%J.err" \
            "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python unified_training.py --dataset IDD-AW --model $model --strategy gen_CNetSeg --domain-filter clear_day --real-gen-ratio 0.5 --no-early-stop --max-iters 80000"
        ((submitted++))
    fi
done

# gen_CUT
echo ""
echo "=== gen_CUT ==="
for model in deeplabv3plus_r50 pspnet_r50 segformer_mit-b5; do
    if check_corrupted "${WEIGHTS_ROOT}/gen_CUT/idd-aw_cd/${model}_ratio0p50"; then
        short_model=$(echo $model | sed 's/deeplabv3plus_r50/dlv3/;s/pspnet_r50/psp/;s/segformer_mit-b5/segf/')
        mem=16000
        gmem=16G
        time=16:00
        if [ "$model" == "segformer_mit-b5" ]; then
            mem=20000
            gmem=24G
            time=24:00
        fi
        echo "Submitting: gen_CUT/idd-aw_cd/$model..."
        bsub -J "rtfix2_CUT_iddaw_${short_model}" \
            -q BatchGPU \
            -R "rusage[mem=${mem}]" \
            -gpu "num=1:gmem=${gmem}" \
            -W ${time} \
            -o "${LOG_DIR}/rtfix2_CUT_iddaw_${short_model}_%J.out" \
            -e "${LOG_DIR}/rtfix2_CUT_iddaw_${short_model}_%J.err" \
            "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python unified_training.py --dataset IDD-AW --model $model --strategy gen_CUT --domain-filter clear_day --real-gen-ratio 0.5 --no-early-stop --max-iters 80000"
        ((submitted++))
    fi
done

# gen_IP2P
echo ""
echo "=== gen_IP2P ==="
for model in deeplabv3plus_r50 pspnet_r50 segformer_mit-b5; do
    if check_corrupted "${WEIGHTS_ROOT}/gen_IP2P/idd-aw_cd/${model}_ratio0p50"; then
        short_model=$(echo $model | sed 's/deeplabv3plus_r50/dlv3/;s/pspnet_r50/psp/;s/segformer_mit-b5/segf/')
        mem=16000
        gmem=16G
        time=16:00
        if [ "$model" == "segformer_mit-b5" ]; then
            mem=20000
            gmem=24G
            time=24:00
        fi
        echo "Submitting: gen_IP2P/idd-aw_cd/$model..."
        bsub -J "rtfix2_IP2P_iddaw_${short_model}" \
            -q BatchGPU \
            -R "rusage[mem=${mem}]" \
            -gpu "num=1:gmem=${gmem}" \
            -W ${time} \
            -o "${LOG_DIR}/rtfix2_IP2P_iddaw_${short_model}_%J.out" \
            -e "${LOG_DIR}/rtfix2_IP2P_iddaw_${short_model}_%J.err" \
            "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python unified_training.py --dataset IDD-AW --model $model --strategy gen_IP2P --domain-filter clear_day --real-gen-ratio 0.5 --no-early-stop --max-iters 80000"
        ((submitted++))
    fi
done

# ===============================
# STD_* STRATEGIES
# ===============================

# photometric_distort
echo ""
echo "=== photometric_distort ==="
for model in deeplabv3plus_r50 pspnet_r50 segformer_mit-b5; do
    if check_corrupted "${WEIGHTS_ROOT}/photometric_distort/idd-aw_cd/${model}"; then
        short_model=$(echo $model | sed 's/deeplabv3plus_r50/dlv3/;s/pspnet_r50/psp/;s/segformer_mit-b5/segf/')
        mem=16000
        gmem=16G
        time=16:00
        if [ "$model" == "segformer_mit-b5" ]; then
            mem=20000
            gmem=24G
            time=24:00
        fi
        echo "Submitting: photometric_distort/idd-aw_cd/$model..."
        bsub -J "rtfix2_photom_iddaw_${short_model}" \
            -q BatchGPU \
            -R "rusage[mem=${mem}]" \
            -gpu "num=1:gmem=${gmem}" \
            -W ${time} \
            -o "${LOG_DIR}/rtfix2_photom_iddaw_${short_model}_%J.out" \
            -e "${LOG_DIR}/rtfix2_photom_iddaw_${short_model}_%J.err" \
            "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python unified_training.py --dataset IDD-AW --model $model --strategy photometric_distort --domain-filter clear_day --no-early-stop --max-iters 80000"
        ((submitted++))
    fi
done

# std_autoaugment
echo ""
echo "=== std_autoaugment ==="
for model in deeplabv3plus_r50 pspnet_r50 segformer_mit-b5; do
    if check_corrupted "${WEIGHTS_ROOT}/std_autoaugment/idd-aw_cd/${model}"; then
        short_model=$(echo $model | sed 's/deeplabv3plus_r50/dlv3/;s/pspnet_r50/psp/;s/segformer_mit-b5/segf/')
        mem=16000
        gmem=16G
        time=16:00
        if [ "$model" == "segformer_mit-b5" ]; then
            mem=20000
            gmem=24G
            time=24:00
        fi
        echo "Submitting: std_autoaugment/idd-aw_cd/$model..."
        bsub -J "rtfix2_autoaug_iddaw_${short_model}" \
            -q BatchGPU \
            -R "rusage[mem=${mem}]" \
            -gpu "num=1:gmem=${gmem}" \
            -W ${time} \
            -o "${LOG_DIR}/rtfix2_autoaug_iddaw_${short_model}_%J.out" \
            -e "${LOG_DIR}/rtfix2_autoaug_iddaw_${short_model}_%J.err" \
            "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python unified_training.py --dataset IDD-AW --model $model --strategy std_autoaugment --domain-filter clear_day --no-early-stop --max-iters 80000"
        ((submitted++))
    fi
done

# std_cutmix
echo ""
echo "=== std_cutmix ==="
for model in deeplabv3plus_r50 pspnet_r50 segformer_mit-b5; do
    if check_corrupted "${WEIGHTS_ROOT}/std_cutmix/idd-aw_cd/${model}"; then
        short_model=$(echo $model | sed 's/deeplabv3plus_r50/dlv3/;s/pspnet_r50/psp/;s/segformer_mit-b5/segf/')
        mem=16000
        gmem=16G
        time=16:00
        if [ "$model" == "segformer_mit-b5" ]; then
            mem=20000
            gmem=24G
            time=24:00
        fi
        echo "Submitting: std_cutmix/idd-aw_cd/$model..."
        bsub -J "rtfix2_cutmix_iddaw_${short_model}" \
            -q BatchGPU \
            -R "rusage[mem=${mem}]" \
            -gpu "num=1:gmem=${gmem}" \
            -W ${time} \
            -o "${LOG_DIR}/rtfix2_cutmix_iddaw_${short_model}_%J.out" \
            -e "${LOG_DIR}/rtfix2_cutmix_iddaw_${short_model}_%J.err" \
            "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python unified_training.py --dataset IDD-AW --model $model --strategy std_cutmix --domain-filter clear_day --no-early-stop --max-iters 80000"
        ((submitted++))
    fi
done

# std_mixup
echo ""
echo "=== std_mixup ==="
for model in deeplabv3plus_r50 pspnet_r50 segformer_mit-b5; do
    if check_corrupted "${WEIGHTS_ROOT}/std_mixup/idd-aw_cd/${model}"; then
        short_model=$(echo $model | sed 's/deeplabv3plus_r50/dlv3/;s/pspnet_r50/psp/;s/segformer_mit-b5/segf/')
        mem=16000
        gmem=16G
        time=16:00
        if [ "$model" == "segformer_mit-b5" ]; then
            mem=20000
            gmem=24G
            time=24:00
        fi
        echo "Submitting: std_mixup/idd-aw_cd/$model..."
        bsub -J "rtfix2_mixup_iddaw_${short_model}" \
            -q BatchGPU \
            -R "rusage[mem=${mem}]" \
            -gpu "num=1:gmem=${gmem}" \
            -W ${time} \
            -o "${LOG_DIR}/rtfix2_mixup_iddaw_${short_model}_%J.out" \
            -e "${LOG_DIR}/rtfix2_mixup_iddaw_${short_model}_%J.err" \
            "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python unified_training.py --dataset IDD-AW --model $model --strategy std_mixup --domain-filter clear_day --no-early-stop --max-iters 80000"
        ((submitted++))
    fi
done

# std_randaugment
echo ""
echo "=== std_randaugment ==="
for model in deeplabv3plus_r50 pspnet_r50 segformer_mit-b5; do
    if check_corrupted "${WEIGHTS_ROOT}/std_randaugment/idd-aw_cd/${model}"; then
        short_model=$(echo $model | sed 's/deeplabv3plus_r50/dlv3/;s/pspnet_r50/psp/;s/segformer_mit-b5/segf/')
        mem=16000
        gmem=16G
        time=16:00
        if [ "$model" == "segformer_mit-b5" ]; then
            mem=20000
            gmem=24G
            time=24:00
        fi
        echo "Submitting: std_randaugment/idd-aw_cd/$model..."
        bsub -J "rtfix2_randaug_iddaw_${short_model}" \
            -q BatchGPU \
            -R "rusage[mem=${mem}]" \
            -gpu "num=1:gmem=${gmem}" \
            -W ${time} \
            -o "${LOG_DIR}/rtfix2_randaug_iddaw_${short_model}_%J.out" \
            -e "${LOG_DIR}/rtfix2_randaug_iddaw_${short_model}_%J.err" \
            "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python unified_training.py --dataset IDD-AW --model $model --strategy std_randaugment --domain-filter clear_day --no-early-stop --max-iters 80000"
        ((submitted++))
    fi
done

echo ""
echo "=== Summary ==="
echo "Submitted: $submitted training jobs"
echo ""
echo "Notes:"
echo "  - All jobs use --domain-filter clear_day"
echo "  - Models will be saved to idd-aw_cd directories"
echo "  - Monitor with: bjobs -w | grep rtfix2_"
echo "  - Expected training time: 2-4 hours per model"
