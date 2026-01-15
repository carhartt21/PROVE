#!/bin/bash
# Submit test jobs for IDD-AW configurations after retraining
# Generated: 2026-01-14
# NOTE: Run this script AFTER retraining jobs complete!

PROJECT_ROOT="/home/mima2416/repositories/PROVE"
WEIGHTS_ROOT="/scratch/aaa_exchange/AWARE/WEIGHTS"
LOG_DIR="${PROJECT_ROOT}/logs"

submitted=0
skipped=0

echo "=== Submitting IDD-AW Test Jobs ==="
echo ""
echo "NOTE: Tests will be skipped if checkpoint not found (training not complete)!"
echo ""

# Helper function to submit test
submit_test() {
    local NAME="$1"
    local STRATEGY="$2"
    local MODEL="$3"
    local CHECKPOINT="$4"
    local CONFIG="$5"
    local OUTPUT_DIR="$6"
    
    echo "Testing ${NAME}..."
    if [ -f "$CHECKPOINT" ]; then
        bsub -J "test_rtfix_${NAME}" \
            -q BatchGPU \
            -n 10 \
            -R "span[hosts=1]" \
            -R "rusage[mem=8000]" \
            -gpu "num=1:gmem=20G" \
            -W 04:00 \
            -o "${LOG_DIR}/test_rtfix_${NAME}_%J.out" \
            -e "${LOG_DIR}/test_rtfix_${NAME}_%J.err" \
            "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python fine_grained_test.py --config ${CONFIG} --checkpoint ${CHECKPOINT} --dataset IDD-AW --output-dir ${OUTPUT_DIR}"
        ((submitted++))
    else
        echo "  SKIPPED: Checkpoint not found: $CHECKPOINT"
        ((skipped++))
    fi
}

# 1. baseline/idd-aw/DeepLabV3+
submit_test "baseline_iddaw_dlv3" "baseline" "deeplabv3plus_r50" \
    "${WEIGHTS_ROOT}/baseline/idd-aw_cd/deeplabv3plus_r50/iter_80000.pth" \
    "${WEIGHTS_ROOT}/baseline/idd-aw_cd/deeplabv3plus_r50/deeplabv3plus_r50-d8_4xb4-80k_cityscapes-512x1024.py" \
    "results/baseline/idd-aw_cd/deeplabv3plus_r50"

# 2. baseline/idd-aw/SegFormer
submit_test "baseline_iddaw_segf" "baseline" "segformer_mit-b5" \
    "${WEIGHTS_ROOT}/baseline/idd-aw_cd/segformer_mit-b5/iter_80000.pth" \
    "${WEIGHTS_ROOT}/baseline/idd-aw_cd/segformer_mit-b5/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py" \
    "results/baseline/idd-aw_cd/segformer_mit-b5"

# 3. gen_Attribute_Hallucination/idd-aw/DeepLabV3+
submit_test "AttrHall_iddaw_dlv3" "gen_Attribute_Hallucination" "deeplabv3plus_r50_ratio0p50" \
    "${WEIGHTS_ROOT}/gen_Attribute_Hallucination/idd-aw_cd/deeplabv3plus_r50_ratio0p50/iter_80000.pth" \
    "${WEIGHTS_ROOT}/gen_Attribute_Hallucination/idd-aw_cd/deeplabv3plus_r50_ratio0p50/deeplabv3plus_r50-d8_4xb4-80k_cityscapes-512x1024.py" \
    "results/gen_Attribute_Hallucination/idd-aw_cd/deeplabv3plus_r50_ratio0p50"

# 4. gen_Attribute_Hallucination/idd-aw/PSPNet
submit_test "AttrHall_iddaw_psp" "gen_Attribute_Hallucination" "pspnet_r50_ratio0p50" \
    "${WEIGHTS_ROOT}/gen_Attribute_Hallucination/idd-aw_cd/pspnet_r50_ratio0p50/iter_80000.pth" \
    "${WEIGHTS_ROOT}/gen_Attribute_Hallucination/idd-aw_cd/pspnet_r50_ratio0p50/pspnet_r50-d8_4xb4-80k_cityscapes-512x1024.py" \
    "results/gen_Attribute_Hallucination/idd-aw_cd/pspnet_r50_ratio0p50"

# 5. gen_CNetSeg/idd-aw/DeepLabV3+
submit_test "CNetSeg_iddaw_dlv3" "gen_CNetSeg" "deeplabv3plus_r50_ratio0p50" \
    "${WEIGHTS_ROOT}/gen_CNetSeg/idd-aw_cd/deeplabv3plus_r50_ratio0p50/iter_80000.pth" \
    "${WEIGHTS_ROOT}/gen_CNetSeg/idd-aw_cd/deeplabv3plus_r50_ratio0p50/deeplabv3plus_r50-d8_4xb4-80k_cityscapes-512x1024.py" \
    "results/gen_CNetSeg/idd-aw_cd/deeplabv3plus_r50_ratio0p50"

# 6. gen_CNetSeg/idd-aw/PSPNet
submit_test "CNetSeg_iddaw_psp" "gen_CNetSeg" "pspnet_r50_ratio0p50" \
    "${WEIGHTS_ROOT}/gen_CNetSeg/idd-aw_cd/pspnet_r50_ratio0p50/iter_80000.pth" \
    "${WEIGHTS_ROOT}/gen_CNetSeg/idd-aw_cd/pspnet_r50_ratio0p50/pspnet_r50-d8_4xb4-80k_cityscapes-512x1024.py" \
    "results/gen_CNetSeg/idd-aw_cd/pspnet_r50_ratio0p50"

# 7. gen_CUT/idd-aw/DeepLabV3+
submit_test "CUT_iddaw_dlv3" "gen_CUT" "deeplabv3plus_r50_ratio0p50" \
    "${WEIGHTS_ROOT}/gen_CUT/idd-aw_cd/deeplabv3plus_r50_ratio0p50/iter_80000.pth" \
    "${WEIGHTS_ROOT}/gen_CUT/idd-aw_cd/deeplabv3plus_r50_ratio0p50/deeplabv3plus_r50-d8_4xb4-80k_cityscapes-512x1024.py" \
    "results/gen_CUT/idd-aw_cd/deeplabv3plus_r50_ratio0p50"

# 8. gen_CUT/idd-aw/PSPNet
submit_test "CUT_iddaw_psp" "gen_CUT" "pspnet_r50_ratio0p50" \
    "${WEIGHTS_ROOT}/gen_CUT/idd-aw_cd/pspnet_r50_ratio0p50/iter_80000.pth" \
    "${WEIGHTS_ROOT}/gen_CUT/idd-aw_cd/pspnet_r50_ratio0p50/pspnet_r50-d8_4xb4-80k_cityscapes-512x1024.py" \
    "results/gen_CUT/idd-aw_cd/pspnet_r50_ratio0p50"

# 9. gen_CUT/idd-aw/SegFormer
submit_test "CUT_iddaw_segf" "gen_CUT" "segformer_mit-b5_ratio0p50" \
    "${WEIGHTS_ROOT}/gen_CUT/idd-aw_cd/segformer_mit-b5_ratio0p50/iter_80000.pth" \
    "${WEIGHTS_ROOT}/gen_CUT/idd-aw_cd/segformer_mit-b5_ratio0p50/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py" \
    "results/gen_CUT/idd-aw_cd/segformer_mit-b5_ratio0p50"

# 10. gen_IP2P/idd-aw/DeepLabV3+
submit_test "IP2P_iddaw_dlv3" "gen_IP2P" "deeplabv3plus_r50_ratio0p50" \
    "${WEIGHTS_ROOT}/gen_IP2P/idd-aw_cd/deeplabv3plus_r50_ratio0p50/iter_80000.pth" \
    "${WEIGHTS_ROOT}/gen_IP2P/idd-aw_cd/deeplabv3plus_r50_ratio0p50/deeplabv3plus_r50-d8_4xb4-80k_cityscapes-512x1024.py" \
    "results/gen_IP2P/idd-aw_cd/deeplabv3plus_r50_ratio0p50"

# 11. gen_IP2P/idd-aw/PSPNet
submit_test "IP2P_iddaw_psp" "gen_IP2P" "pspnet_r50_ratio0p50" \
    "${WEIGHTS_ROOT}/gen_IP2P/idd-aw_cd/pspnet_r50_ratio0p50/iter_80000.pth" \
    "${WEIGHTS_ROOT}/gen_IP2P/idd-aw_cd/pspnet_r50_ratio0p50/pspnet_r50-d8_4xb4-80k_cityscapes-512x1024.py" \
    "results/gen_IP2P/idd-aw_cd/pspnet_r50_ratio0p50"

echo ""
echo "=== Summary ==="
echo "Submitted: $submitted test jobs"
echo "Skipped: $skipped (checkpoints not found)"
echo ""
echo "If tests were skipped, run this script again after training completes."
echo "Check training status with: bjobs -w | grep rtfix"
