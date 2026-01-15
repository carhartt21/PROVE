#!/bin/bash
# Submit test jobs for retrained IDD-AW models with correct paths
# The models were saved to iddaw_ad (not idd-aw_cd) directory
# Generated: 2026-01-15

PROJECT_ROOT="/home/mima2416/repositories/PROVE"
WEIGHTS_ROOT="/scratch/aaa_exchange/AWARE/WEIGHTS"
LOG_DIR="${PROJECT_ROOT}/logs"

submitted=0

echo "=== Submitting IDD-AW Test Jobs (Fixed Paths) ==="
echo ""

# 1. baseline/iddaw_ad/DeepLabV3+
echo "1. Testing baseline/iddaw_ad/DeepLabV3+..."
CONFIG="${WEIGHTS_ROOT}/baseline/iddaw_ad/deeplabv3plus_r50/configs/deeplabv3plus_r50-d8_4xb4-80k_cityscapes-512x1024.py"
CKPT="${WEIGHTS_ROOT}/baseline/iddaw_ad/deeplabv3plus_r50/iter_80000.pth"
if [ -f "$CKPT" ]; then
    bsub -J "test2_baseline_iddaw_dlv3" \
        -q BatchGPU -n 10 -R "span[hosts=1]" -R "rusage[mem=8000]" -gpu "num=1:gmem=20G" -W 04:00 \
        -o "${LOG_DIR}/test2_baseline_iddaw_dlv3_%J.out" -e "${LOG_DIR}/test2_baseline_iddaw_dlv3_%J.err" \
        "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python fine_grained_test.py --config ${CONFIG} --checkpoint ${CKPT} --dataset IDD-AW --output-dir results/baseline/iddaw_ad/deeplabv3plus_r50"
    ((submitted++))
else
    echo "  SKIP: Checkpoint not found: $CKPT"
fi

# 2. baseline/iddaw_ad/SegFormer
echo "2. Testing baseline/iddaw_ad/SegFormer..."
CONFIG="${WEIGHTS_ROOT}/baseline/iddaw_ad/segformer_mit-b5/configs/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py"
CKPT="${WEIGHTS_ROOT}/baseline/iddaw_ad/segformer_mit-b5/iter_80000.pth"
if [ -f "$CKPT" ]; then
    bsub -J "test2_baseline_iddaw_segf" \
        -q BatchGPU -n 10 -R "span[hosts=1]" -R "rusage[mem=8000]" -gpu "num=1:gmem=20G" -W 04:00 \
        -o "${LOG_DIR}/test2_baseline_iddaw_segf_%J.out" -e "${LOG_DIR}/test2_baseline_iddaw_segf_%J.err" \
        "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python fine_grained_test.py --config ${CONFIG} --checkpoint ${CKPT} --dataset IDD-AW --output-dir results/baseline/iddaw_ad/segformer_mit-b5"
    ((submitted++))
else
    echo "  SKIP: Checkpoint not found: $CKPT"
fi

# 3. gen_Attribute_Hallucination/iddaw_ad/DeepLabV3+
echo "3. Testing gen_Attribute_Hallucination/iddaw_ad/DeepLabV3+..."
CONFIG="${WEIGHTS_ROOT}/gen_Attribute_Hallucination/iddaw_ad/deeplabv3plus_r50_ratio0p50/configs/deeplabv3plus_r50-d8_4xb4-80k_cityscapes-512x1024.py"
CKPT="${WEIGHTS_ROOT}/gen_Attribute_Hallucination/iddaw_ad/deeplabv3plus_r50_ratio0p50/iter_80000.pth"
if [ -f "$CKPT" ]; then
    bsub -J "test2_AttrHall_iddaw_dlv3" \
        -q BatchGPU -n 10 -R "span[hosts=1]" -R "rusage[mem=8000]" -gpu "num=1:gmem=20G" -W 04:00 \
        -o "${LOG_DIR}/test2_AttrHall_iddaw_dlv3_%J.out" -e "${LOG_DIR}/test2_AttrHall_iddaw_dlv3_%J.err" \
        "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python fine_grained_test.py --config ${CONFIG} --checkpoint ${CKPT} --dataset IDD-AW --output-dir results/gen_Attribute_Hallucination/iddaw_ad/deeplabv3plus_r50_ratio0p50"
    ((submitted++))
else
    echo "  SKIP: Checkpoint not found: $CKPT"
fi

# 4. gen_Attribute_Hallucination/iddaw_ad/PSPNet
echo "4. Testing gen_Attribute_Hallucination/iddaw_ad/PSPNet..."
CONFIG="${WEIGHTS_ROOT}/gen_Attribute_Hallucination/iddaw_ad/pspnet_r50_ratio0p50/configs/pspnet_r50-d8_4xb4-80k_cityscapes-512x1024.py"
CKPT="${WEIGHTS_ROOT}/gen_Attribute_Hallucination/iddaw_ad/pspnet_r50_ratio0p50/iter_80000.pth"
if [ -f "$CKPT" ]; then
    bsub -J "test2_AttrHall_iddaw_psp" \
        -q BatchGPU -n 10 -R "span[hosts=1]" -R "rusage[mem=8000]" -gpu "num=1:gmem=20G" -W 04:00 \
        -o "${LOG_DIR}/test2_AttrHall_iddaw_psp_%J.out" -e "${LOG_DIR}/test2_AttrHall_iddaw_psp_%J.err" \
        "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python fine_grained_test.py --config ${CONFIG} --checkpoint ${CKPT} --dataset IDD-AW --output-dir results/gen_Attribute_Hallucination/iddaw_ad/pspnet_r50_ratio0p50"
    ((submitted++))
else
    echo "  SKIP: Checkpoint not found: $CKPT"
fi

# 5. gen_CNetSeg/iddaw_ad/DeepLabV3+
echo "5. Testing gen_CNetSeg/iddaw_ad/DeepLabV3+..."
CONFIG="${WEIGHTS_ROOT}/gen_CNetSeg/iddaw_ad/deeplabv3plus_r50_ratio0p50/configs/deeplabv3plus_r50-d8_4xb4-80k_cityscapes-512x1024.py"
CKPT="${WEIGHTS_ROOT}/gen_CNetSeg/iddaw_ad/deeplabv3plus_r50_ratio0p50/iter_80000.pth"
if [ -f "$CKPT" ]; then
    bsub -J "test2_CNetSeg_iddaw_dlv3" \
        -q BatchGPU -n 10 -R "span[hosts=1]" -R "rusage[mem=8000]" -gpu "num=1:gmem=20G" -W 04:00 \
        -o "${LOG_DIR}/test2_CNetSeg_iddaw_dlv3_%J.out" -e "${LOG_DIR}/test2_CNetSeg_iddaw_dlv3_%J.err" \
        "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python fine_grained_test.py --config ${CONFIG} --checkpoint ${CKPT} --dataset IDD-AW --output-dir results/gen_CNetSeg/iddaw_ad/deeplabv3plus_r50_ratio0p50"
    ((submitted++))
else
    echo "  SKIP: Checkpoint not found: $CKPT"
fi

# 6. gen_CNetSeg/iddaw_ad/PSPNet
echo "6. Testing gen_CNetSeg/iddaw_ad/PSPNet..."
CONFIG="${WEIGHTS_ROOT}/gen_CNetSeg/iddaw_ad/pspnet_r50_ratio0p50/configs/pspnet_r50-d8_4xb4-80k_cityscapes-512x1024.py"
CKPT="${WEIGHTS_ROOT}/gen_CNetSeg/iddaw_ad/pspnet_r50_ratio0p50/iter_80000.pth"
if [ -f "$CKPT" ]; then
    bsub -J "test2_CNetSeg_iddaw_psp" \
        -q BatchGPU -n 10 -R "span[hosts=1]" -R "rusage[mem=8000]" -gpu "num=1:gmem=20G" -W 04:00 \
        -o "${LOG_DIR}/test2_CNetSeg_iddaw_psp_%J.out" -e "${LOG_DIR}/test2_CNetSeg_iddaw_psp_%J.err" \
        "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python fine_grained_test.py --config ${CONFIG} --checkpoint ${CKPT} --dataset IDD-AW --output-dir results/gen_CNetSeg/iddaw_ad/pspnet_r50_ratio0p50"
    ((submitted++))
else
    echo "  SKIP: Checkpoint not found: $CKPT"
fi

# 7. gen_CUT/iddaw_ad/DeepLabV3+
echo "7. Testing gen_CUT/iddaw_ad/DeepLabV3+..."
CONFIG="${WEIGHTS_ROOT}/gen_CUT/iddaw_ad/deeplabv3plus_r50_ratio0p50/configs/deeplabv3plus_r50-d8_4xb4-80k_cityscapes-512x1024.py"
CKPT="${WEIGHTS_ROOT}/gen_CUT/iddaw_ad/deeplabv3plus_r50_ratio0p50/iter_80000.pth"
if [ -f "$CKPT" ]; then
    bsub -J "test2_CUT_iddaw_dlv3" \
        -q BatchGPU -n 10 -R "span[hosts=1]" -R "rusage[mem=8000]" -gpu "num=1:gmem=20G" -W 04:00 \
        -o "${LOG_DIR}/test2_CUT_iddaw_dlv3_%J.out" -e "${LOG_DIR}/test2_CUT_iddaw_dlv3_%J.err" \
        "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python fine_grained_test.py --config ${CONFIG} --checkpoint ${CKPT} --dataset IDD-AW --output-dir results/gen_CUT/iddaw_ad/deeplabv3plus_r50_ratio0p50"
    ((submitted++))
else
    echo "  SKIP: Checkpoint not found: $CKPT"
fi

# 8. gen_CUT/iddaw_ad/PSPNet
echo "8. Testing gen_CUT/iddaw_ad/PSPNet..."
CONFIG="${WEIGHTS_ROOT}/gen_CUT/iddaw_ad/pspnet_r50_ratio0p50/configs/pspnet_r50-d8_4xb4-80k_cityscapes-512x1024.py"
CKPT="${WEIGHTS_ROOT}/gen_CUT/iddaw_ad/pspnet_r50_ratio0p50/iter_80000.pth"
if [ -f "$CKPT" ]; then
    bsub -J "test2_CUT_iddaw_psp" \
        -q BatchGPU -n 10 -R "span[hosts=1]" -R "rusage[mem=8000]" -gpu "num=1:gmem=20G" -W 04:00 \
        -o "${LOG_DIR}/test2_CUT_iddaw_psp_%J.out" -e "${LOG_DIR}/test2_CUT_iddaw_psp_%J.err" \
        "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python fine_grained_test.py --config ${CONFIG} --checkpoint ${CKPT} --dataset IDD-AW --output-dir results/gen_CUT/iddaw_ad/pspnet_r50_ratio0p50"
    ((submitted++))
else
    echo "  SKIP: Checkpoint not found: $CKPT"
fi

# 9. gen_CUT/iddaw_ad/SegFormer
echo "9. Testing gen_CUT/iddaw_ad/SegFormer..."
CONFIG="${WEIGHTS_ROOT}/gen_CUT/iddaw_ad/segformer_mit-b5_ratio0p50/configs/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py"
CKPT="${WEIGHTS_ROOT}/gen_CUT/iddaw_ad/segformer_mit-b5_ratio0p50/iter_80000.pth"
if [ -f "$CKPT" ]; then
    bsub -J "test2_CUT_iddaw_segf" \
        -q BatchGPU -n 10 -R "span[hosts=1]" -R "rusage[mem=8000]" -gpu "num=1:gmem=20G" -W 04:00 \
        -o "${LOG_DIR}/test2_CUT_iddaw_segf_%J.out" -e "${LOG_DIR}/test2_CUT_iddaw_segf_%J.err" \
        "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python fine_grained_test.py --config ${CONFIG} --checkpoint ${CKPT} --dataset IDD-AW --output-dir results/gen_CUT/iddaw_ad/segformer_mit-b5_ratio0p50"
    ((submitted++))
else
    echo "  SKIP: Checkpoint not found: $CKPT"
fi

# 10. gen_IP2P/iddaw_ad/DeepLabV3+
echo "10. Testing gen_IP2P/iddaw_ad/DeepLabV3+..."
CONFIG="${WEIGHTS_ROOT}/gen_IP2P/iddaw_ad/deeplabv3plus_r50_ratio0p50/configs/deeplabv3plus_r50-d8_4xb4-80k_cityscapes-512x1024.py"
CKPT="${WEIGHTS_ROOT}/gen_IP2P/iddaw_ad/deeplabv3plus_r50_ratio0p50/iter_80000.pth"
if [ -f "$CKPT" ]; then
    bsub -J "test2_IP2P_iddaw_dlv3" \
        -q BatchGPU -n 10 -R "span[hosts=1]" -R "rusage[mem=8000]" -gpu "num=1:gmem=20G" -W 04:00 \
        -o "${LOG_DIR}/test2_IP2P_iddaw_dlv3_%J.out" -e "${LOG_DIR}/test2_IP2P_iddaw_dlv3_%J.err" \
        "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python fine_grained_test.py --config ${CONFIG} --checkpoint ${CKPT} --dataset IDD-AW --output-dir results/gen_IP2P/iddaw_ad/deeplabv3plus_r50_ratio0p50"
    ((submitted++))
else
    echo "  SKIP: Checkpoint not found: $CKPT"
fi

# 11. gen_IP2P/iddaw_ad/PSPNet
echo "11. Testing gen_IP2P/iddaw_ad/PSPNet..."
CONFIG="${WEIGHTS_ROOT}/gen_IP2P/iddaw_ad/pspnet_r50_ratio0p50/configs/pspnet_r50-d8_4xb4-80k_cityscapes-512x1024.py"
CKPT="${WEIGHTS_ROOT}/gen_IP2P/iddaw_ad/pspnet_r50_ratio0p50/iter_80000.pth"
if [ -f "$CKPT" ]; then
    bsub -J "test2_IP2P_iddaw_psp" \
        -q BatchGPU -n 10 -R "span[hosts=1]" -R "rusage[mem=8000]" -gpu "num=1:gmem=20G" -W 04:00 \
        -o "${LOG_DIR}/test2_IP2P_iddaw_psp_%J.out" -e "${LOG_DIR}/test2_IP2P_iddaw_psp_%J.err" \
        "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python fine_grained_test.py --config ${CONFIG} --checkpoint ${CKPT} --dataset IDD-AW --output-dir results/gen_IP2P/iddaw_ad/pspnet_r50_ratio0p50"
    ((submitted++))
else
    echo "  SKIP: Checkpoint not found: $CKPT"
fi

echo ""
echo "=== Summary ==="
echo "Submitted: $submitted test jobs"
echo ""
echo "Monitor progress with: bjobs -w | grep test2_"
