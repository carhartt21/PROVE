#!/bin/bash
# Submit test jobs for IDD-AW with dependencies on training jobs
# Generated: 2026-01-14

PROJECT_ROOT="/home/mima2416/repositories/PROVE"
WEIGHTS_ROOT="/scratch/aaa_exchange/AWARE/WEIGHTS"
LOG_DIR="${PROJECT_ROOT}/logs"

submitted=0

echo "=== Submitting IDD-AW Test Jobs with Dependencies ==="
echo ""

# 1. baseline/idd-aw/DeepLabV3+ - depends on rtfix_baseline_iddaw_dlv3
echo "1. Testing baseline/idd-aw/DeepLabV3+..."
bsub -J "test_rtfix_baseline_iddaw_dlv3" \
    -w "done(rtfix_baseline_iddaw_dlv3)" \
    -q BatchGPU -n 10 -R "span[hosts=1]" -R "rusage[mem=8000]" -gpu "num=1:gmem=20G" -W 04:00 \
    -o "${LOG_DIR}/test_rtfix_baseline_iddaw_dlv3_%J.out" -e "${LOG_DIR}/test_rtfix_baseline_iddaw_dlv3_%J.err" \
    "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python fine_grained_test.py --config ${WEIGHTS_ROOT}/baseline/idd-aw_cd/deeplabv3plus_r50/deeplabv3plus_r50-d8_4xb4-80k_cityscapes-512x1024.py --checkpoint ${WEIGHTS_ROOT}/baseline/idd-aw_cd/deeplabv3plus_r50/iter_80000.pth --dataset IDD-AW --output-dir results/baseline/idd-aw_cd/deeplabv3plus_r50"
((submitted++))

# 2. baseline/idd-aw/SegFormer - depends on rtfix_baseline_iddaw_segf
echo "2. Testing baseline/idd-aw/SegFormer..."
bsub -J "test_rtfix_baseline_iddaw_segf" \
    -w "done(rtfix_baseline_iddaw_segf)" \
    -q BatchGPU -n 10 -R "span[hosts=1]" -R "rusage[mem=8000]" -gpu "num=1:gmem=20G" -W 04:00 \
    -o "${LOG_DIR}/test_rtfix_baseline_iddaw_segf_%J.out" -e "${LOG_DIR}/test_rtfix_baseline_iddaw_segf_%J.err" \
    "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python fine_grained_test.py --config ${WEIGHTS_ROOT}/baseline/idd-aw_cd/segformer_mit-b5/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py --checkpoint ${WEIGHTS_ROOT}/baseline/idd-aw_cd/segformer_mit-b5/iter_80000.pth --dataset IDD-AW --output-dir results/baseline/idd-aw_cd/segformer_mit-b5"
((submitted++))

# 3. gen_Attribute_Hallucination/idd-aw/DeepLabV3+
echo "3. Testing gen_Attribute_Hallucination/idd-aw/DeepLabV3+..."
bsub -J "test_rtfix_AttrHall_iddaw_dlv3" \
    -w "done(rtfix_AttrHall_iddaw_dlv3)" \
    -q BatchGPU -n 10 -R "span[hosts=1]" -R "rusage[mem=8000]" -gpu "num=1:gmem=20G" -W 04:00 \
    -o "${LOG_DIR}/test_rtfix_AttrHall_iddaw_dlv3_%J.out" -e "${LOG_DIR}/test_rtfix_AttrHall_iddaw_dlv3_%J.err" \
    "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python fine_grained_test.py --config ${WEIGHTS_ROOT}/gen_Attribute_Hallucination/idd-aw_cd/deeplabv3plus_r50_ratio0p50/deeplabv3plus_r50-d8_4xb4-80k_cityscapes-512x1024.py --checkpoint ${WEIGHTS_ROOT}/gen_Attribute_Hallucination/idd-aw_cd/deeplabv3plus_r50_ratio0p50/iter_80000.pth --dataset IDD-AW --output-dir results/gen_Attribute_Hallucination/idd-aw_cd/deeplabv3plus_r50_ratio0p50"
((submitted++))

# 4. gen_Attribute_Hallucination/idd-aw/PSPNet
echo "4. Testing gen_Attribute_Hallucination/idd-aw/PSPNet..."
bsub -J "test_rtfix_AttrHall_iddaw_psp" \
    -w "done(rtfix_AttrHall_iddaw_psp)" \
    -q BatchGPU -n 10 -R "span[hosts=1]" -R "rusage[mem=8000]" -gpu "num=1:gmem=20G" -W 04:00 \
    -o "${LOG_DIR}/test_rtfix_AttrHall_iddaw_psp_%J.out" -e "${LOG_DIR}/test_rtfix_AttrHall_iddaw_psp_%J.err" \
    "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python fine_grained_test.py --config ${WEIGHTS_ROOT}/gen_Attribute_Hallucination/idd-aw_cd/pspnet_r50_ratio0p50/pspnet_r50-d8_4xb4-80k_cityscapes-512x1024.py --checkpoint ${WEIGHTS_ROOT}/gen_Attribute_Hallucination/idd-aw_cd/pspnet_r50_ratio0p50/iter_80000.pth --dataset IDD-AW --output-dir results/gen_Attribute_Hallucination/idd-aw_cd/pspnet_r50_ratio0p50"
((submitted++))

# 5. gen_CNetSeg/idd-aw/DeepLabV3+
echo "5. Testing gen_CNetSeg/idd-aw/DeepLabV3+..."
bsub -J "test_rtfix_CNetSeg_iddaw_dlv3" \
    -w "done(rtfix_CNetSeg_iddaw_dlv3)" \
    -q BatchGPU -n 10 -R "span[hosts=1]" -R "rusage[mem=8000]" -gpu "num=1:gmem=20G" -W 04:00 \
    -o "${LOG_DIR}/test_rtfix_CNetSeg_iddaw_dlv3_%J.out" -e "${LOG_DIR}/test_rtfix_CNetSeg_iddaw_dlv3_%J.err" \
    "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python fine_grained_test.py --config ${WEIGHTS_ROOT}/gen_CNetSeg/idd-aw_cd/deeplabv3plus_r50_ratio0p50/deeplabv3plus_r50-d8_4xb4-80k_cityscapes-512x1024.py --checkpoint ${WEIGHTS_ROOT}/gen_CNetSeg/idd-aw_cd/deeplabv3plus_r50_ratio0p50/iter_80000.pth --dataset IDD-AW --output-dir results/gen_CNetSeg/idd-aw_cd/deeplabv3plus_r50_ratio0p50"
((submitted++))

# 6. gen_CNetSeg/idd-aw/PSPNet
echo "6. Testing gen_CNetSeg/idd-aw/PSPNet..."
bsub -J "test_rtfix_CNetSeg_iddaw_psp" \
    -w "done(rtfix_CNetSeg_iddaw_psp)" \
    -q BatchGPU -n 10 -R "span[hosts=1]" -R "rusage[mem=8000]" -gpu "num=1:gmem=20G" -W 04:00 \
    -o "${LOG_DIR}/test_rtfix_CNetSeg_iddaw_psp_%J.out" -e "${LOG_DIR}/test_rtfix_CNetSeg_iddaw_psp_%J.err" \
    "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python fine_grained_test.py --config ${WEIGHTS_ROOT}/gen_CNetSeg/idd-aw_cd/pspnet_r50_ratio0p50/pspnet_r50-d8_4xb4-80k_cityscapes-512x1024.py --checkpoint ${WEIGHTS_ROOT}/gen_CNetSeg/idd-aw_cd/pspnet_r50_ratio0p50/iter_80000.pth --dataset IDD-AW --output-dir results/gen_CNetSeg/idd-aw_cd/pspnet_r50_ratio0p50"
((submitted++))

# 7. gen_CUT/idd-aw/DeepLabV3+
echo "7. Testing gen_CUT/idd-aw/DeepLabV3+..."
bsub -J "test_rtfix_CUT_iddaw_dlv3" \
    -w "done(rtfix_CUT_iddaw_dlv3)" \
    -q BatchGPU -n 10 -R "span[hosts=1]" -R "rusage[mem=8000]" -gpu "num=1:gmem=20G" -W 04:00 \
    -o "${LOG_DIR}/test_rtfix_CUT_iddaw_dlv3_%J.out" -e "${LOG_DIR}/test_rtfix_CUT_iddaw_dlv3_%J.err" \
    "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python fine_grained_test.py --config ${WEIGHTS_ROOT}/gen_CUT/idd-aw_cd/deeplabv3plus_r50_ratio0p50/deeplabv3plus_r50-d8_4xb4-80k_cityscapes-512x1024.py --checkpoint ${WEIGHTS_ROOT}/gen_CUT/idd-aw_cd/deeplabv3plus_r50_ratio0p50/iter_80000.pth --dataset IDD-AW --output-dir results/gen_CUT/idd-aw_cd/deeplabv3plus_r50_ratio0p50"
((submitted++))

# 8. gen_CUT/idd-aw/PSPNet
echo "8. Testing gen_CUT/idd-aw/PSPNet..."
bsub -J "test_rtfix_CUT_iddaw_psp" \
    -w "done(rtfix_CUT_iddaw_psp)" \
    -q BatchGPU -n 10 -R "span[hosts=1]" -R "rusage[mem=8000]" -gpu "num=1:gmem=20G" -W 04:00 \
    -o "${LOG_DIR}/test_rtfix_CUT_iddaw_psp_%J.out" -e "${LOG_DIR}/test_rtfix_CUT_iddaw_psp_%J.err" \
    "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python fine_grained_test.py --config ${WEIGHTS_ROOT}/gen_CUT/idd-aw_cd/pspnet_r50_ratio0p50/pspnet_r50-d8_4xb4-80k_cityscapes-512x1024.py --checkpoint ${WEIGHTS_ROOT}/gen_CUT/idd-aw_cd/pspnet_r50_ratio0p50/iter_80000.pth --dataset IDD-AW --output-dir results/gen_CUT/idd-aw_cd/pspnet_r50_ratio0p50"
((submitted++))

# 9. gen_CUT/idd-aw/SegFormer
echo "9. Testing gen_CUT/idd-aw/SegFormer..."
bsub -J "test_rtfix_CUT_iddaw_segf" \
    -w "done(rtfix_CUT_iddaw_segf)" \
    -q BatchGPU -n 10 -R "span[hosts=1]" -R "rusage[mem=8000]" -gpu "num=1:gmem=20G" -W 04:00 \
    -o "${LOG_DIR}/test_rtfix_CUT_iddaw_segf_%J.out" -e "${LOG_DIR}/test_rtfix_CUT_iddaw_segf_%J.err" \
    "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python fine_grained_test.py --config ${WEIGHTS_ROOT}/gen_CUT/idd-aw_cd/segformer_mit-b5_ratio0p50/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py --checkpoint ${WEIGHTS_ROOT}/gen_CUT/idd-aw_cd/segformer_mit-b5_ratio0p50/iter_80000.pth --dataset IDD-AW --output-dir results/gen_CUT/idd-aw_cd/segformer_mit-b5_ratio0p50"
((submitted++))

# 10. gen_IP2P/idd-aw/DeepLabV3+
echo "10. Testing gen_IP2P/idd-aw/DeepLabV3+..."
bsub -J "test_rtfix_IP2P_iddaw_dlv3" \
    -w "done(rtfix_IP2P_iddaw_dlv3)" \
    -q BatchGPU -n 10 -R "span[hosts=1]" -R "rusage[mem=8000]" -gpu "num=1:gmem=20G" -W 04:00 \
    -o "${LOG_DIR}/test_rtfix_IP2P_iddaw_dlv3_%J.out" -e "${LOG_DIR}/test_rtfix_IP2P_iddaw_dlv3_%J.err" \
    "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python fine_grained_test.py --config ${WEIGHTS_ROOT}/gen_IP2P/idd-aw_cd/deeplabv3plus_r50_ratio0p50/deeplabv3plus_r50-d8_4xb4-80k_cityscapes-512x1024.py --checkpoint ${WEIGHTS_ROOT}/gen_IP2P/idd-aw_cd/deeplabv3plus_r50_ratio0p50/iter_80000.pth --dataset IDD-AW --output-dir results/gen_IP2P/idd-aw_cd/deeplabv3plus_r50_ratio0p50"
((submitted++))

# 11. gen_IP2P/idd-aw/PSPNet
echo "11. Testing gen_IP2P/idd-aw/PSPNet..."
bsub -J "test_rtfix_IP2P_iddaw_psp" \
    -w "done(rtfix_IP2P_iddaw_psp)" \
    -q BatchGPU -n 10 -R "span[hosts=1]" -R "rusage[mem=8000]" -gpu "num=1:gmem=20G" -W 04:00 \
    -o "${LOG_DIR}/test_rtfix_IP2P_iddaw_psp_%J.out" -e "${LOG_DIR}/test_rtfix_IP2P_iddaw_psp_%J.err" \
    "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python fine_grained_test.py --config ${WEIGHTS_ROOT}/gen_IP2P/idd-aw_cd/pspnet_r50_ratio0p50/pspnet_r50-d8_4xb4-80k_cityscapes-512x1024.py --checkpoint ${WEIGHTS_ROOT}/gen_IP2P/idd-aw_cd/pspnet_r50_ratio0p50/iter_80000.pth --dataset IDD-AW --output-dir results/gen_IP2P/idd-aw_cd/pspnet_r50_ratio0p50"
((submitted++))

echo ""
echo "=== Summary ==="
echo "Submitted: $submitted test jobs with dependencies"
echo ""
echo "Test jobs will wait for corresponding training jobs to complete."
echo "Check status with: bjobs -w | grep rtfix"
