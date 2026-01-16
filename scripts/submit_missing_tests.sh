#!/bin/bash
# Submit test jobs for missing MapillaryVistas configurations and buggy IDD-AW configurations
# Based on TESTING_COVERAGE.md analysis (2026-01-14)

WEIGHTS_ROOT="/scratch/aaa_exchange/AWARE/WEIGHTS"
LOG_DIR="/home/mima2416/repositories/PROVE/logs"
SCRIPT_DIR="/home/mima2416/repositories/PROVE"

submitted=0

echo "=== Submitting Missing MapillaryVistas Tests ==="
echo ""

# 1. gen_Attribute_Hallucination/mapillaryvistas/PSPNet
echo "1. Testing gen_Attribute_Hallucination/mapillaryvistas/PSPNet..."
CHECKPOINT="${WEIGHTS_ROOT}/gen_Attribute_Hallucination/mapillaryvistas_cd/pspnet_r50_ratio0p50/iter_80000.pth"
if [ -f "$CHECKPOINT" ]; then
    bsub -J "test_AttrHall_mapi_psp" \
        -q BatchGPU \
        -n 10 \
        -R "rusage[mem=16000]" \
        -gpu "num=1:gmem=16G" \
        -W 02:00 \
        -o "${LOG_DIR}/test_AttrHall_mapi_psp_%J.out" \
        -e "${LOG_DIR}/test_AttrHall_mapi_psp_%J.err" \
        "source ~/.bashrc && conda activate prove && cd ${SCRIPT_DIR} && python fine_grained_test.py --checkpoint ${CHECKPOINT} --dataset MapillaryVistas --output-dir results/gen_Attribute_Hallucination/mapillaryvistas_cd/pspnet_r50_ratio0p50"
    ((submitted++))
else
    echo "  WARNING: Checkpoint not found: $CHECKPOINT"
fi

# 2. gen_IP2P/mapillaryvistas/PSPNet
echo "2. Testing gen_IP2P/mapillaryvistas/PSPNet..."
CHECKPOINT="${WEIGHTS_ROOT}/gen_IP2P/mapillaryvistas_cd/pspnet_r50_ratio0p50/iter_80000.pth"
if [ -f "$CHECKPOINT" ]; then
    bsub -J "test_IP2P_mapi_psp" \
        -q BatchGPU \
        -n 10 \
        -R "rusage[mem=16000]" \
        -gpu "num=1:gmem=16G" \
        -W 02:00 \
        -o "${LOG_DIR}/test_IP2P_mapi_psp_%J.out" \
        -e "${LOG_DIR}/test_IP2P_mapi_psp_%J.err" \
        "source ~/.bashrc && conda activate prove && cd ${SCRIPT_DIR} && python fine_grained_test.py --checkpoint ${CHECKPOINT} --dataset MapillaryVistas --output-dir results/gen_IP2P/mapillaryvistas_cd/pspnet_r50_ratio0p50"
    ((submitted++))
else
    echo "  WARNING: Checkpoint not found: $CHECKPOINT"
fi

# 3. gen_Weather_Effect_Generator/mapillaryvistas/PSPNet
echo "3. Testing gen_Weather_Effect_Generator/mapillaryvistas/PSPNet..."
CHECKPOINT="${WEIGHTS_ROOT}/gen_Weather_Effect_Generator/mapillaryvistas_cd/pspnet_r50_ratio0p50/iter_80000.pth"
if [ -f "$CHECKPOINT" ]; then
    bsub -J "test_WeatherEG_mapi_psp" \
        -q BatchGPU \
        -n 10 \
        -R "rusage[mem=16000]" \
        -gpu "num=1:gmem=16G" \
        -W 02:00 \
        -o "${LOG_DIR}/test_WeatherEG_mapi_psp_%J.out" \
        -e "${LOG_DIR}/test_WeatherEG_mapi_psp_%J.err" \
        "source ~/.bashrc && conda activate prove && cd ${SCRIPT_DIR} && python fine_grained_test.py --checkpoint ${CHECKPOINT} --dataset MapillaryVistas --output-dir results/gen_Weather_Effect_Generator/mapillaryvistas_cd/pspnet_r50_ratio0p50"
    ((submitted++))
else
    echo "  WARNING: Checkpoint not found: $CHECKPOINT"
fi

# 4. gen_albumentations_weather/mapillaryvistas/DeepLabV3+
echo "4. Testing gen_albumentations_weather/mapillaryvistas/DeepLabV3+..."
CHECKPOINT="${WEIGHTS_ROOT}/gen_albumentations_weather/mapillaryvistas_cd/deeplabv3plus_r50_ratio0p50/iter_80000.pth"
if [ -f "$CHECKPOINT" ]; then
    bsub -J "test_albweather_mapi_dlv3" \
        -q BatchGPU \
        -n 10 \
        -R "rusage[mem=16000]" \
        -gpu "num=1:gmem=16G" \
        -W 02:00 \
        -o "${LOG_DIR}/test_albweather_mapi_dlv3_%J.out" \
        -e "${LOG_DIR}/test_albweather_mapi_dlv3_%J.err" \
        "source ~/.bashrc && conda activate prove && cd ${SCRIPT_DIR} && python fine_grained_test.py --checkpoint ${CHECKPOINT} --dataset MapillaryVistas --output-dir results/gen_albumentations_weather/mapillaryvistas_cd/deeplabv3plus_r50_ratio0p50"
    ((submitted++))
else
    echo "  WARNING: Checkpoint not found: $CHECKPOINT"
fi

# 5. gen_augmenters/mapillaryvistas/DeepLabV3+
echo "5. Testing gen_augmenters/mapillaryvistas/DeepLabV3+..."
CHECKPOINT="${WEIGHTS_ROOT}/gen_augmenters/mapillaryvistas_cd/deeplabv3plus_r50_ratio0p50/iter_80000.pth"
if [ -f "$CHECKPOINT" ]; then
    bsub -J "test_augmenters_mapi_dlv3" \
        -q BatchGPU \
        -n 10 \
        -R "rusage[mem=16000]" \
        -gpu "num=1:gmem=16G" \
        -W 02:00 \
        -o "${LOG_DIR}/test_augmenters_mapi_dlv3_%J.out" \
        -e "${LOG_DIR}/test_augmenters_mapi_dlv3_%J.err" \
        "source ~/.bashrc && conda activate prove && cd ${SCRIPT_DIR} && python fine_grained_test.py --checkpoint ${CHECKPOINT} --dataset MapillaryVistas --output-dir results/gen_augmenters/mapillaryvistas_cd/deeplabv3plus_r50_ratio0p50"
    ((submitted++))
else
    echo "  WARNING: Checkpoint not found: $CHECKPOINT"
fi

# 6. gen_automold/mapillaryvistas/DeepLabV3+
echo "6. Testing gen_automold/mapillaryvistas/DeepLabV3+..."
CHECKPOINT="${WEIGHTS_ROOT}/gen_automold/mapillaryvistas_cd/deeplabv3plus_r50_ratio0p50/iter_80000.pth"
if [ -f "$CHECKPOINT" ]; then
    bsub -J "test_automold_mapi_dlv3" \
        -q BatchGPU \
        -n 10 \
        -R "rusage[mem=16000]" \
        -gpu "num=1:gmem=16G" \
        -W 02:00 \
        -o "${LOG_DIR}/test_automold_mapi_dlv3_%J.out" \
        -e "${LOG_DIR}/test_automold_mapi_dlv3_%J.err" \
        "source ~/.bashrc && conda activate prove && cd ${SCRIPT_DIR} && python fine_grained_test.py --checkpoint ${CHECKPOINT} --dataset MapillaryVistas --output-dir results/gen_automold/mapillaryvistas_cd/deeplabv3plus_r50_ratio0p50"
    ((submitted++))
else
    echo "  WARNING: Checkpoint not found: $CHECKPOINT"
fi

# 7. gen_cycleGAN/mapillaryvistas/PSPNet
echo "7. Testing gen_cycleGAN/mapillaryvistas/PSPNet..."
CHECKPOINT="${WEIGHTS_ROOT}/gen_cycleGAN/mapillaryvistas_cd/pspnet_r50_ratio0p50/iter_80000.pth"
if [ -f "$CHECKPOINT" ]; then
    bsub -J "test_cycleGAN_mapi_psp" \
        -q BatchGPU \
        -n 10 \
        -R "rusage[mem=16000]" \
        -gpu "num=1:gmem=16G" \
        -W 02:00 \
        -o "${LOG_DIR}/test_cycleGAN_mapi_psp_%J.out" \
        -e "${LOG_DIR}/test_cycleGAN_mapi_psp_%J.err" \
        "source ~/.bashrc && conda activate prove && cd ${SCRIPT_DIR} && python fine_grained_test.py --checkpoint ${CHECKPOINT} --dataset MapillaryVistas --output-dir results/gen_cycleGAN/mapillaryvistas_cd/pspnet_r50_ratio0p50"
    ((submitted++))
else
    echo "  WARNING: Checkpoint not found: $CHECKPOINT"
fi

# 8. gen_flux_kontext/mapillaryvistas/PSPNet
echo "8. Testing gen_flux_kontext/mapillaryvistas/PSPNet..."
CHECKPOINT="${WEIGHTS_ROOT}/gen_flux_kontext/mapillaryvistas_cd/pspnet_r50_ratio0p50/iter_80000.pth"
if [ -f "$CHECKPOINT" ]; then
    bsub -J "test_fluxkontext_mapi_psp" \
        -q BatchGPU \
        -n 10 \
        -R "rusage[mem=16000]" \
        -gpu "num=1:gmem=16G" \
        -W 02:00 \
        -o "${LOG_DIR}/test_fluxkontext_mapi_psp_%J.out" \
        -e "${LOG_DIR}/test_fluxkontext_mapi_psp_%J.err" \
        "source ~/.bashrc && conda activate prove && cd ${SCRIPT_DIR} && python fine_grained_test.py --checkpoint ${CHECKPOINT} --dataset MapillaryVistas --output-dir results/gen_flux_kontext/mapillaryvistas_cd/pspnet_r50_ratio0p50"
    ((submitted++))
else
    echo "  WARNING: Checkpoint not found: $CHECKPOINT"
fi

echo ""
echo "=== Submitting Buggy IDD-AW Retests ==="
echo ""

# Buggy IDD-AW configs (mIoU < 5%)
# 9. baseline/idd-aw/DeepLabV3+
echo "9. Retesting baseline/idd-aw/DeepLabV3+..."
CHECKPOINT="${WEIGHTS_ROOT}/baseline/idd-aw_cd/deeplabv3plus_r50/iter_80000.pth"
if [ -f "$CHECKPOINT" ]; then
    bsub -J "test_baseline_iddaw_dlv3" \
        -q BatchGPU \
        -n 10 \
        -R "rusage[mem=16000]" \
        -gpu "num=1:gmem=16G" \
        -W 02:00 \
        -o "${LOG_DIR}/test_baseline_iddaw_dlv3_%J.out" \
        -e "${LOG_DIR}/test_baseline_iddaw_dlv3_%J.err" \
        "source ~/.bashrc && conda activate prove && cd ${SCRIPT_DIR} && python fine_grained_test.py --checkpoint ${CHECKPOINT} --dataset IDD-AW --output-dir results/baseline/idd-aw_cd/deeplabv3plus_r50"
    ((submitted++))
else
    echo "  WARNING: Checkpoint not found: $CHECKPOINT"
fi

# 10. baseline/idd-aw/SegFormer
echo "10. Retesting baseline/idd-aw/SegFormer..."
CHECKPOINT="${WEIGHTS_ROOT}/baseline/idd-aw_cd/segformer_mit-b5/iter_80000.pth"
if [ -f "$CHECKPOINT" ]; then
    bsub -J "test_baseline_iddaw_segf" \
        -q BatchGPU \
        -n 10 \
        -R "rusage[mem=16000]" \
        -gpu "num=1:gmem=16G" \
        -W 02:00 \
        -o "${LOG_DIR}/test_baseline_iddaw_segf_%J.out" \
        -e "${LOG_DIR}/test_baseline_iddaw_segf_%J.err" \
        "source ~/.bashrc && conda activate prove && cd ${SCRIPT_DIR} && python fine_grained_test.py --checkpoint ${CHECKPOINT} --dataset IDD-AW --output-dir results/baseline/idd-aw_cd/segformer_mit-b5"
    ((submitted++))
else
    echo "  WARNING: Checkpoint not found: $CHECKPOINT"
fi

# 11. gen_Attribute_Hallucination/idd-aw/DeepLabV3+
echo "11. Retesting gen_Attribute_Hallucination/idd-aw/DeepLabV3+..."
CHECKPOINT="${WEIGHTS_ROOT}/gen_Attribute_Hallucination/idd-aw_cd/deeplabv3plus_r50_ratio0p50/iter_80000.pth"
if [ -f "$CHECKPOINT" ]; then
    bsub -J "test_AttrHall_iddaw_dlv3" \
        -q BatchGPU \
        -n 10 \
        -R "rusage[mem=16000]" \
        -gpu "num=1:gmem=16G" \
        -W 02:00 \
        -o "${LOG_DIR}/test_AttrHall_iddaw_dlv3_%J.out" \
        -e "${LOG_DIR}/test_AttrHall_iddaw_dlv3_%J.err" \
        "source ~/.bashrc && conda activate prove && cd ${SCRIPT_DIR} && python fine_grained_test.py --checkpoint ${CHECKPOINT} --dataset IDD-AW --output-dir results/gen_Attribute_Hallucination/idd-aw_cd/deeplabv3plus_r50_ratio0p50"
    ((submitted++))
else
    echo "  WARNING: Checkpoint not found: $CHECKPOINT"
fi

# 12. gen_Attribute_Hallucination/idd-aw/PSPNet
echo "12. Retesting gen_Attribute_Hallucination/idd-aw/PSPNet..."
CHECKPOINT="${WEIGHTS_ROOT}/gen_Attribute_Hallucination/idd-aw_cd/pspnet_r50_ratio0p50/iter_80000.pth"
if [ -f "$CHECKPOINT" ]; then
    bsub -J "test_AttrHall_iddaw_psp" \
        -q BatchGPU \
        -n 10 \
        -R "rusage[mem=16000]" \
        -gpu "num=1:gmem=16G" \
        -W 02:00 \
        -o "${LOG_DIR}/test_AttrHall_iddaw_psp_%J.out" \
        -e "${LOG_DIR}/test_AttrHall_iddaw_psp_%J.err" \
        "source ~/.bashrc && conda activate prove && cd ${SCRIPT_DIR} && python fine_grained_test.py --checkpoint ${CHECKPOINT} --dataset IDD-AW --output-dir results/gen_Attribute_Hallucination/idd-aw_cd/pspnet_r50_ratio0p50"
    ((submitted++))
else
    echo "  WARNING: Checkpoint not found: $CHECKPOINT"
fi

# 13. gen_CNetSeg/idd-aw/DeepLabV3+
echo "13. Retesting gen_CNetSeg/idd-aw/DeepLabV3+..."
CHECKPOINT="${WEIGHTS_ROOT}/gen_CNetSeg/idd-aw_cd/deeplabv3plus_r50_ratio0p50/iter_80000.pth"
if [ -f "$CHECKPOINT" ]; then
    bsub -J "test_CNetSeg_iddaw_dlv3" \
        -q BatchGPU \
        -R "rusage[mem=16000]" \
        -gpu "num=1:gmem=16G" \
        -W 02:00 \
        -o "${LOG_DIR}/test_CNetSeg_iddaw_dlv3_%J.out" \
        -e "${LOG_DIR}/test_CNetSeg_iddaw_dlv3_%J.err" \
        "source ~/.bashrc && conda activate prove && cd ${SCRIPT_DIR} && python fine_grained_test.py --checkpoint ${CHECKPOINT} --dataset IDD-AW --output-dir results/gen_CNetSeg/idd-aw_cd/deeplabv3plus_r50_ratio0p50"
    ((submitted++))
else
    echo "  WARNING: Checkpoint not found: $CHECKPOINT"
fi

# 14. gen_CNetSeg/idd-aw/PSPNet
echo "14. Retesting gen_CNetSeg/idd-aw/PSPNet..."
CHECKPOINT="${WEIGHTS_ROOT}/gen_CNetSeg/idd-aw_cd/pspnet_r50_ratio0p50/iter_80000.pth"
if [ -f "$CHECKPOINT" ]; then
    bsub -J "test_CNetSeg_iddaw_psp" \
        -q BatchGPU \
        -R "rusage[mem=16000]" \
        -gpu "num=1:gmem=16G" \
        -W 02:00 \
        -o "${LOG_DIR}/test_CNetSeg_iddaw_psp_%J.out" \
        -e "${LOG_DIR}/test_CNetSeg_iddaw_psp_%J.err" \
        "source ~/.bashrc && conda activate prove && cd ${SCRIPT_DIR} && python fine_grained_test.py --checkpoint ${CHECKPOINT} --dataset IDD-AW --output-dir results/gen_CNetSeg/idd-aw_cd/pspnet_r50_ratio0p50"
    ((submitted++))
else
    echo "  WARNING: Checkpoint not found: $CHECKPOINT"
fi

# 15. gen_CUT/idd-aw/DeepLabV3+
echo "15. Retesting gen_CUT/idd-aw/DeepLabV3+..."
CHECKPOINT="${WEIGHTS_ROOT}/gen_CUT/idd-aw_cd/deeplabv3plus_r50_ratio0p50/iter_80000.pth"
if [ -f "$CHECKPOINT" ]; then
    bsub -J "test_CUT_iddaw_dlv3" \
        -q BatchGPU \
        -R "rusage[mem=16000]" \
        -gpu "num=1:gmem=16G" \
        -W 02:00 \
        -o "${LOG_DIR}/test_CUT_iddaw_dlv3_%J.out" \
        -e "${LOG_DIR}/test_CUT_iddaw_dlv3_%J.err" \
        "source ~/.bashrc && conda activate prove && cd ${SCRIPT_DIR} && python fine_grained_test.py --checkpoint ${CHECKPOINT} --dataset IDD-AW --output-dir results/gen_CUT/idd-aw_cd/deeplabv3plus_r50_ratio0p50"
    ((submitted++))
else
    echo "  WARNING: Checkpoint not found: $CHECKPOINT"
fi

# 16. gen_CUT/idd-aw/PSPNet
echo "16. Retesting gen_CUT/idd-aw/PSPNet..."
CHECKPOINT="${WEIGHTS_ROOT}/gen_CUT/idd-aw_cd/pspnet_r50_ratio0p50/iter_80000.pth"
if [ -f "$CHECKPOINT" ]; then
    bsub -J "test_CUT_iddaw_psp" \
        -q BatchGPU \
        -R "rusage[mem=16000]" \
        -gpu "num=1:gmem=16G" \
        -W 02:00 \
        -o "${LOG_DIR}/test_CUT_iddaw_psp_%J.out" \
        -e "${LOG_DIR}/test_CUT_iddaw_psp_%J.err" \
        "source ~/.bashrc && conda activate prove && cd ${SCRIPT_DIR} && python fine_grained_test.py --checkpoint ${CHECKPOINT} --dataset IDD-AW --output-dir results/gen_CUT/idd-aw_cd/pspnet_r50_ratio0p50"
    ((submitted++))
else
    echo "  WARNING: Checkpoint not found: $CHECKPOINT"
fi

# 17. gen_CUT/idd-aw/SegFormer
echo "17. Retesting gen_CUT/idd-aw/SegFormer..."
CHECKPOINT="${WEIGHTS_ROOT}/gen_CUT/idd-aw_cd/segformer_mit-b5_ratio0p50/iter_80000.pth"
if [ -f "$CHECKPOINT" ]; then
    bsub -J "test_CUT_iddaw_segf" \
        -q BatchGPU \
        -R "rusage[mem=16000]" \
        -gpu "num=1:gmem=16G" \
        -W 02:00 \
        -o "${LOG_DIR}/test_CUT_iddaw_segf_%J.out" \
        -e "${LOG_DIR}/test_CUT_iddaw_segf_%J.err" \
        "source ~/.bashrc && conda activate prove && cd ${SCRIPT_DIR} && python fine_grained_test.py --checkpoint ${CHECKPOINT} --dataset IDD-AW --output-dir results/gen_CUT/idd-aw_cd/segformer_mit-b5_ratio0p50"
    ((submitted++))
else
    echo "  WARNING: Checkpoint not found: $CHECKPOINT"
fi

# 18. gen_IP2P/idd-aw/DeepLabV3+
echo "18. Retesting gen_IP2P/idd-aw/DeepLabV3+..."
CHECKPOINT="${WEIGHTS_ROOT}/gen_IP2P/idd-aw_cd/deeplabv3plus_r50_ratio0p50/iter_80000.pth"
if [ -f "$CHECKPOINT" ]; then
    bsub -J "test_IP2P_iddaw_dlv3" \
        -q BatchGPU \
        -R "rusage[mem=16000]" \
        -gpu "num=1:gmem=16G" \
        -W 02:00 \
        -o "${LOG_DIR}/test_IP2P_iddaw_dlv3_%J.out" \
        -e "${LOG_DIR}/test_IP2P_iddaw_dlv3_%J.err" \
        "source ~/.bashrc && conda activate prove && cd ${SCRIPT_DIR} && python fine_grained_test.py --checkpoint ${CHECKPOINT} --dataset IDD-AW --output-dir results/gen_IP2P/idd-aw_cd/deeplabv3plus_r50_ratio0p50"
    ((submitted++))
else
    echo "  WARNING: Checkpoint not found: $CHECKPOINT"
fi

# 19. gen_IP2P/idd-aw/PSPNet
echo "19. Retesting gen_IP2P/idd-aw/PSPNet..."
CHECKPOINT="${WEIGHTS_ROOT}/gen_IP2P/idd-aw_cd/pspnet_r50_ratio0p50/iter_80000.pth"
if [ -f "$CHECKPOINT" ]; then
    bsub -J "test_IP2P_iddaw_psp" \
        -q BatchGPU \
        -R "rusage[mem=16000]" \
        -gpu "num=1:gmem=16G" \
        -W 02:00 \
        -o "${LOG_DIR}/test_IP2P_iddaw_psp_%J.out" \
        -e "${LOG_DIR}/test_IP2P_iddaw_psp_%J.err" \
        "source ~/.bashrc && conda activate prove && cd ${SCRIPT_DIR} && python fine_grained_test.py --checkpoint ${CHECKPOINT} --dataset IDD-AW --output-dir results/gen_IP2P/idd-aw_cd/pspnet_r50_ratio0p50"
    ((submitted++))
else
    echo "  WARNING: Checkpoint not found: $CHECKPOINT"
fi

echo ""
echo "=== Summary ==="
echo "Submitted: $submitted test jobs"
echo ""
echo "Test time estimates: ~30-60 minutes per job"
