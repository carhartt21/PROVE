#!/bin/bash
# Retrain IDD-AW models that were corrupted by incorrect CityscapesLabelIdToTrainId transform
# These models need to be retrained from scratch with the fixed config
# Generated: 2026-01-14

PROJECT_ROOT="/home/mima2416/repositories/PROVE"
WEIGHTS_ROOT="/scratch/aaa_exchange/AWARE/WEIGHTS"
LOG_DIR="${PROJECT_ROOT}/logs"

submitted=0

echo "=== Retraining Corrupted IDD-AW Models ==="
echo ""
echo "These models were trained before the fix (commit e268acf) on Jan 11."
echo "The CityscapesLabelIdToTrainId transform was incorrectly applied to IDD-AW data."
echo ""

# 1. baseline/idd-aw/DeepLabV3+
echo "1. Retraining baseline/idd-aw/DeepLabV3+..."
bsub -J "rtfix_baseline_iddaw_dlv3" \
    -q BatchGPU \
    -R "rusage[mem=16000]" \
    -gpu "num=1:gmem=16G" \
    -W 16:00 \
    -o "${LOG_DIR}/rtfix_baseline_iddaw_dlv3_%J.out" \
    -e "${LOG_DIR}/rtfix_baseline_iddaw_dlv3_%J.err" \
    "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python unified_training.py --dataset IDD-AW --model deeplabv3plus_r50 --strategy baseline --no-early-stop --max-iters 80000"
((submitted++))

# 2. baseline/idd-aw/SegFormer
echo "2. Retraining baseline/idd-aw/SegFormer..."
bsub -J "rtfix_baseline_iddaw_segf" \
    -q BatchGPU \
    -R "rusage[mem=20000]" \
    -gpu "num=1:gmem=24G" \
    -W 24:00 \
    -o "${LOG_DIR}/rtfix_baseline_iddaw_segf_%J.out" \
    -e "${LOG_DIR}/rtfix_baseline_iddaw_segf_%J.err" \
    "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python unified_training.py --dataset IDD-AW --model segformer_mit-b5 --strategy baseline --no-early-stop --max-iters 80000"
((submitted++))

# 3. gen_Attribute_Hallucination/idd-aw/DeepLabV3+
echo "3. Retraining gen_Attribute_Hallucination/idd-aw/DeepLabV3+..."
bsub -J "rtfix_AttrHall_iddaw_dlv3" \
    -q BatchGPU \
    -R "rusage[mem=16000]" \
    -gpu "num=1:gmem=16G" \
    -W 16:00 \
    -o "${LOG_DIR}/rtfix_AttrHall_iddaw_dlv3_%J.out" \
    -e "${LOG_DIR}/rtfix_AttrHall_iddaw_dlv3_%J.err" \
    "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python unified_training.py --dataset IDD-AW --model deeplabv3plus_r50 --strategy gen_Attribute_Hallucination --real-gen-ratio 0.5 --no-early-stop --max-iters 80000"
((submitted++))

# 4. gen_Attribute_Hallucination/idd-aw/PSPNet
echo "4. Retraining gen_Attribute_Hallucination/idd-aw/PSPNet..."
bsub -J "rtfix_AttrHall_iddaw_psp" \
    -q BatchGPU \
    -R "rusage[mem=16000]" \
    -gpu "num=1:gmem=16G" \
    -W 16:00 \
    -o "${LOG_DIR}/rtfix_AttrHall_iddaw_psp_%J.out" \
    -e "${LOG_DIR}/rtfix_AttrHall_iddaw_psp_%J.err" \
    "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python unified_training.py --dataset IDD-AW --model pspnet_r50 --strategy gen_Attribute_Hallucination --real-gen-ratio 0.5 --no-early-stop --max-iters 80000"
((submitted++))

# 5. gen_CNetSeg/idd-aw/DeepLabV3+
echo "5. Retraining gen_CNetSeg/idd-aw/DeepLabV3+..."
bsub -J "rtfix_CNetSeg_iddaw_dlv3" \
    -q BatchGPU \
    -R "rusage[mem=16000]" \
    -gpu "num=1:gmem=16G" \
    -W 16:00 \
    -o "${LOG_DIR}/rtfix_CNetSeg_iddaw_dlv3_%J.out" \
    -e "${LOG_DIR}/rtfix_CNetSeg_iddaw_dlv3_%J.err" \
    "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python unified_training.py --dataset IDD-AW --model deeplabv3plus_r50 --strategy gen_CNetSeg --real-gen-ratio 0.5 --no-early-stop --max-iters 80000"
((submitted++))

# 6. gen_CNetSeg/idd-aw/PSPNet
echo "6. Retraining gen_CNetSeg/idd-aw/PSPNet..."
bsub -J "rtfix_CNetSeg_iddaw_psp" \
    -q BatchGPU \
    -R "rusage[mem=16000]" \
    -gpu "num=1:gmem=16G" \
    -W 16:00 \
    -o "${LOG_DIR}/rtfix_CNetSeg_iddaw_psp_%J.out" \
    -e "${LOG_DIR}/rtfix_CNetSeg_iddaw_psp_%J.err" \
    "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python unified_training.py --dataset IDD-AW --model pspnet_r50 --strategy gen_CNetSeg --real-gen-ratio 0.5 --no-early-stop --max-iters 80000"
((submitted++))

# 7. gen_CUT/idd-aw/DeepLabV3+
echo "7. Retraining gen_CUT/idd-aw/DeepLabV3+..."
bsub -J "rtfix_CUT_iddaw_dlv3" \
    -q BatchGPU \
    -R "rusage[mem=16000]" \
    -gpu "num=1:gmem=16G" \
    -W 16:00 \
    -o "${LOG_DIR}/rtfix_CUT_iddaw_dlv3_%J.out" \
    -e "${LOG_DIR}/rtfix_CUT_iddaw_dlv3_%J.err" \
    "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python unified_training.py --dataset IDD-AW --model deeplabv3plus_r50 --strategy gen_CUT --real-gen-ratio 0.5 --no-early-stop --max-iters 80000"
((submitted++))

# 8. gen_CUT/idd-aw/PSPNet
echo "8. Retraining gen_CUT/idd-aw/PSPNet..."
bsub -J "rtfix_CUT_iddaw_psp" \
    -q BatchGPU \
    -R "rusage[mem=16000]" \
    -gpu "num=1:gmem=16G" \
    -W 16:00 \
    -o "${LOG_DIR}/rtfix_CUT_iddaw_psp_%J.out" \
    -e "${LOG_DIR}/rtfix_CUT_iddaw_psp_%J.err" \
    "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python unified_training.py --dataset IDD-AW --model pspnet_r50 --strategy gen_CUT --real-gen-ratio 0.5 --no-early-stop --max-iters 80000"
((submitted++))

# 9. gen_CUT/idd-aw/SegFormer
echo "9. Retraining gen_CUT/idd-aw/SegFormer..."
bsub -J "rtfix_CUT_iddaw_segf" \
    -q BatchGPU \
    -R "rusage[mem=20000]" \
    -gpu "num=1:gmem=24G" \
    -W 24:00 \
    -o "${LOG_DIR}/rtfix_CUT_iddaw_segf_%J.out" \
    -e "${LOG_DIR}/rtfix_CUT_iddaw_segf_%J.err" \
    "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python unified_training.py --dataset IDD-AW --model segformer_mit-b5 --strategy gen_CUT --real-gen-ratio 0.5 --no-early-stop --max-iters 80000"
((submitted++))

# 10. gen_IP2P/idd-aw/DeepLabV3+
echo "10. Retraining gen_IP2P/idd-aw/DeepLabV3+..."
bsub -J "rtfix_IP2P_iddaw_dlv3" \
    -q BatchGPU \
    -R "rusage[mem=16000]" \
    -gpu "num=1:gmem=16G" \
    -W 16:00 \
    -o "${LOG_DIR}/rtfix_IP2P_iddaw_dlv3_%J.out" \
    -e "${LOG_DIR}/rtfix_IP2P_iddaw_dlv3_%J.err" \
    "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python unified_training.py --dataset IDD-AW --model deeplabv3plus_r50 --strategy gen_IP2P --real-gen-ratio 0.5 --no-early-stop --max-iters 80000"
((submitted++))

# 11. gen_IP2P/idd-aw/PSPNet
echo "11. Retraining gen_IP2P/idd-aw/PSPNet..."
bsub -J "rtfix_IP2P_iddaw_psp" \
    -q BatchGPU \
    -R "rusage[mem=16000]" \
    -gpu "num=1:gmem=16G" \
    -W 16:00 \
    -o "${LOG_DIR}/rtfix_IP2P_iddaw_psp_%J.out" \
    -e "${LOG_DIR}/rtfix_IP2P_iddaw_psp_%J.err" \
    "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python unified_training.py --dataset IDD-AW --model pspnet_r50 --strategy gen_IP2P --real-gen-ratio 0.5 --no-early-stop --max-iters 80000"
((submitted++))

echo ""
echo "=== Summary ==="
echo "Submitted: $submitted retraining jobs"
echo ""
echo "Training time estimates:"
echo "  - DeepLabV3+/PSPNet: ~8-12 hours"
echo "  - SegFormer: ~12-20 hours"
echo ""
echo "Note: Old corrupted weights will be overwritten automatically."
echo "After training completes, run tests with fine_grained_test.py"
