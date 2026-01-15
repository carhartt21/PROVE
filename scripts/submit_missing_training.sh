#!/bin/bash
# Submit missing training jobs to achieve 100% training coverage
# Missing configs:
# 1. gen_Weather_Effect_Generator/bdd10k/SegFormer (resume from iter_70000)
# 2-9. std_minimal: bdd10k/SF, idd-aw/SF, mapillaryvistas/all, outside15k/all

cd /home/mima2416/repositories/PROVE
PROJECT_ROOT="/home/mima2416/repositories/PROVE"
WEIGHTS_ROOT="/scratch/aaa_exchange/AWARE/WEIGHTS"
LOG_DIR="${PROJECT_ROOT}/logs"

echo "=== Submitting Missing Training Jobs ==="
echo ""

submitted=0

# 1. Resume gen_Weather_Effect_Generator/bdd10k/SegFormer from iter_70000
echo "1. Resuming gen_Weather_Effect_Generator/bdd10k/SegFormer from iter_70000..."
CHECKPOINT="${WEIGHTS_ROOT}/gen_Weather_Effect_Generator/bdd10k_cd/segformer_mit-b5_ratio0p50/iter_70000.pth"
if [ -f "$CHECKPOINT" ]; then
    bsub -J "rt_gen_Weather_E_bdd10_segf" \
        -n 10 \
        -q BatchGPU \
        -R "rusage[mem=20000]" \
        -gpu "num=1:gmem=24G" \
        -W 24:00 \
        -o "${LOG_DIR}/rt_gen_Weather_bdd10_segf_%J.out" \
        -e "${LOG_DIR}/rt_gen_Weather_bdd10_segf_%J.err" \
        "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python unified_training.py --dataset BDD10k --model segformer_mit-b5 --strategy gen_Weather_Effect_Generator --real-gen-ratio 0.5 --no-early-stop --max-iters 80000 --resume-from ${CHECKPOINT}"
    ((submitted++))
else
    echo "  WARNING: Checkpoint not found: $CHECKPOINT"
fi

# 2. std_minimal/bdd10k/SegFormer
echo "2. Submitting std_minimal/bdd10k/SegFormer..."
bsub -J "rt_std_minimal_bdd10_segf" \
    -q BatchGPU \
    -R "rusage[mem=20000]" \
    -gpu "num=1:gmem=24G" \
    -W 24:00 \
    -o "${LOG_DIR}/rt_std_minimal_bdd10_segf_%J.out" \
    -e "${LOG_DIR}/rt_std_minimal_bdd10_segf_%J.err" \
    "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python unified_training.py --dataset BDD10k --model segformer_mit-b5 --strategy std_minimal --no-early-stop --max-iters 80000"
((submitted++))

# 3. std_minimal/idd-aw/SegFormer
echo "3. Submitting std_minimal/idd-aw/SegFormer..."
bsub -J "rt_std_minimal_iddaw_segf" \
    -q BatchGPU \
    -R "rusage[mem=20000]" \
    -gpu "num=1:gmem=24G" \
    -W 24:00 \
    -o "${LOG_DIR}/rt_std_minimal_iddaw_segf_%J.out" \
    -e "${LOG_DIR}/rt_std_minimal_iddaw_segf_%J.err" \
    "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python unified_training.py --dataset IDD-AW --model segformer_mit-b5 --strategy std_minimal --no-early-stop --max-iters 80000"
((submitted++))

# 4. std_minimal/mapillaryvistas/DeepLabV3+
echo "4. Submitting std_minimal/mapillaryvistas/DeepLabV3+..."
bsub -J "rt_std_minimal_mapi_dlv3p" \
    -q BatchGPU \
    -R "rusage[mem=16000]" \
    -gpu "num=1:gmem=16G" \
    -W 16:00 \
    -o "${LOG_DIR}/rt_std_minimal_mapi_dlv3p_%J.out" \
    -e "${LOG_DIR}/rt_std_minimal_mapi_dlv3p_%J.err" \
    "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python unified_training.py --dataset MapillaryVistas --model deeplabv3plus_r50 --strategy std_minimal --no-early-stop --max-iters 80000"
((submitted++))

# 5. std_minimal/mapillaryvistas/PSPNet
echo "5. Submitting std_minimal/mapillaryvistas/PSPNet..."
bsub -J "rt_std_minimal_mapi_pspn" \
    -q BatchGPU \
    -R "rusage[mem=16000]" \
    -gpu "num=1:gmem=16G" \
    -W 16:00 \
    -o "${LOG_DIR}/rt_std_minimal_mapi_pspn_%J.out" \
    -e "${LOG_DIR}/rt_std_minimal_mapi_pspn_%J.err" \
    "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python unified_training.py --dataset MapillaryVistas --model pspnet_r50 --strategy std_minimal --no-early-stop --max-iters 80000"
((submitted++))

# 6. std_minimal/mapillaryvistas/SegFormer
echo "6. Submitting std_minimal/mapillaryvistas/SegFormer..."
bsub -J "rt_std_minimal_mapi_segf" \
    -q BatchGPU \
    -R "rusage[mem=20000]" \
    -gpu "num=1:gmem=24G" \
    -W 24:00 \
    -o "${LOG_DIR}/rt_std_minimal_mapi_segf_%J.out" \
    -e "${LOG_DIR}/rt_std_minimal_mapi_segf_%J.err" \
    "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python unified_training.py --dataset MapillaryVistas --model segformer_mit-b5 --strategy std_minimal --no-early-stop --max-iters 80000"
((submitted++))

# 7. std_minimal/outside15k/DeepLabV3+
echo "7. Submitting std_minimal/outside15k/DeepLabV3+..."
bsub -J "rt_std_minimal_outs_dlv3p" \
    -q BatchGPU \
    -R "rusage[mem=16000]" \
    -gpu "num=1:gmem=16G" \
    -W 16:00 \
    -o "${LOG_DIR}/rt_std_minimal_outs_dlv3p_%J.out" \
    -e "${LOG_DIR}/rt_std_minimal_outs_dlv3p_%J.err" \
    "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python unified_training.py --dataset OUTSIDE15k --model deeplabv3plus_r50 --strategy std_minimal --no-early-stop --max-iters 80000"
((submitted++))

# 8. std_minimal/outside15k/PSPNet
echo "8. Submitting std_minimal/outside15k/PSPNet..."
bsub -J "rt_std_minimal_outs_pspn" \
    -q BatchGPU \
    -R "rusage[mem=16000]" \
    -gpu "num=1:gmem=16G" \
    -W 16:00 \
    -o "${LOG_DIR}/rt_std_minimal_outs_pspn_%J.out" \
    -e "${LOG_DIR}/rt_std_minimal_outs_pspn_%J.err" \
    "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python unified_training.py --dataset OUTSIDE15k --model pspnet_r50 --strategy std_minimal --no-early-stop --max-iters 80000"
((submitted++))

# 9. std_minimal/outside15k/SegFormer
echo "9. Submitting std_minimal/outside15k/SegFormer..."
bsub -J "rt_std_minimal_outs_segf" \
    -q BatchGPU \
    -R "rusage[mem=20000]" \
    -gpu "num=1:gmem=24G" \
    -W 24:00 \
    -o "${LOG_DIR}/rt_std_minimal_outs_segf_%J.out" \
    -e "${LOG_DIR}/rt_std_minimal_outs_segf_%J.err" \
    "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && python unified_training.py --dataset OUTSIDE15k --model segformer_mit-b5 --strategy std_minimal --no-early-stop --max-iters 80000"
((submitted++))

echo ""
echo "=== Summary ==="
echo "Submitted: $submitted training jobs"
echo ""
echo "Note: These are Stage 1 (clear_day) training jobs."
echo "Training time estimates:"
echo "  - DeepLabV3+/PSPNet: ~8-12 hours"
echo "  - SegFormer: ~12-20 hours"
