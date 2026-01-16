#!/bin/bash
# =============================================================================
# PROVE Training Job Submission Template
# =============================================================================
# Usage: ./submit_training.sh [OPTIONS]
#
# This script submits training jobs to the LSF cluster. It can submit a single
# job or batch jobs with various configurations.
#
# Examples:
#   # Single job
#   ./submit_training.sh --dataset BDD10k --model segformer_mit-b5 --strategy gen_cycleGAN
#
#   # Stage 1 training (clear_day only)
#   ./submit_training.sh --dataset BDD10k --model segformer_mit-b5 --strategy baseline --domain-filter clear_day
#
#   # With custom ratio
#   ./submit_training.sh --dataset BDD10k --model segformer_mit-b5 --strategy gen_cycleGAN --ratio 0.5
#
#   # Dry run (show command without submitting)
#   ./submit_training.sh --dataset BDD10k --model segformer_mit-b5 --strategy baseline --dry-run
# =============================================================================

set -e

# Default values
PROVE_DIR="/home/mima2416/repositories/PROVE"
LOG_DIR="${PROVE_DIR}/logs"
QUEUE="BatchGPU"
GPU_MEM="24G"
GPU_MODE="shared"
NUM_CPUS=10
WALL_TIME="24:00"
MAX_ITERS=""
DOMAIN_FILTER=""
RATIO="0.5"
STD_STRATEGY=""
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --strategy) STRATEGY="$2"; shift 2 ;;
        --std-strategy) STD_STRATEGY="$2"; shift 2 ;;
        --ratio) RATIO="$2"; shift 2 ;;
        --domain-filter) DOMAIN_FILTER="$2"; shift 2 ;;
        --max-iters) MAX_ITERS="$2"; shift 2 ;;
        --queue) QUEUE="$2"; shift 2 ;;
        --gpu-mem) GPU_MEM="$2"; shift 2 ;;
        --num-cpus) NUM_CPUS="$2"; shift 2 ;;
        --wall-time) WALL_TIME="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Required:"
            echo "  --dataset DATASET     Dataset name (ACDC, BDD10k, IDD-AW, MapillaryVistas, OUTSIDE15k)"
            echo "  --model MODEL         Model name (deeplabv3plus_r50, pspnet_r50, segformer_mit-b5)"
            echo "  --strategy STRATEGY   Strategy name (baseline, gen_*, std_*)"
            echo ""
            echo "Optional:"
            echo "  --std-strategy STD    Additional standard augmentation (std_cutmix, std_mixup, etc.)"
            echo "  --ratio RATIO         Real-to-generated ratio (default: 0.5)"
            echo "  --domain-filter DOM   Filter to specific domain (e.g., clear_day)"
            echo "  --max-iters N         Maximum iterations (default: 80000)"
            echo "  --queue QUEUE         LSF queue (default: BatchGPU)"
            echo "  --gpu-mem SIZE        GPU memory (default: 24G)"
            echo "  --num-cpus N          Number of CPUs (default: 8)"
            echo "  --wall-time TIME      Wall time limit (default: 24:00)"
            echo "  --dry-run             Show command without submitting"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Validate required arguments
if [ -z "$DATASET" ] || [ -z "$MODEL" ] || [ -z "$STRATEGY" ]; then
    echo "Error: --dataset, --model, and --strategy are required"
    echo "Use --help for usage information"
    exit 1
fi

# Create log directory
mkdir -p "$LOG_DIR"

# Build job name
JOB_NAME="train_${STRATEGY}_${DATASET}_${MODEL}"
if [ -n "$DOMAIN_FILTER" ]; then
    JOB_NAME="${JOB_NAME}_${DOMAIN_FILTER}"
fi

# Build training command
TRAIN_CMD="cd ${PROVE_DIR} && python unified_training.py"
TRAIN_CMD="${TRAIN_CMD} --dataset ${DATASET}"
TRAIN_CMD="${TRAIN_CMD} --model ${MODEL}"
TRAIN_CMD="${TRAIN_CMD} --strategy ${STRATEGY}"

if [ -n "$STD_STRATEGY" ]; then
    TRAIN_CMD="${TRAIN_CMD} --std-strategy ${STD_STRATEGY}"
fi

if [[ "$STRATEGY" == gen_* ]]; then
    TRAIN_CMD="${TRAIN_CMD} --real-gen-ratio ${RATIO}"
fi

if [ -n "$DOMAIN_FILTER" ]; then
    TRAIN_CMD="${TRAIN_CMD} --domain-filter ${DOMAIN_FILTER}"
fi

if [ -n "$MAX_ITERS" ]; then
    TRAIN_CMD="${TRAIN_CMD} --max-iters ${MAX_ITERS}"
fi

# Full command with environment setup
FULL_CMD="source ~/.bashrc && conda activate prove && ${TRAIN_CMD}"

# Show what will be run
echo "=== Training Job Submission ==="
echo "Job Name:    ${JOB_NAME}"
echo "Dataset:     ${DATASET}"
echo "Model:       ${MODEL}"
echo "Strategy:    ${STRATEGY}"
[ -n "$STD_STRATEGY" ] && echo "Std Strategy: ${STD_STRATEGY}"
[ -n "$DOMAIN_FILTER" ] && echo "Domain:      ${DOMAIN_FILTER}"
[[ "$STRATEGY" == gen_* ]] && echo "Ratio:       ${RATIO}"
[ -n "$MAX_ITERS" ] && echo "Max Iters:   ${MAX_ITERS}"
echo "Queue:       ${QUEUE}"
echo "GPU Memory:  ${GPU_MEM}"
echo ""
echo "Command: ${TRAIN_CMD}"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would submit:"
    echo "bsub -J \"${JOB_NAME}\" -q ${QUEUE} -gpu \"num=1:mode=${GPU_MODE}:gmem=${GPU_MEM}\" -n ${NUM_CPUS} -W ${WALL_TIME} -o ${LOG_DIR}/${JOB_NAME}.log -e ${LOG_DIR}/${JOB_NAME}.err \"${FULL_CMD}\""
else
    bsub -J "${JOB_NAME}" \
         -q "${QUEUE}" \
         -gpu "num=1:mode=${GPU_MODE}:gmem=${GPU_MEM}" \
         -n "${NUM_CPUS}" \
         -W "${WALL_TIME}" \
         -o "${LOG_DIR}/${JOB_NAME}.log" \
         -e "${LOG_DIR}/${JOB_NAME}.err" \
         "${FULL_CMD}"
    echo "Job submitted!"
fi
