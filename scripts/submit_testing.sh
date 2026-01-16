#!/bin/bash
# =============================================================================
# PROVE Testing Job Submission Template
# =============================================================================
# Usage: ./submit_testing.sh [OPTIONS]
#
# This script submits testing jobs to the LSF cluster for fine-grained 
# per-domain and per-class evaluation.
#
# Examples:
#   # Test a specific model
#   ./submit_testing.sh --checkpoint /path/to/iter_80000.pth --dataset BDD10k
#
#   # Test with custom output directory
#   ./submit_testing.sh --checkpoint /path/to/iter_80000.pth --dataset ACDC --output-dir results/my_test
#
#   # Dry run
#   ./submit_testing.sh --checkpoint /path/to/iter_80000.pth --dataset BDD10k --dry-run
#
#   # Auto-detect config from checkpoint directory
#   ./submit_testing.sh --weights-dir /path/to/weights/strategy/dataset/model --dataset BDD10k
# =============================================================================

set -e

# Default values
PROVE_DIR="/home/mima2416/repositories/PROVE"
LOG_DIR="${PROVE_DIR}/logs"
QUEUE="BatchGPU"
GPU_MEM="24G"
GPU_MODE="shared"
NUM_CPUS=10
WALL_TIME="0:30"
BATCH_SIZE=8
TEST_SPLIT="val"
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --weights-dir) WEIGHTS_DIR="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --test-split) TEST_SPLIT="$2"; shift 2 ;;
        --queue) QUEUE="$2"; shift 2 ;;
        --gpu-mem) GPU_MEM="$2"; shift 2 ;;
        --num-cpus) NUM_CPUS="$2"; shift 2 ;;
        --wall-time) WALL_TIME="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Required:"
            echo "  --checkpoint PATH     Path to checkpoint file (iter_*.pth)"
            echo "  --dataset DATASET     Dataset name (ACDC, BDD10k, IDD-AW, MapillaryVistas, OUTSIDE15k)"
            echo ""
            echo "Auto-detection (alternative to --checkpoint + --config):"
            echo "  --weights-dir PATH    Weights directory (auto-detects config and checkpoint)"
            echo ""
            echo "Optional:"
            echo "  --config PATH         Path to config file (auto-detected if not specified)"
            echo "  --output-dir PATH     Output directory (auto-generated if not specified)"
            echo "  --batch-size N        Batch size for inference (default: 8)"
            echo "  --test-split SPLIT    Test split: val or test (default: val)"
            echo "  --queue QUEUE         LSF queue (default: BatchGPU)"
            echo "  --gpu-mem SIZE        GPU memory (default: 24G)"
            echo "  --num-cpus N          Number of CPUs (default: 10)"
            echo "  --wall-time TIME      Wall time limit (default: 0:30)"
            echo "  --dry-run             Show command without submitting"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Handle weights-dir mode
if [ -n "$WEIGHTS_DIR" ]; then
    # Auto-detect checkpoint
    if [ -z "$CHECKPOINT" ]; then
        CHECKPOINT=$(ls "${WEIGHTS_DIR}"/iter_*.pth 2>/dev/null | sort -V | tail -1)
        if [ -z "$CHECKPOINT" ]; then
            echo "Error: No checkpoint found in ${WEIGHTS_DIR}"
            exit 1
        fi
    fi
    
    # Auto-detect config
    if [ -z "$CONFIG" ]; then
        CONFIG=$(ls "${WEIGHTS_DIR}/configs/"*.py 2>/dev/null | head -1)
        if [ -z "$CONFIG" ]; then
            CONFIG="${WEIGHTS_DIR}/training_config.py"
        fi
    fi
fi

# Validate required arguments
if [ -z "$CHECKPOINT" ] || [ -z "$DATASET" ]; then
    echo "Error: --checkpoint (or --weights-dir) and --dataset are required"
    echo "Use --help for usage information"
    exit 1
fi

# Auto-detect config from checkpoint path if not specified
if [ -z "$CONFIG" ]; then
    CKPT_DIR=$(dirname "$CHECKPOINT")
    CONFIG=$(ls "${CKPT_DIR}/configs/"*.py 2>/dev/null | head -1)
    if [ -z "$CONFIG" ]; then
        CONFIG="${CKPT_DIR}/training_config.py"
    fi
fi

# Validate config exists
if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file not found: ${CONFIG}"
    echo "Please specify --config explicitly"
    exit 1
fi

# Validate checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found: ${CHECKPOINT}"
    exit 1
fi

# Auto-generate output directory if not specified
if [ -z "$OUTPUT_DIR" ]; then
    # Extract path components from checkpoint path
    # e.g., /path/to/WEIGHTS/strategy/dataset/model/iter_80000.pth
    CKPT_DIR=$(dirname "$CHECKPOINT")
    MODEL_NAME=$(basename "$CKPT_DIR")
    DATASET_DIR=$(dirname "$CKPT_DIR")
    DATASET_NAME=$(basename "$DATASET_DIR")
    STRATEGY_DIR=$(dirname "$DATASET_DIR")
    STRATEGY_NAME=$(basename "$STRATEGY_DIR")
    
    OUTPUT_DIR="results/${STRATEGY_NAME}/${DATASET_NAME}/${MODEL_NAME}"
fi

# Create log directory
mkdir -p "$LOG_DIR"

# Build job name
JOB_NAME="test_$(basename $(dirname $(dirname $CHECKPOINT)))_$(basename $(dirname $CHECKPOINT))_${DATASET}"

# Build test command
TEST_CMD="cd ${PROVE_DIR} && python fine_grained_test.py"
TEST_CMD="${TEST_CMD} --config ${CONFIG}"
TEST_CMD="${TEST_CMD} --checkpoint ${CHECKPOINT}"
TEST_CMD="${TEST_CMD} --dataset ${DATASET}"
TEST_CMD="${TEST_CMD} --output-dir ${OUTPUT_DIR}"
TEST_CMD="${TEST_CMD} --batch-size ${BATCH_SIZE}"
TEST_CMD="${TEST_CMD} --test-split ${TEST_SPLIT}"

# Full command with environment setup
FULL_CMD="source ~/.bashrc && conda activate prove && ${TEST_CMD}"

# Show what will be run
echo "=== Testing Job Submission ==="
echo "Job Name:    ${JOB_NAME}"
echo "Config:      ${CONFIG}"
echo "Checkpoint:  ${CHECKPOINT}"
echo "Dataset:     ${DATASET}"
echo "Output:      ${OUTPUT_DIR}"
echo "Batch Size:  ${BATCH_SIZE}"
echo "Test Split:  ${TEST_SPLIT}"
echo "Queue:       ${QUEUE}"
echo "GPU Memory:  ${GPU_MEM}"
echo ""
echo "Command: python fine_grained_test.py --config ... --checkpoint ... --dataset ${DATASET} --output-dir ${OUTPUT_DIR}"
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
