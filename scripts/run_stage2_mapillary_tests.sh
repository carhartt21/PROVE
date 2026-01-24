#!/bin/bash
# =============================================================================
# Run Missing Stage 2 MapillaryVistas Tests Locally
# =============================================================================
# This script runs fine-grained tests for all Stage 2 MapillaryVistas
# configurations that have completed training but are missing test results.
#
# Stage 2 models are trained on ALL domains (not just clear_day).
#
# Usage:
#   ./scripts/run_stage2_mapillary_tests.sh              # Run all tests
#   ./scripts/run_stage2_mapillary_tests.sh --dry-run    # Show what would be run
#   ./scripts/run_stage2_mapillary_tests.sh --limit N    # Run only first N tests
#   ./scripts/run_stage2_mapillary_tests.sh --gpu 1      # Use specific GPU
#   ./scripts/run_stage2_mapillary_tests.sh --list-gpus  # List available GPUs
#   ./scripts/run_stage2_mapillary_tests.sh --help       # Show all options
#
# Prerequisites:
#   - CUDA GPU available
#   - 'prove' conda/mamba environment activated
#
# Estimated time: ~10 minutes per test
# =============================================================================

set -e

# Configuration - STAGE 2 uses WEIGHTS_STAGE_2 directory
WEIGHTS_ROOT="/scratch/aaa_exchange/AWARE/WEIGHTS_STAGE_2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$PROJECT_ROOT/logs/stage2_mapillary_tests_$(date +%Y%m%d_%H%M%S).log"

# Default parameters
DRY_RUN=false
LIMIT=0
BATCH_SIZE=8
GPU_ID=""  # Empty means use default (CUDA_VISIBLE_DEVICES or GPU 0)

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --limit)
            LIMIT=$2
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE=$2
            shift 2
            ;;
        --gpu)
            GPU_ID=$2
            shift 2
            ;;
        --list-gpus)
            echo "Available GPUs:"
            nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"
            exit 0
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Run Stage 2 MapillaryVistas tests (trained on ALL domains)"
            echo ""
            echo "Options:"
            echo "  --dry-run       Show what would be run without executing"
            echo "  --limit N       Run only the first N tests"
            echo "  --batch-size N  Batch size for inference (default: 8)"
            echo "  --gpu GPU_ID    Use specific GPU (e.g., --gpu 0, --gpu 1)"
            echo "  --list-gpus     List available GPUs"
            echo "  -h, --help      Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --dry-run                # Preview tests"
            echo "  $0 --gpu 1 --limit 10       # Run 10 tests on GPU 1"
            echo "  $0 --gpu 0 --batch-size 4   # Run all tests on GPU 0 with batch size 4"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--limit N] [--batch-size N] [--gpu GPU_ID]"
            echo "Try '$0 --help' for more information."
            exit 1
            ;;
    esac
done

# Set GPU if specified
if [ -n "$GPU_ID" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

# Ensure log directory exists
mkdir -p "$PROJECT_ROOT/logs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to log messages
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo -e "$msg"
    if [ "$DRY_RUN" = false ]; then
        echo "$msg" >> "$LOG_FILE"
    fi
}

# Function to run a single test
run_test() {
    local config_path="$1"
    local checkpoint_path="$2"
    local output_dir="$3"
    
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY-RUN]${NC} Would run:"
        echo "  Config: $config_path"
        echo "  Checkpoint: $checkpoint_path"
        echo "  Output: $output_dir"
        return 0
    fi
    
    log "${BLUE}Running test:${NC}"
    log "  Config: $config_path"
    log "  Checkpoint: $checkpoint_path"
    log "  Output: $output_dir"
    
    python "$PROJECT_ROOT/fine_grained_test.py" \
        --config "$config_path" \
        --checkpoint "$checkpoint_path" \
        --output-dir "$output_dir" \
        --dataset MapillaryVistas \
        --batch-size "$BATCH_SIZE" \
        2>&1 | tee -a "$LOG_FILE"
    
    local exit_code=${PIPESTATUS[0]}
    if [ $exit_code -eq 0 ]; then
        log "${GREEN}✓ Test completed successfully${NC}"
    else
        log "${RED}✗ Test failed with exit code $exit_code${NC}"
    fi
    
    return $exit_code
}

# Header
echo "============================================================"
echo "Stage 2 MapillaryVistas Fine-Grained Tests"
echo "============================================================"
echo "Time: $(date)"
echo "Weights root: $WEIGHTS_ROOT"
echo "Batch size: $BATCH_SIZE"
if [ -n "$GPU_ID" ]; then
    echo "GPU: $GPU_ID (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
else
    echo "GPU: ${CUDA_VISIBLE_DEVICES:-default}"
fi
if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}DRY RUN MODE - No tests will be executed${NC}"
fi
if [ $LIMIT -gt 0 ]; then
    echo "Limit: $LIMIT tests"
fi
echo "Log file: $LOG_FILE"
echo "============================================================"
echo ""

# Build list of tests to run
declare -a TESTS_TO_RUN

# Find all MapillaryVistas configurations with completed training but no test results
while IFS= read -r checkpoint; do
    # Get the model directory (parent of iter_80000.pth)
    model_dir=$(dirname "$checkpoint")
    
    # Check for training config
    config_path="$model_dir/training_config.py"
    if [ ! -f "$config_path" ]; then
        log "${YELLOW}Warning: No config found at $config_path, skipping${NC}"
        continue
    fi
    
    # Check if test results already exist
    output_dir="$model_dir/test_results_detailed"
    if find "$output_dir" -name "results.json" 2>/dev/null | grep -q .; then
        log "${GREEN}Skipping (already tested): $model_dir${NC}"
        continue
    fi
    
    # Add to list
    TESTS_TO_RUN+=("$config_path|$checkpoint|$output_dir")
    
done < <(find "$WEIGHTS_ROOT" -path "*/mapillaryvistas/*/iter_80000.pth" 2>/dev/null | sort)

# Summary
total_tests=${#TESTS_TO_RUN[@]}
echo "Found $total_tests configurations needing tests"
echo ""

if [ $total_tests -eq 0 ]; then
    echo -e "${GREEN}All Stage 2 MapillaryVistas tests are complete!${NC}"
    exit 0
fi

# Apply limit if specified
if [ $LIMIT -gt 0 ] && [ $LIMIT -lt $total_tests ]; then
    total_tests=$LIMIT
    echo "Limiting to $LIMIT tests"
fi

# Confirm before running (unless dry-run)
if [ "$DRY_RUN" = false ]; then
    echo ""
    echo -e "${YELLOW}About to run $total_tests tests.${NC}"
    echo "Estimated time: ~$(( total_tests * 10 )) minutes"
    echo ""
    read -p "Continue? [y/N] " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# Run tests
success_count=0
fail_count=0
skip_count=0

for i in "${!TESTS_TO_RUN[@]}"; do
    # Check limit
    if [ $LIMIT -gt 0 ] && [ $i -ge $LIMIT ]; then
        break
    fi
    
    # Parse test info
    IFS='|' read -r config_path checkpoint output_dir <<< "${TESTS_TO_RUN[$i]}"
    
    echo ""
    echo "============================================================"
    echo "Test $(( i + 1 ))/$total_tests"
    echo "============================================================"
    
    # Run the test
    if run_test "$config_path" "$checkpoint" "$output_dir"; then
        ((success_count++)) || true
    else
        ((fail_count++)) || true
    fi
done

# Final summary
echo ""
echo "============================================================"
echo "SUMMARY"
echo "============================================================"
echo -e "Successful: ${GREEN}$success_count${NC}"
echo -e "Failed: ${RED}$fail_count${NC}"
echo -e "Skipped: ${YELLOW}$skip_count${NC}"
echo "Total: $total_tests"
if [ "$DRY_RUN" = false ]; then
    echo "Log file: $LOG_FILE"
fi
echo "============================================================"

# Exit with error if any tests failed
if [ $fail_count -gt 0 ]; then
    exit 1
fi
