#!/bin/bash
# Submit missing test jobs for extended training study
# These are models that have completed 160k training but are missing test results
#
# Missing tests:
# 1. gen_Img2Img / outside15k / pspnet_r50
# 2. gen_Img2Img / outside15k / segformer_mit-b5
# 3. gen_LANIT / idd-aw / deeplabv3plus_r50

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

WEIGHTS_ROOT="/scratch/aaa_exchange/AWARE/WEIGHTS_EXTENDED"
QUEUE="BatchGPU"
GPU_MEM="16G"
GPU_MODE="shared"
NUM_CPUS=4

# Default: dry run
DRY_RUN=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --submit)
            DRY_RUN=false
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--submit]"
            echo "  Without --submit: dry run (shows commands)"
            echo "  With --submit: actually submit jobs"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

mkdir -p logs

echo "=============================================="
echo "Submit Missing Extended Training Test Jobs"
echo "=============================================="
echo ""
echo "Weights root: $WEIGHTS_ROOT"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "Mode: DRY RUN (use --submit to actually submit)"
else
    echo "Mode: SUBMIT"
fi
echo ""

# Define missing tests
declare -a MISSING_TESTS=(
    "gen_Img2Img:OUTSIDE15k:pspnet_r50:outside15k"
    "gen_Img2Img:OUTSIDE15k:segformer_mit-b5:outside15k"
    "gen_LANIT:IDD-AW:deeplabv3plus_r50:idd-aw"
)

for entry in "${MISSING_TESTS[@]}"; do
    IFS=':' read -r strategy dataset_upper model dataset_lower <<< "$entry"
    
    checkpoint="${WEIGHTS_ROOT}/${strategy}/${dataset_lower}/${model}/iter_160000.pth"
    job_name="ext_test_${dataset_upper}_${model}_${strategy}_160k"
    test_cmd="$SCRIPT_DIR/test_unified.sh single --dataset $dataset_upper --model $model --strategy $strategy --checkpoint $checkpoint --work-dir $WEIGHTS_ROOT"
    
    echo "----------------------------------------------"
    echo "Job: $job_name"
    echo "  Strategy:   $strategy"
    echo "  Dataset:    $dataset_upper"
    echo "  Model:      $model"
    echo "  Checkpoint: $checkpoint"
    echo ""
    
    # Verify checkpoint exists
    if [ ! -f "$checkpoint" ]; then
        echo "  WARNING: Checkpoint not found, skipping"
        continue
    fi
    
    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY RUN] Would submit:"
        echo "  bsub -gpu \"num=1:mode=${GPU_MODE}:gmem=${GPU_MEM}\" \\"
        echo "       -q ${QUEUE} -R \"span[hosts=1]\" -n ${NUM_CPUS} \\"
        echo "       -oo \"logs/${job_name}_%J.log\" -eo \"logs/${job_name}_%J.err\" \\"
        echo "       -J \"${job_name}\" \"$test_cmd\""
    else
        echo "  Submitting..."
        bsub -gpu "num=1:mode=${GPU_MODE}:gmem=${GPU_MEM}" \
            -q "${QUEUE}" \
            -R "span[hosts=1]" \
            -n "${NUM_CPUS}" \
            -oo "logs/${job_name}_%J.log" \
            -eo "logs/${job_name}_%J.err" \
            -L /bin/bash \
            -J "${job_name}" \
            "${test_cmd}"
    fi
    echo ""
done

echo "=============================================="
echo "Done"
echo "=============================================="
