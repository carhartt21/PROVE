#!/bin/bash
# Submit detailed tests for strategies with missing per-domain results
# Generated on Jan 8, 2026

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

WEIGHTS_ROOT="/scratch/aaa_exchange/AWARE/WEIGHTS"
DATA_ROOT="/scratch/aaa_exchange/AWARE/FINAL_SPLITS"

echo "==================================="
echo "Submit Missing Detailed Tests"
echo "==================================="

count=0

# std_autoaugment - 12 configs
for dataset in bdd10k idd-aw mapillaryvistas outside15k; do
    for model in deeplabv3plus_r50 pspnet_r50 segformer_mit-b5; do
        count=$((count + 1))
        echo ""
        echo "[$count] std_autoaugment / $dataset / $model"
        
        bash "$SCRIPT_DIR/test_unified.sh" submit-detailed \
            --dataset "$dataset" \
            --model "$model" \
            --strategy std_autoaugment \
            --ratio 1.0 \
            --work-dir "$WEIGHTS_ROOT" \
            --data-root "$DATA_ROOT"
    done
done

# gen_CUT - 11 configs (missing outside15k/pspnet_r50)
for dataset in bdd10k idd-aw mapillaryvistas; do
    for model in deeplabv3plus_r50 pspnet_r50 segformer_mit-b5; do
        count=$((count + 1))
        echo ""
        echo "[$count] gen_CUT / $dataset / $model"
        
        bash "$SCRIPT_DIR/test_unified.sh" submit-detailed \
            --dataset "$dataset" \
            --model "$model" \
            --strategy gen_CUT \
            --ratio 1.0 \
            --work-dir "$WEIGHTS_ROOT" \
            --data-root "$DATA_ROOT"
    done
done

# gen_CUT outside15k - only deeplabv3plus_r50 and segformer_mit-b5
for model in deeplabv3plus_r50 segformer_mit-b5; do
    count=$((count + 1))
    echo ""
    echo "[$count] gen_CUT / outside15k / $model"
    
    bash "$SCRIPT_DIR/test_unified.sh" submit-detailed \
        --dataset outside15k \
        --model "$model" \
        --strategy gen_CUT \
        --ratio 1.0 \
        --work-dir "$WEIGHTS_ROOT" \
        --data-root "$DATA_ROOT"
done

# gen_StyleID - 3 configs
count=$((count + 1))
echo ""
echo "[$count] gen_StyleID / bdd10k / deeplabv3plus_r50"
bash "$SCRIPT_DIR/test_unified.sh" submit-detailed \
    --dataset bdd10k \
    --model deeplabv3plus_r50 \
    --strategy gen_StyleID \
    --ratio 1.0 \
    --work-dir "$WEIGHTS_ROOT" \
    --data-root "$DATA_ROOT"

count=$((count + 1))
echo ""
echo "[$count] gen_StyleID / mapillaryvistas / pspnet_r50"
bash "$SCRIPT_DIR/test_unified.sh" submit-detailed \
    --dataset mapillaryvistas \
    --model pspnet_r50 \
    --strategy gen_StyleID \
    --ratio 1.0 \
    --work-dir "$WEIGHTS_ROOT" \
    --data-root "$DATA_ROOT"

count=$((count + 1))
echo ""
echo "[$count] gen_StyleID / outside15k / segformer_mit-b5"
bash "$SCRIPT_DIR/test_unified.sh" submit-detailed \
    --dataset outside15k \
    --model segformer_mit-b5 \
    --strategy gen_StyleID \
    --ratio 1.0 \
    --work-dir "$WEIGHTS_ROOT" \
    --data-root "$DATA_ROOT"

echo ""
echo "==================================="
echo "Total jobs submitted: $count"
echo "==================================="
