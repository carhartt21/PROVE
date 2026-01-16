#!/bin/bash
# =============================================================================
# Restructure WEIGHTS directories
# 
# Purpose: Separate Stage 1 (_cd = clear_day) and Stage 2 (_ad = all_domains)
# - WEIGHTS: Stage 1 - clear_day training only
# - WEIGHTS_STAGE_2: Stage 2 - all_domains training
#
# This makes it clearer which stage each model belongs to.
# =============================================================================

set -e

WEIGHTS_DIR="/scratch/aaa_exchange/AWARE/WEIGHTS"
WEIGHTS_STAGE_2_DIR="/scratch/aaa_exchange/AWARE/WEIGHTS_STAGE_2"

# Create the Stage 2 directory if it doesn't exist
mkdir -p "$WEIGHTS_STAGE_2_DIR"

echo "==================================================================="
echo "Moving _ad directories from WEIGHTS to WEIGHTS_STAGE_2"
echo "==================================================================="
echo ""
echo "Source: $WEIGHTS_DIR"
echo "Destination: $WEIGHTS_STAGE_2_DIR"
echo ""

# Find all strategies that have _ad directories
cd "$WEIGHTS_DIR"

count=0
for strategy_dir in */; do
    strategy="${strategy_dir%/}"
    
    # Skip non-strategy directories
    [[ "$strategy" == "domain_adaptation_ablation" ]] && continue
    
    # Find _ad directories in this strategy
    has_ad=false
    for ad_dir in "$strategy_dir"*_ad/; do
        if [[ -d "$ad_dir" ]]; then
            has_ad=true
            break
        fi
    done
    
    if [[ "$has_ad" == true ]]; then
        # Create strategy directory in Stage 2
        mkdir -p "$WEIGHTS_STAGE_2_DIR/$strategy"
        
        # Move all _ad directories
        for ad_dir in "$strategy_dir"*_ad; do
            if [[ -d "$ad_dir" ]]; then
                dataset_ad=$(basename "$ad_dir")
                echo "  Moving $strategy/$dataset_ad -> WEIGHTS_STAGE_2/$strategy/$dataset_ad"
                mv "$ad_dir" "$WEIGHTS_STAGE_2_DIR/$strategy/"
                ((count++))
            fi
        done
    fi
done

echo ""
echo "==================================================================="
echo "Completed! Moved $count directories."
echo ""
echo "WEIGHTS now contains only Stage 1 (_cd = clear_day) models"
echo "WEIGHTS_STAGE_2 contains Stage 2 (_ad = all_domains) models"
echo "==================================================================="

# Verify the move
echo ""
echo "Verifying..."
echo ""
echo "Remaining _ad in WEIGHTS:"
find "$WEIGHTS_DIR" -maxdepth 2 -type d -name "*_ad" | wc -l
echo ""
echo "_ad directories in WEIGHTS_STAGE_2:"
find "$WEIGHTS_STAGE_2_DIR" -maxdepth 2 -type d -name "*_ad" | wc -l
