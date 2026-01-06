#!/bin/bash
# Submit missing ratio ablation jobs in chunks by strategy
# Run this from an HPC login node where bsub is available

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration
STRATEGIES=("gen_LANIT" "gen_step1x_new" "gen_automold" "gen_TSIT" "gen_NST")
# Ratios to test (0.0 to 1.0, excluding 0.5 which is standard training)
# 0.0 = 100% synthetic images (new addition)
RATIOS=("0.0" "0.125" "0.25" "0.375" "0.625" "0.75" "0.875" "1.0")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

usage() {
    echo "Submit Missing Ratio Ablation Jobs (Chunked by Strategy)"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --strategy STRATEGY   Submit jobs for a specific strategy only"
    echo "  --ratio RATIO         Submit jobs for a specific ratio only"
    echo "  --list                List missing jobs without submitting"
    echo "  --all                 Submit all missing jobs (all strategies)"
    echo "  --interactive         Interactive mode: confirm each strategy"
    echo "  --limit N             Limit number of jobs per strategy"
    echo "  --help                Show this help message"
    echo ""
    echo "Available strategies: ${STRATEGIES[*]}"
    echo "Available ratios: ${RATIOS[*]}"
    echo ""
    echo "Examples:"
    echo "  $0 --list                           # List all missing jobs"
    echo "  $0 --strategy gen_TSIT              # Submit only gen_TSIT jobs"
    echo "  $0 --strategy gen_LANIT --ratio 1.0 # Submit gen_LANIT ratio 1.0 only"
    echo "  $0 --interactive                    # Submit with confirmation per strategy"
    echo "  $0 --all                            # Submit everything"
}

# Parse arguments
FILTER_STRATEGY=""
FILTER_RATIO=""
LIST_ONLY=false
SUBMIT_ALL=false
INTERACTIVE=false
LIMIT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --strategy)
            FILTER_STRATEGY="$2"
            shift 2
            ;;
        --ratio)
            FILTER_RATIO="$2"
            shift 2
            ;;
        --list)
            LIST_ONLY=true
            shift
            ;;
        --all)
            SUBMIT_ALL=true
            shift
            ;;
        --interactive)
            INTERACTIVE=true
            shift
            ;;
        --limit)
            LIMIT="--limit $2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Check if bsub is available (unless just listing)
if [[ "$LIST_ONLY" != "true" ]]; then
    if ! command -v bsub &> /dev/null; then
        echo -e "${RED}ERROR: bsub command not found. Please run this from an HPC login node.${NC}"
        exit 1
    fi
fi

echo "=============================================="
echo "Missing Ratio Ablation Jobs Submission"
echo "=============================================="
echo ""

# Function to count missing jobs for a strategy/ratio
count_missing() {
    local strategy=$1
    local ratio=$2
    local ratio_arg=""
    
    if [[ -n "$ratio" ]]; then
        ratio_arg="--ratio $ratio"
    fi
    
    "$SCRIPT_DIR/submit_ratio_ablation.sh" --strategy "$strategy" $ratio_arg --list 2>/dev/null | \
        grep -c "^${strategy}" || echo "0"
}

# Function to submit jobs for a strategy
submit_strategy() {
    local strategy=$1
    local ratio_filter=$2
    local ratio_arg=""
    
    if [[ -n "$ratio_filter" ]]; then
        ratio_arg="--ratio $ratio_filter"
    fi
    
    echo -e "${BLUE}Submitting: $strategy ${ratio_filter}${NC}"
    
    if [[ "$LIST_ONLY" == "true" ]]; then
        "$SCRIPT_DIR/submit_ratio_ablation.sh" --strategy "$strategy" $ratio_arg --list $LIMIT
    else
        "$SCRIPT_DIR/submit_ratio_ablation.sh" --strategy "$strategy" $ratio_arg $LIMIT
    fi
}

# Summary of missing jobs
echo "Missing Jobs Summary:"
echo "---------------------"
echo ""

declare -A STRATEGY_COUNTS

for strategy in "${STRATEGIES[@]}"; do
    if [[ -n "$FILTER_STRATEGY" && "$strategy" != "$FILTER_STRATEGY" ]]; then
        continue
    fi
    
    total=0
    echo -e "${YELLOW}$strategy:${NC}"
    
    for ratio in "${RATIOS[@]}"; do
        if [[ -n "$FILTER_RATIO" && "$ratio" != "$FILTER_RATIO" ]]; then
            continue
        fi
        
        # Count existing completed experiments
        # Convert ratio using Python to match exact format used in training config
        # Format: f'_ratio{real_gen_ratio:.2f}'.replace('.', 'p')
        # e.g., 0.125 -> 0p12, 0.375 -> 0p38 (rounds to 2 decimals)
        ratio_pattern=$(python3 -c "print(f'{${ratio}:.2f}'.replace('.', 'p'))")
        
        count=$(find /scratch/aaa_exchange/AWARE/WEIGHTS_RATIO_ABLATION/${strategy} \
                -path "*_ratio${ratio_pattern}*" -name "iter_80000.pth" 2>/dev/null | wc -l)
        
        missing=$((15 - count))  # 5 datasets × 3 models = 15
        if [[ $missing -gt 0 ]]; then
            echo "  Ratio $ratio: $missing missing (${count}/15 complete)"
            total=$((total + missing))
        else
            echo "  Ratio $ratio: ✓ Complete"
        fi
    done
    
    STRATEGY_COUNTS[$strategy]=$total
    
    if [[ $total -gt 0 ]]; then
        echo -e "  ${RED}Total missing: $total${NC}"
    else
        echo -e "  ${GREEN}All complete!${NC}"
    fi
    echo ""
done

# Calculate grand total
grand_total=0
for count in "${STRATEGY_COUNTS[@]}"; do
    grand_total=$((grand_total + count))
done

echo "=============================================="
echo -e "Grand Total Missing: ${RED}$grand_total${NC} jobs"
echo "=============================================="
echo ""

# Exit if just listing
if [[ "$LIST_ONLY" == "true" ]]; then
    echo "Use --all or --strategy to submit jobs."
    exit 0
fi

# Confirm submission
if [[ "$SUBMIT_ALL" != "true" && "$INTERACTIVE" != "true" && -z "$FILTER_STRATEGY" ]]; then
    echo "No submission mode specified."
    echo "Use --all, --interactive, or --strategy <name>"
    exit 1
fi

# Submit jobs
echo ""
echo "Starting submission..."
echo ""

for strategy in "${STRATEGIES[@]}"; do
    if [[ -n "$FILTER_STRATEGY" && "$strategy" != "$FILTER_STRATEGY" ]]; then
        continue
    fi
    
    if [[ ${STRATEGY_COUNTS[$strategy]} -eq 0 ]]; then
        echo -e "${GREEN}Skipping $strategy (already complete)${NC}"
        continue
    fi
    
    if [[ "$INTERACTIVE" == "true" ]]; then
        echo -e "${YELLOW}Strategy: $strategy (${STRATEGY_COUNTS[$strategy]} jobs)${NC}"
        read -p "Submit? [y/N] " confirm
        if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
            echo "Skipped."
            continue
        fi
    fi
    
    echo ""
    echo "=============================================="
    echo -e "${BLUE}Submitting: $strategy${NC}"
    echo "=============================================="
    
    if [[ -n "$FILTER_RATIO" ]]; then
        submit_strategy "$strategy" "$FILTER_RATIO"
    else
        for ratio in "${RATIOS[@]}"; do
            # Skip if ratio is complete for this strategy
            ratio_pattern=$(python3 -c "print(f'{${ratio}:.2f}'.replace('.', 'p'))")
            count=$(find /scratch/aaa_exchange/AWARE/WEIGHTS_RATIO_ABLATION/${strategy} \
                    -path "*_ratio${ratio_pattern}*" -name "iter_80000.pth" 2>/dev/null | wc -l)
            
            if [[ $count -lt 15 ]]; then
                submit_strategy "$strategy" "$ratio"
            fi
        done
    fi
    
    echo ""
done

echo "=============================================="
echo -e "${GREEN}Submission complete!${NC}"
echo "=============================================="
echo ""
echo "Monitor with: bjobs -w"
echo "Check logs:   ls -la logs/ratio_*.log"
