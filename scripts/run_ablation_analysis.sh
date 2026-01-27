#!/bin/bash
#
# Run ablation analysis after testing is complete
# 
# This script:
# 1. Checks if ablation tests have completed
# 2. Runs the ratio ablation analysis
# 3. Runs the extended training analysis
# 4. Generates visualization figures
#
# Usage:
#   ./scripts/run_ablation_analysis.sh [--force]
#

set -e
cd "$(dirname "$0")/.."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "PROVE Ablation Analysis Runner"
echo "=========================================="
echo ""

# Check for running jobs
jobs=$(bjobs -u $USER 2>/dev/null | grep -E "abl_|ext_" | wc -l)
if [ "$jobs" -gt 0 ] && [ "$1" != "--force" ]; then
    echo -e "${YELLOW}Warning: $jobs ablation test jobs still running/pending${NC}"
    echo "Run with --force to analyze partial results"
    echo ""
    echo "Monitor with: bjobs -u $USER | grep -E 'abl_|ext_'"
    exit 1
fi

# Count test results
ratio_results=$(find /scratch/aaa_exchange/AWARE/WEIGHTS_RATIO_ABLATION -name "results.json" 2>/dev/null | wc -l)
extended_results=$(find /scratch/aaa_exchange/AWARE/WEIGHTS_EXTENDED -name "results.json" 2>/dev/null | wc -l)

echo "Found test results:"
echo "  - Ratio Ablation: $ratio_results"
echo "  - Extended Training: $extended_results"
echo ""

# Activate environment
echo "Activating environment..."
source ~/.bashrc
mamba activate prove

# Run ratio ablation analysis
echo ""
echo "=========================================="
echo "Running Ratio Ablation Analysis"
echo "=========================================="
python analysis_scripts/analyze_ratio_ablation.py --detailed

# Generate ratio ablation visualizations
echo ""
echo "=========================================="
echo "Generating Ratio Ablation Figures"
echo "=========================================="
if [ -f "analysis_scripts/visualize_ratio_ablation.py" ]; then
    python analysis_scripts/visualize_ratio_ablation.py || echo "Warning: Visualization failed"
else
    echo "Skipping: visualize_ratio_ablation.py not found"
fi

# Run extended training analysis
echo ""
echo "=========================================="
echo "Running Extended Training Analysis"
echo "=========================================="
python analysis_scripts/analyze_extended_training.py --detailed || python analysis_scripts/analyze_extended_training.py

# Generate extended training visualizations
echo ""
echo "=========================================="
echo "Generating Extended Training Figures"
echo "=========================================="
if [ -f "analysis_scripts/visualize_extended_training.py" ]; then
    python analysis_scripts/visualize_extended_training.py || echo "Warning: Visualization failed"
else
    echo "Skipping: visualize_extended_training.py not found"
fi

echo ""
echo -e "${GREEN}=========================================="
echo "Analysis Complete!"
echo "==========================================${NC}"
echo ""
echo "Output locations:"
echo "  - Ratio ablation figures: result_figures/ratio_ablation/"
echo "  - Extended training figures: result_figures/extended_training/"
echo ""
echo "Summary reports:"
echo "  - docs/RATIO_ABLATION_ANALYSIS.md"
echo "  - docs/EXTENDED_TRAINING_ANALYSIS.md"
