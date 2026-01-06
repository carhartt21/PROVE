#!/bin/bash
# Submit missing extended training jobs
# Run this from an HPC login node where bsub is available

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=============================================="
echo "Missing Extended Training Jobs Submission"
echo "=============================================="
echo ""
echo "This script will submit the following missing jobs:"
echo "  - gen_NST: 15 jobs (all datasets/models)"
echo "  - gen_flux1_kontext: 15 jobs (all datasets/models)"  
echo "  - gen_step1x_new/BDD10k: 3 jobs (all models)"
echo ""
echo "Total: 33 jobs"
echo ""

# Check if bsub is available
if ! command -v bsub &> /dev/null; then
    echo "ERROR: bsub command not found. Please run this from an HPC login node."
    exit 1
fi

read -p "Press Enter to continue or Ctrl+C to cancel..."

echo ""
echo "=== Submitting gen_NST jobs (15) ==="
"$SCRIPT_DIR/submit_extended_training.sh" --strategy gen_NST

echo ""
echo "=== Submitting gen_flux1_kontext jobs (15) ==="
"$SCRIPT_DIR/submit_extended_training.sh" --strategy gen_flux1_kontext

echo ""
echo "=== Submitting gen_step1x_new/BDD10k jobs (3) ==="
"$SCRIPT_DIR/submit_extended_training.sh" --strategy gen_step1x_new --dataset BDD10k

echo ""
echo "=============================================="
echo "All missing extended training jobs submitted!"
echo "=============================================="
echo ""
echo "Monitor with: bjobs -w"
