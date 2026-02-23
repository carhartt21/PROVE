#!/bin/bash
# Submit BDD10k fine-grained re-test jobs using LSF
# Run this script from a node with bsub access

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOBS_DIR="${SCRIPT_DIR}/bdd10k_retest_jobs"

echo "BDD10k Fine-Grained Re-Test Submission (LSF)"
echo "============================================="
echo ""

if [ ! -d "$JOBS_DIR" ]; then
    echo "Error: Jobs directory not found: $JOBS_DIR"
    echo "Please run: python scripts/retest_bdd10k_fine_grained.py"
    exit 1
fi

# Count jobs
JOB_COUNT=$(ls "$JOBS_DIR"/*.sh 2>/dev/null | wc -l)
echo "Found $JOB_COUNT job scripts in $JOBS_DIR"
echo ""

# Check if bsub is available
if ! command -v bsub &> /dev/null; then
    echo "Error: bsub not found. Are you on a node with LSF access?"
    echo ""
    echo "To submit jobs, run this script on a login node with LSF access."
    exit 1
fi

# Confirmation
read -p "Submit all $JOB_COUNT jobs? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Create logs directory
mkdir -p ${AWARE_DATA_ROOT}/LOGS/retest_bdd10k

# Submit jobs
echo ""
echo "Submitting jobs..."
SUBMITTED=0
for script in "$JOBS_DIR"/*.sh; do
    if [ -f "$script" ]; then
        JOB_NAME=$(basename "$script" .sh)
        if OUTPUT=$(bsub < "$script" 2>&1); then
            echo "  Submitted: $JOB_NAME"
            SUBMITTED=$((SUBMITTED + 1))
        else
            echo "  Failed: $JOB_NAME - $OUTPUT"
        fi
    fi
done

echo ""
echo "Submitted $SUBMITTED / $JOB_COUNT jobs"
echo ""
echo "Monitor with: bjobs"
echo "Logs will be in: ${AWARE_DATA_ROOT}/LOGS/retest_bdd10k/"
