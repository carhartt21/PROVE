#!/bin/bash
# Submit all IDD-AW retraining jobs using LSF
# Run this from a node with bsub available (e.g., login node)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_DIR="$SCRIPT_DIR/iddaw_retrain_jobs"

echo "=============================================="
echo "IDD-AW Retraining Job Submission (LSF)"
echo "=============================================="
echo ""
echo "This will submit 33 jobs to retrain all IDD-AW models"
echo "with the fixed label handling."
echo ""
echo "Scripts directory: $JOB_DIR"
echo ""

# Check if bsub is available
if ! command -v bsub &> /dev/null; then
    echo "ERROR: bsub command not found."
    echo "Please run this script from a node with LSF access"
    exit 1
fi

# Count scripts
NUM_SCRIPTS=$(ls "$JOB_DIR"/*.sh 2>/dev/null | wc -l)
echo "Found $NUM_SCRIPTS job scripts"
echo ""

read -p "Do you want to submit all jobs? [y/N] " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Create logs directory
mkdir -p /home/chge7185/repositories/PROVE/logs/retrain

# Submit jobs
JOB_IDS=()
for script in "$JOB_DIR"/*.sh; do
    echo "Submitting: $(basename "$script")"
    result=$(bsub < "$script" 2>&1)
    if [[ $result =~ "is submitted" ]]; then
        job_id=$(echo "$result" | grep -oP '<\K[0-9]+')
        JOB_IDS+=($job_id)
        echo "  -> Job ID: $job_id"
    else
        echo "  -> ERROR: $result"
    fi
done

echo ""
echo "=============================================="
echo "SUMMARY"
echo "=============================================="
echo "Submitted ${#JOB_IDS[@]} jobs"
if [ ${#JOB_IDS[@]} -gt 0 ]; then
    echo ""
    echo "Job IDs: ${JOB_IDS[*]}"
    echo ""
    echo "Monitor with: bjobs"
    echo "Logs in: logs/retrain/"
fi
