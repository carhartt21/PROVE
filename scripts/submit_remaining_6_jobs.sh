#!/bin/bash
# Submit remaining 6 retraining jobs
# These are:
#   - retrain_gen_step1x_new.sh
#   - retrain_photometric_distort.sh
#   - retrain_std_autoaugment.sh
#   - retrain_std_cutmix.sh
#   - retrain_std_mixup.sh
#   - retrain_std_randaugment.sh
#
# Usage: ./submit_remaining_6_jobs.sh
#        Or submit individually: bsub < /path/to/script.sh

SCRIPTS_DIR="/home/mima2416/repositories/PROVE/scripts/retrain_jobs"

REMAINING_SCRIPTS=(
    "retrain_gen_step1x_new.sh"
    "retrain_photometric_distort.sh"
    "retrain_std_autoaugment.sh"
    "retrain_std_cutmix.sh"
    "retrain_std_mixup.sh"
    "retrain_std_randaugment.sh"
)

echo "========================================"
echo "Submitting 6 remaining retraining jobs"
echo "========================================"
echo ""

submitted=0
for script in "${REMAINING_SCRIPTS[@]}"; do
    script_path="${SCRIPTS_DIR}/${script}"
    
    if [[ -f "$script_path" ]]; then
        echo "Submitting: $script"
        bsub < "$script_path"
        ((submitted++))
    else
        echo "WARNING: Script not found: $script_path"
    fi
done

echo ""
echo "========================================"
echo "Submitted $submitted jobs"
echo "========================================"
echo ""
echo "To check job status:"
echo "  bjobs"
echo ""
echo "To check specific job:"
echo "  bjobs <job_id>"
echo ""
echo "To view logs:"
echo "  tail -f /home/mima2416/repositories/PROVE/logs/retrain/retrain_*_<job_id>.out"
