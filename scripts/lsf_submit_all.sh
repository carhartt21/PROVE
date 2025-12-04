#!/bin/bash
# PROVE: Submit all LSF training jobs
# Dataset: /scratch/aaa_exchange/AWARE/FINAL_SPLITS/train/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo "PROVE: Submitting All LSF Jobs"
echo "============================================"

# Create logs directory
mkdir -p logs

# Submit Cityscapes training
JOB1=$(bsub < ${SCRIPT_DIR}/lsf_train_cityscapes.sh | awk '{print $2}' | tr -d '<>')
echo "Submitted Cityscapes training: Job ${JOB1}"

# Submit Mapillary training
JOB2=$(bsub < ${SCRIPT_DIR}/lsf_train_mapillary.sh | awk '{print $2}' | tr -d '<>')
echo "Submitted Mapillary training: Job ${JOB2}"

# Submit Joint training (Cityscapes labels)
JOB3=$(bsub < ${SCRIPT_DIR}/lsf_train_joint.sh | awk '{print $2}' | tr -d '<>')
echo "Submitted Joint (CS) training: Job ${JOB3}"

# Submit Joint training (Unified labels) - modify config inline
JOB4=$(bsub -J prove_joint_unified < ${SCRIPT_DIR}/lsf_train_joint.sh | awk '{print $2}' | tr -d '<>')
echo "Submitted Joint (Unified) training: Job ${JOB4}"

echo ""
echo "============================================"
echo "All jobs submitted!"
echo "Monitor with: bjobs -u \$USER"
echo "============================================"
