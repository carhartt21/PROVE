#!/bin/bash
# PROVE: Submit all AWARE dataset LSF training jobs
# Dataset: /scratch/aaa_exchange/AWARE/FINAL_SPLITS/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo "PROVE: Submitting All AWARE Dataset Jobs"
echo "============================================"

# Create logs directory
mkdir -p logs

# Submit ACDC training
JOB1=$(bsub < ${SCRIPT_DIR}/lsf_train_acdc.sh | awk '{print $2}' | tr -d '<>')
echo "Submitted ACDC training: Job ${JOB1}"

# Submit BDD10k training
JOB2=$(bsub < ${SCRIPT_DIR}/lsf_train_bdd10k.sh | awk '{print $2}' | tr -d '<>')
echo "Submitted BDD10k training: Job ${JOB2}"

# Submit BDD100k training
JOB3=$(bsub < ${SCRIPT_DIR}/lsf_train_bdd100k.sh | awk '{print $2}' | tr -d '<>')
echo "Submitted BDD100k training: Job ${JOB3}"

# Submit OUTSIDE15k training
JOB4=$(bsub < ${SCRIPT_DIR}/lsf_train_outside15k.sh | awk '{print $2}' | tr -d '<>')
echo "Submitted OUTSIDE15k training: Job ${JOB4}"

# Submit IDD-AW training
JOB5=$(bsub < ${SCRIPT_DIR}/lsf_train_iddaw.sh | awk '{print $2}' | tr -d '<>')
echo "Submitted IDD-AW training: Job ${JOB5}"

# Submit MapillaryVistas (AWARE) training
JOB6=$(bsub < ${SCRIPT_DIR}/lsf_train_mapillary_aware.sh | awk '{print $2}' | tr -d '<>')
echo "Submitted MapillaryVistas (AWARE) training: Job ${JOB6}"

# Submit Joint ALL datasets training
JOB7=$(bsub < ${SCRIPT_DIR}/lsf_train_joint_all.sh | awk '{print $2}' | tr -d '<>')
echo "Submitted Joint ALL training: Job ${JOB7}"

echo ""
echo "============================================"
echo "All jobs submitted!"
echo "Monitor with: bjobs -u \$USER"
echo "============================================"
