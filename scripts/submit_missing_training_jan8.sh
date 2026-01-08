#!/bin/bash
# ============================================================================
# PROVE: Submit Missing Training Jobs (Without A100 Requirement)
# ============================================================================
# This script resubmits all the training jobs that were pending due to 
# A100 GPU availability. It removes the gmodel=NVIDIAA100_PCIE_40GB requirement.
#
# Jobs include:
#   - gen_augmenters: 15 jobs
#   - gen_VisualCloze: 18 jobs  
#   - gen_Qwen_Image_Edit: 4 jobs
#   - gen_albumentations_weather: 18 jobs
#   - gen_cyclediffusion: 12 jobs
#   - gen_step1x_v1p2: 18 jobs
#   - gen_flux_kontext: 6 jobs
#   - gen_StyleID (resume): 2 jobs
#   - photometric_distort (resume): 1 job
#
# Usage:
#   ./scripts/submit_missing_training_jan8.sh [--dry-run]
#
# ============================================================================

set -e

DRY_RUN=false
WEIGHTS_DIR="/scratch/aaa_exchange/AWARE/WEIGHTS"
DATA_ROOT="/scratch/aaa_exchange/AWARE/FINAL_SPLITS"
QUEUE="BatchGPU"
GPU_MEM="16G"
MAX_TIME="24:00"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

submit_job() {
    local JOB_NAME="$1"
    local TRAIN_CMD="$2"
    
    # No A100 requirement - just request any GPU with enough memory
    local SUBMIT_CMD="bsub -J \"${JOB_NAME}\" -q ${QUEUE} -n 4 -gpu \"num=1:mode=exclusive_process:gmem=${GPU_MEM}\" -W ${MAX_TIME} -o logs/${JOB_NAME}_%J.log -e logs/${JOB_NAME}_%J.err \"${TRAIN_CMD}\""
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY RUN] Would submit: ${JOB_NAME}"
    else
        echo "Submitting: ${JOB_NAME}"
        eval "$SUBMIT_CMD"
    fi
}

echo "========================================================================"
echo "PROVE Missing Training Jobs Submission (Without A100 Requirement)"
echo "========================================================================"
echo ""
echo "Dry Run: $DRY_RUN"
echo "Queue: $QUEUE"
echo "GPU Memory: $GPU_MEM (any GPU)"
echo "Max Time: $MAX_TIME"
echo ""

submitted=0

# ============================================================================
# gen_augmenters: 15 jobs
# ============================================================================
echo ""
echo "======== gen_augmenters (15 jobs) ========"
STRATEGY="gen_augmenters"

for dataset in "BDD10k" "IDD-AW" "MapillaryVistas"; do
    for model in "deeplabv3plus_r50" "pspnet_r50" "segformer_mit-b5"; do
        for variant in "" "clear_day"; do
            # Skip BDD10k deeplabv3plus_r50 (already exists)
            if [[ "$dataset" == "BDD10k" && "$model" == "deeplabv3plus_r50" ]]; then
                continue
            fi
            
            if [[ -n "$variant" ]]; then
                JOB_NAME="prove_${dataset}_${model}_${variant}_${STRATEGY}"
                TRAIN_CMD="mamba run -n prove python unified_training.py --dataset ${dataset} --model ${model} --strategy ${STRATEGY} --domain-filter ${variant} --ratio 1.0 --work-dir ${WEIGHTS_DIR} --data-root ${DATA_ROOT} --no-early-stop"
            else
                JOB_NAME="prove_${dataset}_${model}_${STRATEGY}"
                TRAIN_CMD="mamba run -n prove python unified_training.py --dataset ${dataset} --model ${model} --strategy ${STRATEGY} --ratio 1.0 --work-dir ${WEIGHTS_DIR} --data-root ${DATA_ROOT} --no-early-stop"
            fi
            submit_job "$JOB_NAME" "$TRAIN_CMD"
            ((submitted++)) || true
        done
    done
done

# ============================================================================
# gen_VisualCloze: 18 jobs (all datasets, all models, with and without clear_day)
# ============================================================================
echo ""
echo "======== gen_VisualCloze (18 jobs) ========"
STRATEGY="gen_VisualCloze"

for dataset in "BDD10k" "IDD-AW" "MapillaryVistas"; do
    for model in "deeplabv3plus_r50" "pspnet_r50" "segformer_mit-b5"; do
        for variant in "" "clear_day"; do
            if [[ -n "$variant" ]]; then
                JOB_NAME="prove_${dataset}_${model}_${variant}_${STRATEGY}"
                TRAIN_CMD="source ~/.bashrc && conda activate prove && python unified_training.py --strategy ${STRATEGY} --datasets ${dataset} --architectures ${model} --work-dir ${WEIGHTS_DIR} --data-root ${DATA_ROOT} --domain-filter ${variant}"
            else
                JOB_NAME="prove_${dataset}_${model}_${STRATEGY}"
                TRAIN_CMD="source ~/.bashrc && conda activate prove && python unified_training.py --strategy ${STRATEGY} --datasets ${dataset} --architectures ${model} --work-dir ${WEIGHTS_DIR} --data-root ${DATA_ROOT}"
            fi
            submit_job "$JOB_NAME" "$TRAIN_CMD"
            ((submitted++)) || true
        done
    done
done

# ============================================================================
# gen_Qwen_Image_Edit: 4 jobs (IDD-AW only)
# ============================================================================
echo ""
echo "======== gen_Qwen_Image_Edit (4 jobs) ========"
STRATEGY="gen_Qwen_Image_Edit"

for model in "deeplabv3plus_r50" "pspnet_r50"; do
    for variant in "" "clear_day"; do
        if [[ -n "$variant" ]]; then
            JOB_NAME="prove_IDD-AW_${model}_${variant}_${STRATEGY}"
            TRAIN_CMD="source ~/.bashrc && conda activate prove && python unified_training.py --strategy ${STRATEGY} --datasets IDD-AW --architectures ${model} --work-dir ${WEIGHTS_DIR} --data-root ${DATA_ROOT} --domain-filter ${variant}"
        else
            JOB_NAME="prove_IDD-AW_${model}_${STRATEGY}"
            TRAIN_CMD="source ~/.bashrc && conda activate prove && python unified_training.py --strategy ${STRATEGY} --datasets IDD-AW --architectures ${model} --work-dir ${WEIGHTS_DIR} --data-root ${DATA_ROOT}"
        fi
        submit_job "$JOB_NAME" "$TRAIN_CMD"
        ((submitted++)) || true
    done
done

# ============================================================================
# gen_albumentations_weather: 18 jobs
# ============================================================================
echo ""
echo "======== gen_albumentations_weather (18 jobs) ========"
STRATEGY="gen_albumentations_weather"

for dataset in "BDD10k" "IDD-AW" "MapillaryVistas"; do
    for model in "deeplabv3plus_r50" "pspnet_r50" "segformer_mit-b5"; do
        for variant in "" "clear_day"; do
            if [[ -n "$variant" ]]; then
                JOB_NAME="prove_${dataset}_${model}_${variant}_${STRATEGY}"
                TRAIN_CMD="source ~/.bashrc && conda activate prove && python unified_training.py --strategy ${STRATEGY} --datasets ${dataset} --architectures ${model} --work-dir ${WEIGHTS_DIR} --data-root ${DATA_ROOT} --domain-filter ${variant}"
            else
                JOB_NAME="prove_${dataset}_${model}_${STRATEGY}"
                TRAIN_CMD="source ~/.bashrc && conda activate prove && python unified_training.py --strategy ${STRATEGY} --datasets ${dataset} --architectures ${model} --work-dir ${WEIGHTS_DIR} --data-root ${DATA_ROOT}"
            fi
            submit_job "$JOB_NAME" "$TRAIN_CMD"
            ((submitted++)) || true
        done
    done
done

# ============================================================================
# gen_cyclediffusion: 12 jobs (BDD10k and IDD-AW)
# ============================================================================
echo ""
echo "======== gen_cyclediffusion (12 jobs) ========"
STRATEGY="gen_cyclediffusion"

for dataset in "BDD10k" "IDD-AW"; do
    for model in "deeplabv3plus_r50" "pspnet_r50" "segformer_mit-b5"; do
        for variant in "" "clear_day"; do
            if [[ -n "$variant" ]]; then
                JOB_NAME="prove_${dataset}_${model}_${variant}_${STRATEGY}"
                TRAIN_CMD="source ~/.bashrc && conda activate prove && python unified_training.py --strategy ${STRATEGY} --datasets ${dataset} --architectures ${model} --work-dir ${WEIGHTS_DIR} --data-root ${DATA_ROOT} --domain-filter ${variant}"
            else
                JOB_NAME="prove_${dataset}_${model}_${STRATEGY}"
                TRAIN_CMD="source ~/.bashrc && conda activate prove && python unified_training.py --strategy ${STRATEGY} --datasets ${dataset} --architectures ${model} --work-dir ${WEIGHTS_DIR} --data-root ${DATA_ROOT}"
            fi
            submit_job "$JOB_NAME" "$TRAIN_CMD"
            ((submitted++)) || true
        done
    done
done

# ============================================================================
# gen_step1x_v1p2: 18 jobs
# ============================================================================
echo ""
echo "======== gen_step1x_v1p2 (18 jobs) ========"
STRATEGY="gen_step1x_v1p2"

for dataset in "BDD10k" "IDD-AW" "MapillaryVistas"; do
    for model in "deeplabv3plus_r50" "pspnet_r50" "segformer_mit-b5"; do
        for variant in "" "clear_day"; do
            if [[ -n "$variant" ]]; then
                JOB_NAME="prove_${dataset}_${model}_${variant}_${STRATEGY}"
                TRAIN_CMD="source ~/.bashrc && conda activate prove && python unified_training.py --strategy ${STRATEGY} --datasets ${dataset} --architectures ${model} --work-dir ${WEIGHTS_DIR} --data-root ${DATA_ROOT} --domain-filter ${variant}"
            else
                JOB_NAME="prove_${dataset}_${model}_${STRATEGY}"
                TRAIN_CMD="source ~/.bashrc && conda activate prove && python unified_training.py --strategy ${STRATEGY} --datasets ${dataset} --architectures ${model} --work-dir ${WEIGHTS_DIR} --data-root ${DATA_ROOT}"
            fi
            submit_job "$JOB_NAME" "$TRAIN_CMD"
            ((submitted++)) || true
        done
    done
done

# ============================================================================
# gen_flux_kontext: 6 jobs (MapillaryVistas only)
# ============================================================================
echo ""
echo "======== gen_flux_kontext (6 jobs) ========"
STRATEGY="gen_flux_kontext"

for model in "deeplabv3plus_r50" "pspnet_r50" "segformer_mit-b5"; do
    for variant in "" "clear_day"; do
        if [[ -n "$variant" ]]; then
            JOB_NAME="prove_MapillaryVistas_${model}_${variant}_${STRATEGY}"
            TRAIN_CMD="source ~/.bashrc && conda activate prove && python unified_training.py --strategy ${STRATEGY} --datasets MapillaryVistas --architectures ${model} --work-dir ${WEIGHTS_DIR} --data-root ${DATA_ROOT} --domain-filter ${variant}"
        else
            JOB_NAME="prove_MapillaryVistas_${model}_${STRATEGY}"
            TRAIN_CMD="source ~/.bashrc && conda activate prove && python unified_training.py --strategy ${STRATEGY} --datasets MapillaryVistas --architectures ${model} --work-dir ${WEIGHTS_DIR} --data-root ${DATA_ROOT}"
        fi
        submit_job "$JOB_NAME" "$TRAIN_CMD"
        ((submitted++)) || true
    done
done

# ============================================================================
# Resume jobs: gen_StyleID and photometric_distort
# ============================================================================
echo ""
echo "======== Resume Jobs (3 jobs) ========"

# gen_StyleID resume - BDD10k pspnet_r50
if [[ -f "${WEIGHTS_DIR}/gen_StyleID/bdd10k/pspnet_r50/iter_60000.pth" ]]; then
    JOB_NAME="prove_resume_BDD10k_pspnet_r50_gen_StyleID"
    TRAIN_CMD="mamba run -n prove python unified_training.py --dataset BDD10k --model pspnet_r50 --strategy gen_StyleID --resume-from ${WEIGHTS_DIR}/gen_StyleID/bdd10k/pspnet_r50/iter_60000.pth --ratio 1.0 --work-dir ${WEIGHTS_DIR} --data-root ${DATA_ROOT} --no-early-stop"
    submit_job "$JOB_NAME" "$TRAIN_CMD"
    ((submitted++)) || true
fi

# gen_StyleID resume - IDD-AW deeplabv3plus_r50 clear_day
if [[ -f "${WEIGHTS_DIR}/gen_StyleID/idd-aw/deeplabv3plus_r50_clear_day/iter_50000.pth" ]]; then
    JOB_NAME="prove_resume_IDD-AW_deeplabv3plus_r50_clear_day_gen_StyleID"
    TRAIN_CMD="mamba run -n prove python unified_training.py --dataset IDD-AW --model deeplabv3plus_r50 --strategy gen_StyleID --domain-filter clear_day --resume-from ${WEIGHTS_DIR}/gen_StyleID/idd-aw/deeplabv3plus_r50_clear_day/iter_50000.pth --ratio 1.0 --work-dir ${WEIGHTS_DIR} --data-root ${DATA_ROOT} --no-early-stop"
    submit_job "$JOB_NAME" "$TRAIN_CMD"
    ((submitted++)) || true
fi

# photometric_distort resume - MapillaryVistas deeplabv3plus_r50 clear_day
if [[ -f "${WEIGHTS_DIR}/photometric_distort/mapillaryvistas/deeplabv3plus_r50_clear_day/iter_50000.pth" ]]; then
    JOB_NAME="prove_resume_MapillaryVistas_deeplabv3plus_r50_clear_day_photometric_distort"
    TRAIN_CMD="mamba run -n prove python unified_training.py --dataset MapillaryVistas --model deeplabv3plus_r50 --strategy photometric_distort --domain-filter clear_day --resume-from ${WEIGHTS_DIR}/photometric_distort/mapillaryvistas/deeplabv3plus_r50_clear_day/iter_50000.pth --ratio 1.0 --work-dir ${WEIGHTS_DIR} --data-root ${DATA_ROOT} --no-early-stop"
    submit_job "$JOB_NAME" "$TRAIN_CMD"
    ((submitted++)) || true
fi

echo ""
echo "========================================================================"
echo "Submission complete!"
echo "Total jobs submitted: $submitted"
echo "========================================================================"
