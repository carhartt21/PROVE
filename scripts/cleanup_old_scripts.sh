#!/bin/bash
# =============================================================================
# PROVE Scripts Cleanup
# Generated: January 9, 2026
#
# This script removes outdated one-time scripts that are no longer needed.
# Run with --dry-run first to preview what will be deleted.
#
# Usage:
#   ./cleanup_old_scripts.sh --dry-run   # Preview deletions
#   ./cleanup_old_scripts.sh --delete    # Actually delete files
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

DRY_RUN=true
DELETE=false

# Parse arguments
case "${1:-}" in
    --dry-run)
        DRY_RUN=true
        DELETE=false
        ;;
    --delete)
        DRY_RUN=false
        DELETE=true
        ;;
    *)
        echo "Usage: $0 [--dry-run|--delete]"
        echo ""
        echo "Options:"
        echo "  --dry-run   Preview what would be deleted (default)"
        echo "  --delete    Actually delete the files"
        exit 1
        ;;
esac

# Scripts to delete (script:reason format)
SCRIPTS=(
    # Date-specific one-time submissions
    "submit_detailed_tests_jan8.sh:Date-specific (Jan 8) one-time test submission"
    "submit_missing_training_jan8.sh:Date-specific (Jan 8) one-time training submission"
    
    # "Remaining/specific count" scripts
    "submit_remaining_6_jobs.sh:One-time: Submit exactly 6 specific jobs"
    
    # One-time reorganization/cleanup
    "reorganize_cyclediffusion.sh:One-time directory restructure (dataset/domain -> domain/dataset)"
    "cleanup_buggy_weights.sh:One-time cleanup for label transform bug (already executed)"
    
    # Auto-generated mass resubmissions
    "resubmit_training_without_a100.sh:Auto-generated one-time resubmit (1938 lines of hardcoded jobs)"
    
    # One-time dataset-specific submissions
    "submit_additional_training_jobs.sh:One-time: 18 jobs for gen_albumentations_weather + gen_cyclediffusion"
    "submit_flux_kontext_training.sh:One-time: gen_flux_kontext MapillaryVistas (6 jobs)"
    "submit_gen_augmenters_training.sh:One-time: Missing gen_augmenters training"
    "submit_idd_aw_qwen_training.sh:One-time: IDD-AW gen_Qwen_Image_Edit (failed Dec 22)"
    "submit_incomplete_training.sh:One-time: Resubmit interrupted training jobs"
    "submit_missing_extended_training.sh:One-time: 33 specific missing extended training jobs"
    "submit_missing_extended_training_tests.sh:One-time: 3 specific missing test jobs"
    "submit_missing_training_jobs.sh:One-time: Specific BDD10k and gen_step1x_v1p2 jobs"
    "submit_visualcloze_training.sh:One-time: gen_VisualCloze training submission"
    
    # Broken/obsolete template
    "submit_lsf_job.sh:Obsolete template with broken syntax"
)

echo "=============================================="
if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}DRY RUN - Preview of scripts to delete${NC}"
else
    echo -e "${RED}DELETING scripts${NC}"
fi
echo "=============================================="
echo ""

deleted=0
skipped=0

for entry in "${SCRIPTS[@]}"; do
    script="${entry%%:*}"
    reason="${entry#*:}"
    filepath="$SCRIPT_DIR/$script"
    
    if [ -f "$filepath" ]; then
        if [ "$DELETE" = true ]; then
            echo -e "${RED}DELETE${NC}: $script"
            echo "        Reason: $reason"
            rm "$filepath"
            ((deleted++)) || true
        else
            echo -e "${YELLOW}WOULD DELETE${NC}: $script"
            echo "        Reason: $reason"
            ((deleted++)) || true
        fi
    else
        echo -e "${GREEN}SKIP${NC}: $script (not found)"
        ((skipped++)) || true
    fi
done

echo ""
echo "=============================================="
echo "Summary:"
if [ "$DRY_RUN" = true ]; then
    echo "  Would delete: $deleted scripts"
else
    echo "  Deleted: $deleted scripts"
fi
echo "  Skipped (not found): $skipped scripts"
echo "=============================================="

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo -e "${YELLOW}This was a dry run. To actually delete, run:${NC}"
    echo "  $0 --delete"
fi
