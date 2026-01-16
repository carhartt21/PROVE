#!/bin/bash
# Repository Cleanup Script
# Removes outdated one-time use scripts and generated job directories

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# =============================================================================
# ONE-TIME USE PYTHON SCRIPTS (bug fixes, one-time operations)
# =============================================================================
ONE_TIME_PYTHON_SCRIPTS=(
    "delete_wrong_mapillary_models.py"     # One-time cleanup of wrong models
    "find_wrong_training.py"                # One-time diagnostic tool
    "generate_mapillary_retrain_jobs.py"   # Generated retrain jobs - completed
    "reeval_native_class_models.py"         # One-time re-evaluation
    "retrain_affected_models.py"            # Main retraining driver - completed
    "retrain_iddaw_fixed_labels.py"         # One-time IDD-AW retrain
    "retest_bdd10k_fine_grained.py"         # One-time BDD10k retest
    "generate_retest_scripts.py"            # Old retest script generator
)

# =============================================================================
# ONE-TIME USE SHELL SCRIPTS
# =============================================================================
ONE_TIME_SHELL_SCRIPTS=(
    "background_monitor.sh"                 # One-time monitoring setup
    "jobs_reeval_native_classes.sh"         # Generated - one-time
    "cleanup_old_scripts.sh"                # Previous cleanup attempt
    "cleanup_onetime_scripts.sh"            # Previous cleanup attempt
    "submit_all_tests.sh"                   # One-time submission
    "submit_baseline_detailed_tests.sh"     # One-time submission
    "submit_missing_detailed_tests.sh"      # One-time submission
    "submit_missing_extended_training.sh"   # One-time submission
    "submit_missing_extended_training_tests.sh"  # One-time submission
    "submit_std_std_combinations.sh"        # Duplicate/old version
)

# =============================================================================
# GENERATED JOB DIRECTORIES (can be regenerated if needed)
# =============================================================================
GENERATED_DIRS=(
    "fix_outside15k_jobs"       # Generated fix jobs - completed
    "retrain_jobs"              # Generated retrain jobs
    "retrain_mapillary_jobs"    # Generated mapillary jobs
    "retest_jobs"               # Old retest jobs
    "retest_jobs_lsf"           # Generated LSF retest jobs - completed
    "ratio_ablation_jobs"       # Generated ratio ablation jobs
    "training_scripts"          # Generated training scripts
)

# =============================================================================
# ANALYSIS SCRIPTS TO REMOVE (deprecated/old versions)
# =============================================================================
ANALYSIS_SCRIPTS_TO_REMOVE=(
    "analysis_scripts/deprecated"  # Entire deprecated folder
)

# =============================================================================
# ROOT LEVEL FILES TO REMOVE (outdated)
# =============================================================================
ROOT_FILES_TO_REMOVE=(
    "pending_jobs.txt"          # Temporary tracking file
    "weights_summary.json"      # Generated summary - can be regenerated
)

# =============================================================================
# Functions
# =============================================================================

count_files() {
    local dir=$1
    if [[ -d "$dir" ]]; then
        find "$dir" -type f 2>/dev/null | wc -l
    else
        echo "0"
    fi
}

dry_run() {
    echo -e "${YELLOW}=== DRY RUN - No files will be deleted ===${NC}"
    echo ""
    
    total_files=0
    
    echo -e "${GREEN}One-time Python scripts to remove:${NC}"
    for script in "${ONE_TIME_PYTHON_SCRIPTS[@]}"; do
        if [[ -f "$SCRIPT_DIR/$script" ]]; then
            echo "  - scripts/$script"
            total_files=$((total_files + 1))
        fi
    done
    
    echo ""
    echo -e "${GREEN}One-time Shell scripts to remove:${NC}"
    for script in "${ONE_TIME_SHELL_SCRIPTS[@]}"; do
        if [[ -f "$SCRIPT_DIR/$script" ]]; then
            echo "  - scripts/$script"
            total_files=$((total_files + 1))
        fi
    done
    
    echo ""
    echo -e "${GREEN}Generated job directories to remove:${NC}"
    for dir in "${GENERATED_DIRS[@]}"; do
        if [[ -d "$SCRIPT_DIR/$dir" ]]; then
            count=$(count_files "$SCRIPT_DIR/$dir")
            echo "  - scripts/$dir/ ($count files)"
            total_files=$((total_files + count))
        fi
    done
    
    echo ""
    echo -e "${GREEN}Analysis scripts to remove:${NC}"
    for item in "${ANALYSIS_SCRIPTS_TO_REMOVE[@]}"; do
        if [[ -e "$PROJECT_ROOT/$item" ]]; then
            if [[ -d "$PROJECT_ROOT/$item" ]]; then
                count=$(count_files "$PROJECT_ROOT/$item")
                echo "  - $item/ ($count files)"
                total_files=$((total_files + count))
            else
                echo "  - $item"
                total_files=$((total_files + 1))
            fi
        fi
    done
    
    echo ""
    echo -e "${GREEN}Root level files to remove:${NC}"
    for file in "${ROOT_FILES_TO_REMOVE[@]}"; do
        if [[ -f "$PROJECT_ROOT/$file" ]]; then
            echo "  - $file"
            total_files=$((total_files + 1))
        fi
    done
    
    echo ""
    echo -e "${YELLOW}Total: ~$total_files files would be removed${NC}"
}

delete_files() {
    echo -e "${RED}=== DELETING FILES ===${NC}"
    echo ""
    
    # Python scripts
    for script in "${ONE_TIME_PYTHON_SCRIPTS[@]}"; do
        if [[ -f "$SCRIPT_DIR/$script" ]]; then
            rm -f "$SCRIPT_DIR/$script"
            echo "Deleted: scripts/$script"
        fi
    done
    
    # Shell scripts
    for script in "${ONE_TIME_SHELL_SCRIPTS[@]}"; do
        if [[ -f "$SCRIPT_DIR/$script" ]]; then
            rm -f "$SCRIPT_DIR/$script"
            echo "Deleted: scripts/$script"
        fi
    done
    
    # Generated directories
    for dir in "${GENERATED_DIRS[@]}"; do
        if [[ -d "$SCRIPT_DIR/$dir" ]]; then
            rm -rf "$SCRIPT_DIR/$dir"
            echo "Deleted: scripts/$dir/"
        fi
    done
    
    # Analysis scripts
    for item in "${ANALYSIS_SCRIPTS_TO_REMOVE[@]}"; do
        if [[ -e "$PROJECT_ROOT/$item" ]]; then
            rm -rf "$PROJECT_ROOT/$item"
            echo "Deleted: $item"
        fi
    done
    
    # Root files
    for file in "${ROOT_FILES_TO_REMOVE[@]}"; do
        if [[ -f "$PROJECT_ROOT/$file" ]]; then
            rm -f "$PROJECT_ROOT/$file"
            echo "Deleted: $file"
        fi
    done
    
    # Clean __pycache__ directories
    find "$PROJECT_ROOT" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    echo "Cleaned: __pycache__ directories"
    
    echo ""
    echo -e "${GREEN}Cleanup complete!${NC}"
}

archive_files() {
    local archive_dir="$PROJECT_ROOT/archived_scripts_$(date +%Y%m%d)"
    mkdir -p "$archive_dir"
    
    echo -e "${YELLOW}=== ARCHIVING FILES to $archive_dir ===${NC}"
    echo ""
    
    # Python scripts
    for script in "${ONE_TIME_PYTHON_SCRIPTS[@]}"; do
        if [[ -f "$SCRIPT_DIR/$script" ]]; then
            mv "$SCRIPT_DIR/$script" "$archive_dir/"
            echo "Archived: scripts/$script"
        fi
    done
    
    # Shell scripts
    for script in "${ONE_TIME_SHELL_SCRIPTS[@]}"; do
        if [[ -f "$SCRIPT_DIR/$script" ]]; then
            mv "$SCRIPT_DIR/$script" "$archive_dir/"
            echo "Archived: scripts/$script"
        fi
    done
    
    # Generated directories
    for dir in "${GENERATED_DIRS[@]}"; do
        if [[ -d "$SCRIPT_DIR/$dir" ]]; then
            mv "$SCRIPT_DIR/$dir" "$archive_dir/"
            echo "Archived: scripts/$dir/"
        fi
    done
    
    echo ""
    echo -e "${GREEN}Files archived to: $archive_dir${NC}"
    echo "You can delete the archive later with: rm -rf $archive_dir"
}

usage() {
    echo "Usage: $0 [--dry-run|--delete|--archive]"
    echo ""
    echo "Options:"
    echo "  --dry-run   Show what would be deleted without making changes"
    echo "  --delete    Permanently delete files"
    echo "  --archive   Move files to archived_scripts_YYYYMMDD/ directory"
    echo ""
    echo "Without arguments, runs in dry-run mode."
}

# =============================================================================
# Main
# =============================================================================

case "${1:-}" in
    --dry-run|"")
        dry_run
        ;;
    --delete)
        read -p "Are you sure you want to delete these files? [y/N] " confirm
        if [[ "$confirm" =~ ^[Yy]$ ]]; then
            delete_files
        else
            echo "Aborted."
        fi
        ;;
    --archive)
        archive_files
        ;;
    --help|-h)
        usage
        ;;
    *)
        echo "Unknown option: $1"
        usage
        exit 1
        ;;
esac
