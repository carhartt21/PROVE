#!/bin/bash
# Cleanup script for buggy weights (models trained with incorrect label transforms)
# Affected datasets: bdd10k, idd-aw, mapillaryvistas
# 
# The bug: CityscapesLabelIdToTrainId was incorrectly applied to datasets already in trainId format
# This corrupted labels and models only learned 6 of 19 classes
#
# Total space to be freed: ~1.9 TB

set -e

WEIGHTS_DIR="/scratch/aaa_exchange/AWARE/WEIGHTS"
LOG_FILE="/home/mima2416/repositories/PROVE/logs/cleanup_buggy_weights_$(date +%Y%m%d_%H%M%S).log"

# Affected datasets
AFFECTED_DATASETS=("bdd10k" "idd-aw" "mapillaryvistas")

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to calculate directory size
get_size() {
    du -sh "$1" 2>/dev/null | awk '{print $1}'
}

# Create backup list before deletion
create_backup_list() {
    log "Creating backup list of directories to delete..."
    BACKUP_LIST="/home/mima2416/repositories/PROVE/logs/deleted_directories_$(date +%Y%m%d_%H%M%S).txt"
    
    for dataset in "${AFFECTED_DATASETS[@]}"; do
        for dir in "$WEIGHTS_DIR"/*/"$dataset"; do
            if [ -d "$dir" ]; then
                echo "$dir ($(get_size "$dir"))" >> "$BACKUP_LIST"
            fi
        done
    done
    
    log "Backup list saved to: $BACKUP_LIST"
}

# Show summary before deletion
show_summary() {
    log "=== CLEANUP SUMMARY ==="
    log "Affected datasets: ${AFFECTED_DATASETS[*]}"
    log ""
    
    total_dirs=0
    for dataset in "${AFFECTED_DATASETS[@]}"; do
        count=$(ls -d "$WEIGHTS_DIR"/*/"$dataset" 2>/dev/null | wc -l)
        total_dirs=$((total_dirs + count))
        log "$dataset: $count directories"
    done
    
    log ""
    log "Total directories to delete: $total_dirs"
    log "Estimated space to free: ~1.9 TB"
    log ""
}

# Perform deletion
perform_deletion() {
    log "=== STARTING DELETION ==="
    
    deleted_count=0
    for dataset in "${AFFECTED_DATASETS[@]}"; do
        log "Processing $dataset..."
        
        for dir in "$WEIGHTS_DIR"/*/"$dataset"; do
            if [ -d "$dir" ]; then
                size=$(get_size "$dir")
                log "Deleting: $dir ($size)"
                rm -rf "$dir"
                deleted_count=$((deleted_count + 1))
            fi
        done
    done
    
    log ""
    log "=== DELETION COMPLETE ==="
    log "Deleted $deleted_count directories"
}

# Dry run - show what would be deleted
dry_run() {
    echo "=== DRY RUN - No files will be deleted ==="
    echo ""
    
    for dataset in "${AFFECTED_DATASETS[@]}"; do
        echo "[$dataset]"
        for dir in "$WEIGHTS_DIR"/*/"$dataset"; do
            if [ -d "$dir" ]; then
                echo "  Would delete: $dir ($(get_size "$dir"))"
            fi
        done
        echo ""
    done
}

# Main script
main() {
    mkdir -p "$(dirname "$LOG_FILE")"
    
    case "${1:-}" in
        --dry-run)
            dry_run
            ;;
        --confirm)
            log "Starting cleanup of buggy weights..."
            show_summary
            create_backup_list
            perform_deletion
            log "Cleanup complete!"
            ;;
        *)
            echo "Usage: $0 [--dry-run|--confirm]"
            echo ""
            echo "Options:"
            echo "  --dry-run    Show what would be deleted without actually deleting"
            echo "  --confirm    Actually delete the buggy weights"
            echo ""
            echo "This script will delete weights for:"
            echo "  - bdd10k (~630 GB)"
            echo "  - idd-aw (~630 GB)"
            echo "  - mapillaryvistas (~650 GB)"
            echo ""
            echo "Total: ~1.9 TB"
            ;;
    esac
}

main "$@"
