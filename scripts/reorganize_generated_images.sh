#!/bin/bash
# Reorganize GENERATED_IMAGES from condition-first to dataset-first structure
# condition-first: method/<condition>/<dataset>/ → dataset-first: method/<dataset>/<condition>/
#
# Also integrates Cityscapes images that are already in dataset-first format.
#
# Usage:
#   bash scripts/reorganize_generated_images.sh --dry-run   # Preview only
#   bash scripts/reorganize_generated_images.sh              # Execute moves

set -euo pipefail

BASE="${AWARE_DATA_ROOT}/GENERATED_IMAGES"
DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE - No changes will be made ==="
    echo
fi

# Skip these directories per user request
SKIP_DIRS="step1x_new step1x_v1p2 qwen"

# Known weather conditions used across methods
ALL_CONDITIONS="cloudy dawn_dusk foggy night rainy snowy fog rain snow clouds snow_no_flakes bright dark fog_heavy fog_light gravel rain_drizzle rain_heavy rain_torrential shadow snow_heavy snow_light"

# Known dataset names
ALL_DATASETS="ACDC BDD100k BDD10k Cityscapes IDD-AW MapillaryVistas OUTSIDE15k"

move_dir() {
    local src="$1"
    local dst="$2"
    if [ -d "$src" ]; then
        if [ -d "$dst" ]; then
            echo "  WARNING: Destination already exists: $dst (merging contents)"
            if $DRY_RUN; then
                echo "  [DRY RUN] Would merge $src/* → $dst/"
            else
                # Move individual contents to avoid overwriting
                for item in "$src"/*; do
                    local item_name=$(basename "$item")
                    if [ ! -e "$dst/$item_name" ]; then
                        mv "$item" "$dst/"
                    else
                        echo "  WARNING: $dst/$item_name already exists, skipping"
                    fi
                done
                rmdir "$src" 2>/dev/null || echo "  Note: $src not empty after merge, keeping"
            fi
        else
            if $DRY_RUN; then
                echo "  [DRY RUN] mv $src → $dst"
            else
                mkdir -p "$(dirname "$dst")"
                mv "$src" "$dst"
                echo "  MOVED: $src → $dst"
            fi
        fi
    fi
}

reorganize_condition_first() {
    # Reorganize method/<condition>/<dataset>/ → method/<dataset>/<condition>/
    local method_dir="$1"
    local method_name=$(basename "$method_dir")
    
    echo "--- Processing: $method_name ---"
    
    # Find all condition directories that contain dataset subdirectories
    for entry in "$method_dir"/*/; do
        [ -d "$entry" ] || continue
        local entry_name=$(basename "$entry")
        
        # Skip manifest files, Cityscapes (already dataset-first), and non-condition dirs
        case "$entry_name" in
            manifest*|reorganize*|*.csv|*.json|*.txt|*.backup*) continue ;;
        esac
        
        # Check if this is a condition dir containing dataset subdirs
        local is_condition_with_datasets=false
        for ds in $ALL_DATASETS; do
            if [ -d "$entry/$ds" ]; then
                is_condition_with_datasets=true
                break
            fi
        done
        
        if $is_condition_with_datasets; then
            echo "  Condition dir (with dataset subdirs): $entry_name"
            # Move each dataset from condition/dataset → dataset/condition
            for ds_dir in "$entry"/*/; do
                [ -d "$ds_dir" ] || continue
                local ds_name=$(basename "$ds_dir")
                local target="$method_dir/$ds_name/$entry_name"
                move_dir "$ds_dir" "$target"
            done
            # Remove empty condition dir
            if ! $DRY_RUN; then
                rmdir "$entry" 2>/dev/null && echo "  Removed empty dir: $entry_name" || true
            else
                echo "  [DRY RUN] Would remove empty dir: $entry_name (if empty)"
            fi
        else
            # Check if it's already a dataset dir (contains condition subdirs)
            local is_dataset_with_conditions=false
            for item in "$entry"/*/; do
                [ -d "$item" ] || continue
                local item_name=$(basename "$item")
                # Check if subdirs look like conditions
                case "$item_name" in
                    cloudy|dawn_dusk|foggy|night|rainy|snowy|fog|rain|snow|clouds|snow_no_flakes|bright|dark|fog_heavy|fog_light|gravel|rain_drizzle|rain_heavy|rain_torrential|shadow|snow_heavy|snow_light|sunny_day2*|clear_day_to_*|clear_day2*|aware_2_*)
                        is_dataset_with_conditions=true
                        break
                        ;;
                esac
            done
            
            if $is_dataset_with_conditions; then
                echo "  Dataset dir (already correct): $entry_name"
            else
                echo "  Unknown dir type: $entry_name (checking contents...)"
                local first_item=$(ls "$entry" 2>/dev/null | head -1)
                echo "    First item: $first_item"
            fi
        fi
    done
    echo
}

process_special_conditions() {
    # Handle dirs with non-standard condition names (CNetSeg, SUSTechGAN, TSIT)
    local method_dir="$1"
    local method_name=$(basename "$method_dir")
    
    echo "--- Processing special: $method_name ---"
    
    # For methods with special condition names, check if Cityscapes/<special_cond>/ exists
    if [ -d "$method_dir/Cityscapes" ]; then
        for cond_dir in "$method_dir/Cityscapes"/*/; do
            [ -d "$cond_dir" ] || continue
            local cond_name=$(basename "$cond_dir")
            echo "  Cityscapes/$cond_name already in dataset-first format - OK"
        done
    fi
    echo
}

echo "============================================"
echo "Reorganizing GENERATED_IMAGES to dataset-first structure"
echo "============================================"
echo

TOTAL_MOVES=0

# Process each chge7185-owned directory
for method_dir in "$BASE"/*/; do
    [ -d "$method_dir" ] || continue
    method_name=$(basename "$method_dir")
    
    # Skip specified directories
    skip=false
    for skip_dir in $SKIP_DIRS; do
        if [[ "$method_name" == "$skip_dir" ]]; then
            skip=true
            break
        fi
    done
    if $skip; then
        echo "SKIPPING: $method_name (per user request)"
        echo
        continue
    fi
    
    # Check if owned by chge7185
    owner=$(stat -c '%U' "$method_dir" 2>/dev/null || echo "unknown")
    if [[ "$owner" != "chge7185" ]]; then
        echo "SKIPPING: $method_name (owned by $owner, not chge7185)"
        echo
        continue
    fi
    
    reorganize_condition_first "$method_dir"
done

if $DRY_RUN; then
    echo "=== DRY RUN COMPLETE - No changes were made ==="
    echo "Run without --dry-run to execute the moves."
fi
