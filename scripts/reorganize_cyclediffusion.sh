#!/bin/bash
# Script to reorganize cyclediffusion images from dataset/domain to domain/dataset structure
# Run this script with appropriate permissions

BASE="/scratch/aaa_exchange/AWARE/GENERATED_IMAGES/cyclediffusion"

echo "=== Reorganizing cyclediffusion images ==="
echo "Moving from dataset/domain to domain/dataset structure"
echo ""

# Function to move images
move_images() {
    local ds=$1
    local domain=$2
    local source="$BASE/$ds/$domain"
    local target="$BASE/$domain/$ds"
    
    if [ -d "$source" ] && [ "$(find "$source" -type f \( -name '*.png' -o -name '*.jpg' \) | wc -l)" -gt 0 ]; then
        echo "Moving: $ds/$domain -> $domain/$ds"
        
        # Create target directory if it doesn't exist
        mkdir -p "$target"
        
        # Move all files
        mv "$source"/* "$target/" 2>/dev/null
        
        # Count moved files
        count=$(find "$target" -type f \( -name '*.png' -o -name '*.jpg' \) | wc -l)
        echo "  Moved $count images"
    else
        echo "Skipping: $ds/$domain (empty or doesn't exist)"
    fi
}

# Move MapillaryVistas images
echo ""
echo "=== MapillaryVistas ==="
for domain in cloudy dawn_dusk foggy night rainy snowy; do
    move_images "MapillaryVistas" "$domain"
done

# Move OUTSIDE15k images  
echo ""
echo "=== OUTSIDE15k ==="
for domain in cloudy dawn_dusk foggy night rainy snowy; do
    move_images "OUTSIDE15k" "$domain"
done

# Cleanup - remove empty directories
echo ""
echo "=== Cleanup ==="
echo "Removing empty source directories..."

for ds in MapillaryVistas OUTSIDE15k; do
    for domain in cloudy dawn_dusk foggy night rainy snowy; do
        rmdir "$BASE/$ds/$domain" 2>/dev/null
    done
    rmdir "$BASE/$ds" 2>/dev/null && echo "Removed: $BASE/$ds"
done

echo ""
echo "=== Verification ==="
echo "Checking final structure..."

for domain in cloudy dawn_dusk foggy night rainy snowy; do
    for ds in MapillaryVistas OUTSIDE15k; do
        path="$BASE/$domain/$ds"
        if [ -d "$path" ]; then
            count=$(find "$path" -type f \( -name '*.png' -o -name '*.jpg' \) | wc -l)
            echo "$domain/$ds: $count images"
        fi
    done
done

echo ""
echo "Done!"
