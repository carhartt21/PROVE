#!/bin/bash
# Cleanup corrupted weights directories
# Created: 2026-01-15

set -e

WEIGHTS_ROOT="/scratch/aaa_exchange/AWARE/WEIGHTS"

echo "=========================================="
echo "Cleaning up corrupted WEIGHTS directories"
echo "=========================================="

# 1. Remove corrupted iddaw_ad directories (trained without domain filter when should have been Stage 1)
echo ""
echo "=== Removing corrupted iddaw_ad directories ==="
CORRUPTED_DIRS=(
    "baseline/iddaw_ad"
    "gen_Attribute_Hallucination/iddaw_ad"
    "gen_CNetSeg/iddaw_ad"
    "gen_CUT/iddaw_ad"
    "gen_flux_kontext/iddaw_ad"
    "gen_IP2P/iddaw_ad"
    "gen_Qwen_Image_Edit/iddaw_ad"
    "gen_TSIT/iddaw_ad"
    "gen_UniControl/iddaw_ad"
    "gen_VisualCloze/iddaw_ad"
    "std_minimal/iddaw_ad"
)

for dir in "${CORRUPTED_DIRS[@]}"; do
    full_path="${WEIGHTS_ROOT}/${dir}"
    if [ -d "$full_path" ]; then
        size=$(du -sh "$full_path" 2>/dev/null | cut -f1)
        echo "Removing: $full_path ($size)"
        rm -rf "$full_path"
    else
        echo "Already removed: $full_path"
    fi
done

echo ""
echo "=== Cleanup complete ==="
echo ""

# 2. Rename directories with incorrect naming (cd suffix missing)
echo "=== Renaming unsuffixed directories to proper _cd suffix ==="

# These directories were trained with domain_filter=clear_day but don't have _cd suffix
# We will rename them to have proper suffix

RENAME_PAIRS=(
    # gen_albumentations_weather
    "gen_albumentations_weather/bdd10k:gen_albumentations_weather/bdd10k_cd"
    "gen_albumentations_weather/idd-aw:gen_albumentations_weather/idd-aw_cd"
    "gen_albumentations_weather/mapillaryvistas:gen_albumentations_weather/mapillaryvistas_cd"
    "gen_albumentations_weather/outside15k:gen_albumentations_weather/outside15k_cd"
    # gen_Attribute_Hallucination
    "gen_Attribute_Hallucination/bdd10k:gen_Attribute_Hallucination/bdd10k_cd"
    # gen_augmenters
    "gen_augmenters/bdd10k:gen_augmenters/bdd10k_cd"
    "gen_augmenters/idd-aw:gen_augmenters/idd-aw_cd"
    "gen_augmenters/mapillaryvistas:gen_augmenters/mapillaryvistas_cd"
    "gen_augmenters/outside15k:gen_augmenters/outside15k_cd"
    # gen_automold
    "gen_automold/bdd10k:gen_automold/bdd10k_cd"
    "gen_automold/idd-aw:gen_automold/idd-aw_cd"
    "gen_automold/mapillaryvistas:gen_automold/mapillaryvistas_cd"
    "gen_automold/outside15k:gen_automold/outside15k_cd"
    # gen_CNetSeg
    "gen_CNetSeg/bdd10k:gen_CNetSeg/bdd10k_cd"
    "gen_CNetSeg/idd-aw:gen_CNetSeg/idd-aw_cd"
    "gen_CNetSeg/mapillaryvistas:gen_CNetSeg/mapillaryvistas_cd"
    "gen_CNetSeg/outside15k:gen_CNetSeg/outside15k_cd"
    # gen_CUT
    "gen_CUT/bdd10k:gen_CUT/bdd10k_cd"
    # gen_cycleGAN
    "gen_cycleGAN/bdd10k:gen_cycleGAN/bdd10k_cd"
    "gen_cycleGAN/idd-aw:gen_cycleGAN/idd-aw_cd"
    "gen_cycleGAN/mapillaryvistas:gen_cycleGAN/mapillaryvistas_cd"
    "gen_cycleGAN/outside15k:gen_cycleGAN/outside15k_cd"
    # gen_EDICT
    "gen_EDICT/bdd10k:gen_EDICT/bdd10k_cd"
    # gen_Img2Img
    "gen_Img2Img/bdd10k:gen_Img2Img/bdd10k_cd"
    # gen_IP2P
    "gen_IP2P/bdd10k:gen_IP2P/bdd10k_cd"
    # gen_LANIT
    "gen_LANIT/bdd10k:gen_LANIT/bdd10k_cd"
    # gen_Qwen_Image_Edit
    "gen_Qwen_Image_Edit/idd-aw:gen_Qwen_Image_Edit/idd-aw_cd"
    "gen_Qwen_Image_Edit/mapillaryvistas:gen_Qwen_Image_Edit/mapillaryvistas_cd"
    "gen_Qwen_Image_Edit/outside15k:gen_Qwen_Image_Edit/outside15k_cd"
    # gen_SUSTechGAN
    "gen_SUSTechGAN/bdd10k:gen_SUSTechGAN/bdd10k_cd"
    "gen_SUSTechGAN/idd-aw:gen_SUSTechGAN/idd-aw_cd"
    "gen_SUSTechGAN/mapillaryvistas:gen_SUSTechGAN/mapillaryvistas_cd"
    "gen_SUSTechGAN/outside15k:gen_SUSTechGAN/outside15k_cd"
    # gen_TSIT
    "gen_TSIT/bdd10k:gen_TSIT/bdd10k_cd"
    "gen_TSIT/idd-aw:gen_TSIT/idd-aw_cd"
    "gen_TSIT/mapillaryvistas:gen_TSIT/mapillaryvistas_cd"
    "gen_TSIT/outside15k:gen_TSIT/outside15k_cd"
    # gen_UniControl
    "gen_UniControl/bdd10k:gen_UniControl/bdd10k_cd"
)

for pair in "${RENAME_PAIRS[@]}"; do
    IFS=':' read -r old_path new_path <<< "$pair"
    full_old="${WEIGHTS_ROOT}/${old_path}"
    full_new="${WEIGHTS_ROOT}/${new_path}"
    
    if [ -d "$full_old" ]; then
        if [ -d "$full_new" ]; then
            echo "SKIP: Target already exists: $new_path"
        else
            echo "Renaming: $old_path -> $new_path"
            mv "$full_old" "$full_new"
        fi
    else
        echo "Not found: $old_path"
    fi
done

echo ""
echo "=== All operations complete ==="
echo ""

# Summary
echo "=== Summary ==="
echo "Corrupted iddaw_ad directories: ${#CORRUPTED_DIRS[@]} removed"
echo "Renamed directories: ${#RENAME_PAIRS[@]} processed"
