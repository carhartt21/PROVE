#!/bin/bash
# Safe WEIGHTS Directory Renames
# Only performs renames where target directory does NOT exist
# Created: 2026-01-15

set -e

WEIGHTS_ROOT="/scratch/aaa_exchange/AWARE/WEIGHTS"

echo "=========================================="
echo "WEIGHTS Directory Safe Rename Script"
echo "=========================================="
echo "This script only renames directories where the target does NOT exist."
echo "Directories requiring merges will be skipped."
echo ""

# Counter
renamed=0
skipped=0
errors=0

rename_if_safe() {
    local src="$1"
    local dst="$2"
    
    if [ ! -d "$WEIGHTS_ROOT/$src" ]; then
        echo "  SKIP: Source does not exist: $src"
        skipped=$((skipped + 1))
        return
    fi
    
    if [ -d "$WEIGHTS_ROOT/$dst" ]; then
        echo "  SKIP (merge needed): $src → $dst (target exists)"
        skipped=$((skipped + 1))
        return
    fi
    
    echo "  RENAME: $src → $dst"
    mv "$WEIGHTS_ROOT/$src" "$WEIGHTS_ROOT/$dst"
    renamed=$((renamed + 1))
}

echo ""
echo "=== Processing baseline/ ==="
rename_if_safe "baseline/bdd10k" "baseline/bdd10k_ad"
rename_if_safe "baseline/idd-aw" "baseline/idd-aw_ad"
rename_if_safe "baseline/iddaw_ad" "baseline/idd-aw_ad"
rename_if_safe "baseline/iddaw_cd" "baseline/idd-aw_cd"
rename_if_safe "baseline/mapillaryvistas" "baseline/mapillaryvistas_ad"
rename_if_safe "baseline/outside15k" "baseline/outside15k_ad"

echo ""
echo "=== Processing gen_albumentations_weather/ ==="
rename_if_safe "gen_albumentations_weather/bdd10k" "gen_albumentations_weather/bdd10k_ad"
rename_if_safe "gen_albumentations_weather/idd-aw" "gen_albumentations_weather/idd-aw_ad"
rename_if_safe "gen_albumentations_weather/mapillaryvistas" "gen_albumentations_weather/mapillaryvistas_ad"
rename_if_safe "gen_albumentations_weather/outside15k" "gen_albumentations_weather/outside15k_ad"

echo ""
echo "=== Processing gen_Attribute_Hallucination/ ==="
rename_if_safe "gen_Attribute_Hallucination/bdd10k" "gen_Attribute_Hallucination/bdd10k_ad"
rename_if_safe "gen_Attribute_Hallucination/iddaw_ad" "gen_Attribute_Hallucination/idd-aw_ad"
rename_if_safe "gen_Attribute_Hallucination/iddaw_cd" "gen_Attribute_Hallucination/idd-aw_cd"

echo ""
echo "=== Processing gen_augmenters/ ==="
rename_if_safe "gen_augmenters/bdd10k" "gen_augmenters/bdd10k_ad"
rename_if_safe "gen_augmenters/idd-aw" "gen_augmenters/idd-aw_ad"
rename_if_safe "gen_augmenters/mapillaryvistas" "gen_augmenters/mapillaryvistas_ad"
rename_if_safe "gen_augmenters/outside15k" "gen_augmenters/outside15k_ad"

echo ""
echo "=== Processing gen_automold/ ==="
rename_if_safe "gen_automold/bdd10k" "gen_automold/bdd10k_ad"
rename_if_safe "gen_automold/idd-aw" "gen_automold/idd-aw_ad"
rename_if_safe "gen_automold/mapillaryvistas" "gen_automold/mapillaryvistas_ad"
rename_if_safe "gen_automold/outside15k" "gen_automold/outside15k_ad"

echo ""
echo "=== Processing gen_CNetSeg/ ==="
rename_if_safe "gen_CNetSeg/bdd10k" "gen_CNetSeg/bdd10k_ad"
rename_if_safe "gen_CNetSeg/idd-aw" "gen_CNetSeg/idd-aw_ad"
rename_if_safe "gen_CNetSeg/iddaw_ad" "gen_CNetSeg/idd-aw_ad"
rename_if_safe "gen_CNetSeg/iddaw_cd" "gen_CNetSeg/idd-aw_cd"
rename_if_safe "gen_CNetSeg/mapillaryvistas" "gen_CNetSeg/mapillaryvistas_ad"
rename_if_safe "gen_CNetSeg/outside15k" "gen_CNetSeg/outside15k_ad"

echo ""
echo "=== Processing gen_CUT/ ==="
rename_if_safe "gen_CUT/bdd10k" "gen_CUT/bdd10k_ad"
rename_if_safe "gen_CUT/iddaw_ad" "gen_CUT/idd-aw_ad"
rename_if_safe "gen_CUT/iddaw_cd" "gen_CUT/idd-aw_cd"

echo ""
echo "=== Processing gen_cyclediffusion/ ==="
rename_if_safe "gen_cyclediffusion/iddaw_cd" "gen_cyclediffusion/idd-aw_cd"

echo ""
echo "=== Processing gen_cycleGAN/ ==="
rename_if_safe "gen_cycleGAN/bdd10k" "gen_cycleGAN/bdd10k_ad"
rename_if_safe "gen_cycleGAN/idd-aw" "gen_cycleGAN/idd-aw_ad"
rename_if_safe "gen_cycleGAN/iddaw_cd" "gen_cycleGAN/idd-aw_cd"
rename_if_safe "gen_cycleGAN/mapillaryvistas" "gen_cycleGAN/mapillaryvistas_ad"
rename_if_safe "gen_cycleGAN/outside15k" "gen_cycleGAN/outside15k_ad"

echo ""
echo "=== Processing gen_EDICT/ ==="
rename_if_safe "gen_EDICT/bdd10k" "gen_EDICT/bdd10k_ad"

echo ""
echo "=== Processing gen_flux_kontext/ ==="
rename_if_safe "gen_flux_kontext/iddaw_ad" "gen_flux_kontext/idd-aw_ad"
rename_if_safe "gen_flux_kontext/iddaw_cd" "gen_flux_kontext/idd-aw_cd"

echo ""
echo "=== Processing gen_Img2Img/ ==="
rename_if_safe "gen_Img2Img/bdd10k" "gen_Img2Img/bdd10k_ad"

echo ""
echo "=== Processing gen_IP2P/ ==="
rename_if_safe "gen_IP2P/bdd10k" "gen_IP2P/bdd10k_ad"
rename_if_safe "gen_IP2P/iddaw_ad" "gen_IP2P/idd-aw_ad"
rename_if_safe "gen_IP2P/iddaw_cd" "gen_IP2P/idd-aw_cd"

echo ""
echo "=== Processing gen_LANIT/ ==="
rename_if_safe "gen_LANIT/bdd10k" "gen_LANIT/bdd10k_ad"

echo ""
echo "=== Processing gen_Qwen_Image_Edit/ ==="
rename_if_safe "gen_Qwen_Image_Edit/idd-aw" "gen_Qwen_Image_Edit/idd-aw_ad"
rename_if_safe "gen_Qwen_Image_Edit/iddaw_ad" "gen_Qwen_Image_Edit/idd-aw_ad"
rename_if_safe "gen_Qwen_Image_Edit/mapillaryvistas" "gen_Qwen_Image_Edit/mapillaryvistas_ad"
rename_if_safe "gen_Qwen_Image_Edit/outside15k" "gen_Qwen_Image_Edit/outside15k_ad"

echo ""
echo "=== Processing gen_stargan_v2/ ==="
rename_if_safe "gen_stargan_v2/iddaw_cd" "gen_stargan_v2/idd-aw_cd"

echo ""
echo "=== Processing gen_step1x_v1p2/ ==="
rename_if_safe "gen_step1x_v1p2/iddaw_cd" "gen_step1x_v1p2/idd-aw_cd"

echo ""
echo "=== Processing gen_SUSTechGAN/ ==="
rename_if_safe "gen_SUSTechGAN/bdd10k" "gen_SUSTechGAN/bdd10k_ad"
rename_if_safe "gen_SUSTechGAN/idd-aw" "gen_SUSTechGAN/idd-aw_ad"
rename_if_safe "gen_SUSTechGAN/mapillaryvistas" "gen_SUSTechGAN/mapillaryvistas_ad"
rename_if_safe "gen_SUSTechGAN/outside15k" "gen_SUSTechGAN/outside15k_ad"

echo ""
echo "=== Processing gen_TSIT/ ==="
rename_if_safe "gen_TSIT/bdd10k" "gen_TSIT/bdd10k_ad"
rename_if_safe "gen_TSIT/idd-aw" "gen_TSIT/idd-aw_ad"
rename_if_safe "gen_TSIT/iddaw_ad" "gen_TSIT/idd-aw_ad"
rename_if_safe "gen_TSIT/mapillaryvistas" "gen_TSIT/mapillaryvistas_ad"
rename_if_safe "gen_TSIT/outside15k" "gen_TSIT/outside15k_ad"

echo ""
echo "=== Processing gen_UniControl/ ==="
rename_if_safe "gen_UniControl/bdd10k" "gen_UniControl/bdd10k_ad"
rename_if_safe "gen_UniControl/iddaw_ad" "gen_UniControl/idd-aw_ad"

echo ""
echo "=== Processing gen_VisualCloze/ ==="
rename_if_safe "gen_VisualCloze/iddaw_ad" "gen_VisualCloze/idd-aw_ad"

echo ""
echo "=== Processing photometric_distort/ ==="
rename_if_safe "photometric_distort/iddaw_cd" "photometric_distort/idd-aw_cd"

echo ""
echo "=== Processing std_minimal/ ==="
rename_if_safe "std_minimal/iddaw_ad" "std_minimal/idd-aw_ad"

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "  Renamed: $renamed"
echo "  Skipped: $skipped"
echo "  Errors:  $errors"
echo ""
echo "Directories skipped due to merge conflicts need manual handling."
echo "See: docs/WEIGHTS_CONSOLIDATION_PLAN.md for merge instructions."
