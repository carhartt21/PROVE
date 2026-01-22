# Study Coverage Analysis

**Last Updated:** 2026-01-22 (10:35)

## Summary

| Study | Path | Checkpoints | Strategies | Status |
|-------|------|-------------|------------|--------|
| **Stage 1** | `WEIGHTS/` | 306 | 27 | 🔄 MapillaryVistas retraining |
| **Stage 2** | `WEIGHTS_STAGE_2/` | 244 | 27 | ⏳ MapillaryVistas pending |
| **Ratio Ablation** | `WEIGHTS_RATIO_ABLATION/` | 171 | 6 | 🔶 Partial coverage |
| **Extended Training** | `WEIGHTS_EXTENDED/` | 19 | 5 | 🔶 Partial coverage |
| **Combinations** | `WEIGHTS_COMBINATIONS_chge7185/` | 109 | 27 | 🔶 IDD-AW/MV only |
| **Domain Adaptation** | Testing-only | Top 5 + baseline | ⏳ Not started |

---

## Stage 1: Clear Day Training

**Path:** `/scratch/aaa_exchange/AWARE/WEIGHTS/`
**Status:** 🔄 MapillaryVistas retraining in progress (BGR/RGB fix)

### Coverage Matrix

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k | Total |
|----------|:------:|:------:|:---------------:|:----------:|------:|
| baseline | ✅ 3/3 | ✅ 3/3 | 🔄 3/3 | ✅ 3/3 | 12 |
| gen_Attribute_Hallucination | ✅ 3/3 | ✅ 3/3 | 🔄 3/3 | ✅ 3/3 | 12 |
| gen_CNetSeg | ✅ 3/3 | ✅ 3/3 | 🔄 3/3 | ✅ 3/3 | 12 |
| gen_CUT | ✅ 3/3 | ✅ 3/3 | 🔄 3/3 | ✅ 3/3 | 12 |
| gen_IP2P | ✅ 3/3 | ✅ 3/3 | 🔄 3/3 | ✅ 3/3 | 12 |
| gen_Img2Img | ✅ 3/3 | ✅ 3/3 | 🔄 3/3 | ✅ 3/3 | 12 |
| gen_LANIT | ✅ 3/3 | ✅ 3/3 | 🔄 3/3 | ✅ 3/3 | 12 |
| gen_Qwen_Image_Edit | ✅ 3/3 | ✅ 3/3 | 🔄 3/3 | ✅ 3/3 | 12 |
| gen_SUSTechGAN | ✅ 3/3 | ✅ 3/3 | ❌ | ✅ 3/3 | 9 |
| gen_TSIT | ✅ 3/3 | ✅ 3/3 | ❌ | ✅ 3/3 | 9 |
| gen_UniControl | ✅ 3/3 | ✅ 3/3 | ❌ | ✅ 3/3 | 9 |
| gen_VisualCloze | ✅ 3/3 | ✅ 3/3 | ❌ | ✅ 3/3 | 9 |
| gen_Weather_Effect_Generator | ✅ 3/3 | ✅ 3/3 | ❌ | ✅ 3/3 | 9 |
| gen_albumentations_weather | ✅ 3/3 | ✅ 3/3 | 🔄 3/3 | ✅ 3/3 | 12 |
| gen_augmenters | ✅ 3/3 | ✅ 3/3 | 🔄 3/3 | ✅ 3/3 | 12 |
| gen_automold | ✅ 3/3 | ✅ 3/3 | 🔄 3/3 | ✅ 3/3 | 12 |
| gen_cycleGAN | ✅ 3/3 | ✅ 3/3 | 🔄 3/3 | ✅ 3/3 | 12 |
| gen_cyclediffusion | ✅ 3/3 | ✅ 3/3 | 🔄 3/3 | ✅ 3/3 | 12 |
| gen_flux_kontext | ✅ 3/3 | ✅ 3/3 | 🔄 3/3 | ✅ 3/3 | 12 |
| gen_stargan_v2 | ✅ 3/3 | ✅ 3/3 | 🔄 2/3 | ✅ 3/3 | 11 |
| gen_step1x_new | ✅ 3/3 | ✅ 3/3 | 🔄 3/3 | ✅ 3/3 | 12 |
| gen_step1x_v1p2 | ✅ 3/3 | ✅ 3/3 | 🔄 1/3 | ✅ 3/3 | 10 |
| photometric_distort | ✅ 3/3 | ✅ 3/3 | 🔄 3/3 | ✅ 3/3 | 12 |
| std_autoaugment | ✅ 3/3 | ✅ 3/3 | 🔄 3/3 | ✅ 3/3 | 12 |
| std_cutmix | ✅ 3/3 | ✅ 3/3 | 🔄 3/3 | ✅ 3/3 | 12 |
| std_mixup | ✅ 3/3 | ✅ 3/3 | 🔄 3/3 | ✅ 3/3 | 12 |
| std_randaugment | ✅ 3/3 | ✅ 3/3 | 🔄 3/3 | ✅ 3/3 | 12 |

**Legend:** 🔄 = Retraining in progress (BGR/RGB fix)

### Notes
- MapillaryVistas models being retrained due to BGR/RGB bug in `custom_transforms.py`
- Some strategies missing MapillaryVistas data (gen_SUSTechGAN, gen_TSIT, gen_UniControl, gen_VisualCloze, gen_Weather_Effect_Generator)

---

## Stage 2: All Domains Training

**Path:** `/scratch/aaa_exchange/AWARE/WEIGHTS_STAGE_2/`
**Status:** ⏳ MapillaryVistas pending (queued after Stage 1 retraining)

### Coverage Matrix

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k | Total |
|----------|:------:|:------:|:---------------:|:----------:|------:|
| baseline | ✅ 3/3 | ✅ 3/3 | ⏳ | ✅ 3/3 | 9 |
| gen_Attribute_Hallucination | ✅ 3/3 | ✅ 3/3 | ⏳ | ✅ 3/3 | 9 |
| gen_CNetSeg | ✅ 3/3 | ✅ 3/3 | ⏳ | ✅ 3/3 | 9 |
| gen_CUT | ✅ 3/3 | ✅ 3/3 | ⏳ | ✅ 3/3 | 9 |
| gen_IP2P | ✅ 3/3 | ✅ 3/3 | ⏳ | ✅ 3/3 | 9 |
| gen_Img2Img | ✅ 3/3 | ✅ 3/3 | ⏳ | ✅ 3/3 | 9 |
| gen_LANIT | ✅ 3/3 | ✅ 3/3 | ⏳ | ✅ 3/3 | 9 |
| gen_Qwen_Image_Edit | ✅ 3/3 | ✅ 3/3 | ⏳ | ✅ 3/3 | 9 |
| gen_SUSTechGAN | ✅ 3/3 | ✅ 3/3 | ⏳ | ✅ 3/3 | 9 |
| gen_TSIT | ✅ 3/3 | ✅ 3/3 | ⏳ | ✅ 3/3 | 9 |
| gen_UniControl | ✅ 3/3 | ✅ 3/3 | ⏳ | ✅ 3/3 | 9 |
| gen_VisualCloze | ✅ 3/3 | ✅ 3/3 | ⏳ | ✅ 3/3 | 9 |
| gen_Weather_Effect_Generator | ✅ 3/3 | ✅ 3/3 | ⏳ | ✅ 3/3 | 9 |
| gen_albumentations_weather | ✅ 3/3 | ✅ 3/3 | ⏳ | ✅ 3/3 | 9 |
| gen_augmenters | ✅ 3/3 | ✅ 3/3 | ⏳ | ✅ 3/3 | 9 |
| gen_automold | ✅ 3/3 | ✅ 3/3 | ⏳ | ✅ 3/3 | 9 |
| gen_cycleGAN | ✅ 3/3 | ✅ 3/3 | ⏳ | ✅ 3/3 | 9 |
| gen_cyclediffusion | ✅ 3/3 | ✅ 3/3 | ⏳ | ✅ 3/3 | 9 |
| gen_flux_kontext | ✅ 3/3 | ✅ 3/3 | ⏳ | ✅ 3/3 | 9 |
| gen_stargan_v2 | ✅ 3/3 | ✅ 3/3 | ⏳ | ✅ 3/3 | 9 |
| gen_step1x_new | ✅ 3/3 | ✅ 3/3 | ⏳ | ✅ 3/3 | 9 |
| gen_step1x_v1p2 | ✅ 3/3 | ✅ 3/3 | ⏳ | ✅ 3/3 | 9 |
| photometric_distort | ✅ 3/3 | ✅ 3/3 | ⏳ | ✅ 3/3 | 9 |
| std_autoaugment | ✅ 3/3 | ✅ 3/3 | ⏳ | ✅ 3/3 | 9 |
| std_cutmix | ✅ 3/3 | ✅ 3/3 | ⏳ | ✅ 3/3 | 9 |
| std_mixup | ✅ 3/3 | ✅ 3/3 | ⏳ | ✅ 3/3 | 9 |
| std_randaugment | ✅ 3/3 | ✅ 3/3 | ⏳ | ✅ 3/3 | 9 |

**Total:** 244 checkpoints (27 strategies × 3 datasets × 3 models)

### Notes
- MapillaryVistas training will start after Stage 1 retraining completes
- All non-MapillaryVistas configs are complete

---

## Ratio Ablation Study

**Path:** `/scratch/aaa_exchange/AWARE/WEIGHTS_RATIO_ABLATION/`
**Status:** 🔶 Partial coverage (selected strategies only)

### Study Design
Testing real/generated ratios: 0.00, 0.125, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875, 1.00

### Coverage Matrix

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k | Ratios |
|----------|:------:|:------:|:---------------:|:----------:|--------|
| gen_TSIT | 🔶 2/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | Multiple |
| gen_cycleGAN | ❌ | 🔶 2/3 | 🔶 2/3 | 🔶 2/3 | Multiple |
| gen_cyclediffusion | ❌ | 🔶 2/3 | ❌ | ❌ | Limited |
| gen_flux_kontext | ❌ | ❌ | 🔶 2/3 | ❌ | Limited |
| gen_stargan_v2 | ❌ | 🔶 2/3 | 🔶 1/3 | ❌ | Limited |
| gen_step1x_v1p2 | 🔶 2/3 | 🔶 2/3 | 🔶 2/3 | 🔶 2/3 | Multiple |

**Total:** 171 checkpoints across multiple ratios

### Notes
- Focus on top-performing generative strategies
- Ratio 0.50 excluded (standard training in WEIGHTS/)
- Used for finding optimal real/generated balance

---

## Extended Training Study

**Path:** `/scratch/aaa_exchange/AWARE/WEIGHTS_EXTENDED/`
**Status:** 🔶 Partial coverage (selected strategies/datasets)

### Study Design
Extended iterations: 40k, 60k, 80k, 100k, 120k, 140k, 160k, 320k

### Coverage Matrix

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k | Iterations |
|----------|:------:|:------:|:---------------:|:----------:|------------|
| gen_TSIT | 🔶 2/3 | 🔶 2/3 | ❌ | ❌ | Multiple |
| gen_UniControl | 🔶 2/3 | 🔶 2/3 | ❌ | ❌ | Multiple |
| gen_albumentations_weather | 🔶 2/3 | 🔶 2/3 | ❌ | ❌ | Multiple |
| gen_automold | 🔶 2/3 | 🔶 1/3 | ❌ | ❌ | Multiple |
| gen_cycleGAN | 🔶 2/3 | 🔶 2/3 | ❌ | ❌ | Multiple |

**Total:** 19 checkpoints (959 if counting all intermediate checkpoints)

### Notes
- Focus on BDD10k and IDD-AW datasets
- Models: pspnet_r50, segformer_mit-b5 primarily
- Intermediate checkpoints saved at each iteration milestone

---

## Combination Strategies Study

**Path:** `/scratch/aaa_exchange/AWARE/WEIGHTS_COMBINATIONS_chge7185/`
**Status:** 🔶 Complete for IDD-AW and MapillaryVistas, missing BDD10k/OUTSIDE15k

### Study Design
Combining generative + standard augmentation strategies

### Coverage Matrix (27 combinations)

| Strategy Combination | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k |
|---------------------|:------:|:------:|:---------------:|:----------:|
| gen_Attribute_Hallucination+photometric_distort | ❌ | 🔶 2/3 | 🔶 2/3 | ❌ |
| gen_flux_kontext+photometric_distort | ❌ | 🔶 2/3 | 🔶 2/3 | ❌ |
| gen_flux_kontext+std_autoaugment | ❌ | 🔶 2/3 | 🔶 2/3 | ❌ |
| gen_flux_kontext+std_cutmix | ❌ | 🔶 2/3 | 🔶 2/3 | ❌ |
| gen_flux_kontext+std_mixup | ❌ | 🔶 2/3 | 🔶 2/3 | ❌ |
| gen_flux_kontext+std_randaugment | ❌ | 🔶 2/3 | 🔶 2/3 | ❌ |
| gen_Qwen_Image_Edit+photometric_distort | ❌ | 🔶 2/3 | 🔶 2/3 | ❌ |
| gen_Qwen_Image_Edit+std_autoaugment | ❌ | 🔶 2/3 | 🔶 2/3 | ❌ |
| gen_Qwen_Image_Edit+std_cutmix | ❌ | 🔶 2/3 | 🔶 2/3 | ❌ |
| gen_Qwen_Image_Edit+std_mixup | ❌ | 🔶 2/3 | 🔶 2/3 | ❌ |
| gen_Qwen_Image_Edit+std_randaugment | ❌ | 🔶 2/3 | 🔶 2/3 | ❌ |
| gen_stargan_v2+photometric_distort | ❌ | 🔶 2/3 | 🔶 2/3 | ❌ |
| gen_step1x_new+photometric_distort | 🔶 1/3 | 🔶 2/3 | 🔶 2/3 | ❌ |
| gen_step1x_new+std_autoaugment | ❌ | 🔶 2/3 | 🔶 2/3 | ❌ |
| gen_step1x_new+std_cutmix | ❌ | 🔶 2/3 | 🔶 2/3 | ❌ |
| gen_step1x_new+std_mixup | ❌ | 🔶 2/3 | 🔶 2/3 | ❌ |
| gen_step1x_new+std_randaugment | ❌ | 🔶 2/3 | 🔶 2/3 | ❌ |
| std_autoaugment+photometric_distort | ❌ | 🔶 2/3 | 🔶 2/3 | ❌ |
| std_cutmix+photometric_distort | ❌ | 🔶 2/3 | 🔶 2/3 | ❌ |
| std_cutmix+std_autoaugment | ❌ | 🔶 2/3 | 🔶 2/3 | ❌ |
| std_mixup+photometric_distort | ❌ | 🔶 2/3 | 🔶 2/3 | ❌ |
| std_mixup+std_autoaugment | ❌ | 🔶 2/3 | 🔶 2/3 | ❌ |
| std_mixup+std_cutmix | ❌ | 🔶 2/3 | 🔶 2/3 | ❌ |
| std_randaugment+photometric_distort | ❌ | 🔶 2/3 | 🔶 2/3 | ❌ |
| std_randaugment+std_autoaugment | ❌ | 🔶 2/3 | 🔶 2/3 | ❌ |
| std_randaugment+std_cutmix | ❌ | 🔶 2/3 | 🔶 2/3 | ❌ |
| std_randaugment+std_mixup | ❌ | 🔶 2/3 | 🔶 2/3 | ❌ |

**Total:** 109 checkpoints

### Notes
- Conducted by chge7185
- Models: pspnet_r50, segformer_mit-b5 primarily
- Missing BDD10k and OUTSIDE15k coverage

---

## Domain Adaptation Ablation

**Path:** `/scratch/aaa_exchange/AWARE/WEIGHTS/domain_adaptation_ablation/`
**Type:** Testing-only study (uses existing checkpoints from `WEIGHTS/`)
**Status:** ⏳ Not started

### Study Design
Evaluate **cross-dataset domain generalization** using existing models:
- **Source Models:** Checkpoints from `WEIGHTS/` (trained on BDD10k, IDD-AW, MapillaryVistas)
- **Target Test Sets:** 
  - Cityscapes (clear_day condition)
  - ACDC (foggy, night, rainy, snowy)

### Research Questions
1. How well do models trained on one dataset generalize to other domains?
2. Does training on all weather conditions (Stage 2) improve adverse weather performance?
3. Which training datasets provide the best domain generalization?

### Planned Testing Matrix

| Source (Training) | Target (Testing) | Conditions |
|-------------------|------------------|------------|
| BDD10k models | Cityscapes + ACDC | 5 domains |
| IDD-AW models | Cityscapes + ACDC | 5 domains |
| MapillaryVistas models | Cityscapes + ACDC | 5 domains |

### Strategies to Test
- Top 5 generative strategies (gen_Qwen_Image_Edit, gen_Attribute_Hallucination, gen_cycleGAN, gen_flux_kontext, gen_step1x_new)
- Baseline models

### Notes
- **No training required** - uses existing checkpoints from Stage 1 (`WEIGHTS/`)
- Test results stored in `domain_adaptation_ablation/` subdirectories
- Script: `./scripts/submit_domain_adaptation_ablation.sh`
- Doc: [DOMAIN_ADAPTATION_ABLATION.md](DOMAIN_ADAPTATION_ABLATION.md)

---

## Legend

| Symbol | Meaning |
|--------|---------|
| ✅ 3/3 | All 3 models complete (deeplabv3plus_r50, pspnet_r50, segformer_mit-b5) |
| 🔶 2/3 | 2 of 3 models complete |
| 🔶 1/3 | 1 of 3 models complete |
| 🔄 | Retraining in progress |
| ⏳ | Pending/queued |
| ❌ | Not available |
