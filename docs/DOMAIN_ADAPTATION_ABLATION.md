# Domain Adaptation Ablation Study: Cross-Dataset Domain Generalization

## Overview

This experiment evaluates the **cross-dataset domain adaptation** capability of models trained on traffic-focused datasets (BDD10k, IDD-AW, MapillaryVistas) when tested on:
- **Cityscapes** (representing clear_day condition)
- **ACDC** (representing adverse weather conditions: foggy, night, rainy, snowy)

### Motivation

Understanding how models trained on one domain generalize to other weather/lighting conditions is critical for robust autonomous driving systems. This ablation study measures:
1. How well models maintain performance on clear conditions (Cityscapes)
2. How much performance degrades in adverse conditions (ACDC domains)
3. Which training datasets provide the best domain generalization

## Experimental Setup

### Training Datasets (Source Domains)

Models trained on these datasets will be evaluated:

| Dataset | Train Images | Test Images | Label Format | Notes |
|---------|-------------|-------------|--------------|-------|
| BDD10k | ~7,000 | ~1,000 | TrainID (0-18+255) | Berkeley Driving Dataset (segmentation subset) |
| IDD-AW | ~1,800 | ~450 | TrainID (0-19+255) | India Driving Dataset - Adverse Weather |
| MapillaryVistas | ~18,000 | ~2,000 | RGB Color→TrainID | Global street-level imagery, diverse conditions |

### Evaluation Datasets (Target Domains)

#### Clear Day Condition: Cityscapes

**Cityscapes** represents the clear_day baseline condition for comparison:

| City | Test Images | Structure |
|------|-------------|-----------|
| berlin | 544 | `test/images/Cityscapes/berlin/*_leftImg8bit.png` |
| bielefeld | 181 | `test/images/Cityscapes/bielefeld/*_leftImg8bit.png` |
| bonn | 46 | `test/images/Cityscapes/bonn/*_leftImg8bit.png` |
| leverkusen | 58 | `test/images/Cityscapes/leverkusen/*_leftImg8bit.png` |
| mainz | 298 | `test/images/Cityscapes/mainz/*_leftImg8bit.png` |
| munich | 398 | `test/images/Cityscapes/munich/*_leftImg8bit.png` |
| **Total** | **1,525** | |

Labels: `*_gtFine_labelIds.png` (Cityscapes labelID format, converted to trainID)

#### Adverse Conditions: ACDC

**ACDC (Adverse Conditions Dataset with Correspondences)** provides adverse weather domains:

| Domain | Test Images | Structure |
|--------|-------------|-----------|
| foggy | 500 | `test/images/ACDC/foggy/*_rgb_anon.png` |
| night | 506 | `test/images/ACDC/night/*_rgb_anon.png` |
| rainy | 500 | `test/images/ACDC/rainy/*_rgb_anon.png` |
| snowy | 500 | `test/images/ACDC/snowy/*_rgb_anon.png` |
| **Total** | **2,006** | |

Labels: `*_gt_labelIds.png` (Cityscapes labelID format, converted to trainID)

### Combined Domain Summary

| Domain | Source | Images | Condition |
|--------|--------|--------|-----------|
| clear_day | Cityscapes | 1,525 | ☀️ Sunny/overcast daytime |
| foggy | ACDC | 500 | 🌫️ Dense fog |
| night | ACDC | 506 | 🌙 Nighttime |
| rainy | ACDC | 500 | 🌧️ Rain |
| snowy | ACDC | 500 | ❄️ Snow |
| **Total** | | **3,531** | |

### Models

- **PSPNet ResNet-50** (pspnet_r50)
- **SegFormer MiT-B5** (segformer_mit-b5)

### Training Configurations

Two training configurations are compared:

1. **Full Dataset Models** - Trained on all weather conditions from each source dataset
2. **Clear Day Baseline Models** - Trained only on clear_day subset of each source dataset

**Checkpoint Locations:**

```
# Full dataset models (trained on all weather conditions)
# Location: WEIGHTS_STAGE_2
/scratch/aaa_exchange/AWARE/WEIGHTS_STAGE_2/baseline/{dataset}/{model}/iter_80000.pth

# Clear day baseline models (trained on clear_day only)
# Location: WEIGHTS (original)
/scratch/aaa_exchange/AWARE/WEIGHTS/baseline/{dataset}/{model}/iter_80000.pth
```

> **Note:** The script automatically selects the correct weights directory based on the variant:
> - Full dataset (no `_clear_day` suffix) → uses `WEIGHTS_STAGE_2/`
> - Clear day only (`_clear_day` suffix) → uses `WEIGHTS/`

### Research Comparison

This setup enables comparing:
- **Full vs. Clear Day Training**: Does training on all weather conditions help adverse weather performance?
- **Domain Gap**: How much does each model degrade from clear_day to adverse conditions?

## Label Unification

### Critical Handling

Both Cityscapes and ACDC use Cityscapes labelID format (0-33), which must be converted to trainID (0-18) for evaluation:

```python
# Cityscapes labelID → trainID mapping (from label_unification.py)
CITYSCAPES_ID_TO_TRAINID = {
    7: 0,   # road
    8: 1,   # sidewalk
    11: 2,  # building
    12: 3,  # wall
    13: 4,  # fence
    17: 5,  # pole
    19: 6,  # traffic light
    20: 7,  # traffic sign
    21: 8,  # vegetation
    22: 9,  # terrain
    23: 10, # sky
    24: 11, # person
    25: 12, # rider
    26: 13, # car
    27: 14, # truck
    28: 15, # bus
    31: 16, # train
    32: 17, # motorcycle
    33: 18, # bicycle
}
# All other IDs → 255 (ignore)
```

### File Naming Conventions

| Dataset | Image Pattern | Label Pattern |
|---------|---------------|---------------|
| Cityscapes | `{city}_{seq}_{frame}_leftImg8bit.png` | `{city}_{seq}_{frame}_gtFine_labelIds.png` |
| ACDC | `{seq}_frame_{id}_rgb_anon.png` | `{seq}_frame_{id}_gt_labelIds.png` |

## Research Questions

### Primary Questions

1. **Q1: Cross-Dataset Generalization**
   - How well do models trained on BDD10k/IDD-AW/MapillaryVistas generalize to Cityscapes and ACDC?
   - Which source dataset produces the best domain adaptation across all conditions?

2. **Q2: Weather-Specific Performance**
   - How does performance vary across clear_day, foggy, night, rainy, snowy conditions?
   - Is the domain gap from clear_day (Cityscapes) to adverse (ACDC) larger than between source and clear_day?

3. **Q3: Architecture Sensitivity**
   - Do transformer-based models (SegFormer) generalize better than CNN-based (PSPNet)?
   - Is there an architecture-dataset interaction effect?

### Secondary Questions

4. **Q4: Geographic Domain Gap**
   - MapillaryVistas (global) vs BDD10k (US) vs IDD-AW (India) → Cityscapes/ACDC (Europe)
   - Does geographic diversity in training help cross-dataset transfer?

5. **Q5: Dataset Scale Effect**
   - MapillaryVistas (~18k) vs BDD10k (~7k) vs IDD-AW (~1.8k)
   - Does larger training set size correlate with better adaptation?

## Expected Outcomes

### Hypothesis

1. **MapillaryVistas** should provide best cross-dataset generalization due to:
   - Largest training set
   - Most diverse weather conditions and geographic regions
   - Already includes some adverse weather samples

2. **IDD-AW** may perform surprisingly well on adverse conditions due to:
   - Native adverse weather training data
   - Despite smaller size, focused on weather variation

3. **Night domain** likely hardest for all source datasets:
   - Limited nighttime samples in training data
   - Significant appearance shift

4. **Clear_day (Cityscapes)** should show best performance:
   - Most similar to source training distributions
   - Well-lit, structured European urban scenes

## Metrics

- **mIoU** (mean Intersection over Union) - primary metric
- **Per-class IoU** - to identify which semantic classes suffer most in adaptation
- **Per-domain mIoU** - breakdown by weather condition (clear_day, foggy, night, rainy, snowy)

## Implementation

### Data Structure

```
/scratch/aaa_exchange/AWARE/FINAL_SPLITS/test/
├── images/
│   ├── Cityscapes/          # clear_day condition
│   │   ├── berlin/*.png
│   │   ├── bielefeld/*.png
│   │   ├── bonn/*.png
│   │   ├── leverkusen/*.png
│   │   ├── mainz/*.png
│   │   └── munich/*.png
│   └── ACDC/                # adverse conditions
│       ├── foggy/*.png      # flat structure: *_rgb_anon.png
│       ├── night/*.png
│       ├── rainy/*.png
│       └── snowy/*.png
└── labels/
    ├── Cityscapes/          # *_gtFine_labelIds.png
    │   └── {same city structure}
    └── ACDC/                # *_gt_labelIds.png
        └── {same domain structure}
```

### Label Transformation

```python
def transform_cityscapes_label(label: np.ndarray) -> np.ndarray:
    """Transform label from Cityscapes labelID to trainID format."""
    if label.ndim == 3:
        label = label[:, :, 0]
    return CITYSCAPES_LUT[label]  # Lookup table for fast conversion
```

## Job Submission Matrix

### Full Dataset Models (6 jobs)

| Source Dataset | Model | Checkpoint | Target Domains |
|---------------|-------|------------|----------------|
| BDD10k | pspnet_r50 | iter_80000.pth | clear_day + 4 ACDC domains |
| BDD10k | segformer_mit-b5 | iter_80000.pth | clear_day + 4 ACDC domains |
| IDD-AW | pspnet_r50 | iter_80000.pth | clear_day + 4 ACDC domains |
| IDD-AW | segformer_mit-b5 | iter_80000.pth | clear_day + 4 ACDC domains |
| MapillaryVistas | pspnet_r50 | iter_80000.pth | clear_day + 4 ACDC domains |
| MapillaryVistas | segformer_mit-b5 | iter_80000.pth | clear_day + 4 ACDC domains |

### Clear Day Baseline Models (6 jobs)

| Source Dataset | Model | Checkpoint | Target Domains |
|---------------|-------|------------|----------------|
| BDD10k | pspnet_r50_clear_day | iter_80000.pth | clear_day + 4 ACDC domains |
| BDD10k | segformer_mit-b5_clear_day | iter_80000.pth | clear_day + 4 ACDC domains |
| IDD-AW | pspnet_r50_clear_day | iter_80000.pth | clear_day + 4 ACDC domains |
| IDD-AW | segformer_mit-b5_clear_day | iter_80000.pth | clear_day + 4 ACDC domains |
| MapillaryVistas | pspnet_r50_clear_day | iter_80000.pth | clear_day + 4 ACDC domains |
| MapillaryVistas | segformer_mit-b5_clear_day | iter_80000.pth | clear_day + 4 ACDC domains |

**Total: 12 evaluation jobs** (each evaluates 5 domains: clear_day, foggy, night, rainy, snowy)

## Result Storage

Results will be saved to:
```
/scratch/aaa_exchange/AWARE/WEIGHTS/domain_adaptation_ablation/
└── {source_dataset}/
    ├── {model}/                        # Full dataset models
    │   └── domain_adaptation_evaluation.json
    └── {model}_clear_day/              # Clear day baseline models
        └── domain_adaptation_evaluation.json
```

## Script Usage

```bash
# List all available checkpoints and their status
./scripts/submit_domain_adaptation_ablation.sh --list

# Submit ALL jobs (baseline + top 15 strategies)
./scripts/submit_domain_adaptation_ablation.sh --all

# Submit only baseline models
./scripts/submit_domain_adaptation_ablation.sh --all-full        # Full dataset models
./scripts/submit_domain_adaptation_ablation.sh --all-clear-day   # Clear_day only models

# Submit only augmentation strategies (no baseline)
./scripts/submit_domain_adaptation_ablation.sh --all-strategies

# Submit single strategy
./scripts/submit_domain_adaptation_ablation.sh --strategy gen_cyclediffusion

# Submit single baseline job (full dataset)
./scripts/submit_domain_adaptation_ablation.sh \
    --source-dataset BDD10k \
    --model pspnet_r50

# Submit single baseline job (clear_day)
./scripts/submit_domain_adaptation_ablation.sh \
    --source-dataset BDD10k \
    --model pspnet_r50 \
    --variant _clear_day

# Dry run - show commands without executing
./scripts/submit_domain_adaptation_ablation.sh --all --dry-run

# Skip configurations that already have results
./scripts/submit_domain_adaptation_ablation.sh --all --skip-existing
```

### Baseline Checkpoint Availability

| Dataset | Model | Full (WEIGHTS_STAGE_2) | Clear_day (WEIGHTS) |
|---------|-------|:----------------------:|:-------------------:|
| BDD10k | pspnet_r50 | ✅ | ✅ |
| BDD10k | segformer_mit-b5 | ❌ | ✅ |
| IDD-AW | pspnet_r50 | ✅ | ❌ |
| IDD-AW | segformer_mit-b5 | ✅ | ❌ |
| MapillaryVistas | pspnet_r50 | ✅ | ✅ |
| MapillaryVistas | segformer_mit-b5 | ❌ | ✅ |

### Top 15 Augmentation Strategies

Models trained with these augmentation strategies will also be evaluated:

| Strategy | BDD10k | IDD-AW | MapillaryVistas |
|----------|:------:|:------:|:---------------:|
| gen_cyclediffusion | ✅ | ✅ | ❌ |
| gen_flux_kontext | ✅ | ✅ | ✅ |
| gen_step1x_new | ✅ | partial | ✅ |
| gen_step1x_v1p2 | ✅ | ✅ | ✅ |
| gen_stargan_v2 | ✅ | ✅ | ✅ |
| gen_cycleGAN | ✅ | ✅ | ✅ |
| gen_automold | ✅ | ✅ | ✅ |
| gen_albumentations_weather | ✅ | ✅ | ✅ |
| gen_TSIT | ✅ | ✅ | ❌ |
| gen_UniControl | ❌ | ✅ | ✅ |
| std_randaugment | ✅ | partial | ✅ |
| std_autoaugment | ✅ | ❌ | ✅ |
| std_cutmix | ✅ | ❌ | ✅ |
| std_mixup | ✅ | ❌ | ✅ |
| photometric_distort | ✅ | ✅ | ✅ |

> **Note:** Run `./scripts/submit_domain_adaptation_ablation.sh --list` to see the current status of all checkpoints.

## Analysis

After jobs complete, analyze with:
```bash
python analysis_scripts/analyze_domain_adaptation_ablation.py
```

This will generate:
- Cross-dataset performance heatmap (source vs domain)
- Per-domain breakdown bar charts
- Architecture comparison plots
- clear_day vs adverse conditions gap analysis
- Statistical significance tests

### Expected Output

```
Domain Adaptation Results - Full Dataset Models:
                         clear_day  foggy   night   rainy   snowy   Average
Source/Model
BDD10k/dlv3+              XX.X%    XX.X%   XX.X%   XX.X%   XX.X%    XX.X%
BDD10k/pspnet             XX.X%    XX.X%   XX.X%   XX.X%   XX.X%    XX.X%
BDD10k/segformer          XX.X%    XX.X%   XX.X%   XX.X%   XX.X%    XX.X%
IDD-AW/dlv3+              XX.X%    XX.X%   XX.X%   XX.X%   XX.X%    XX.X%
...

Domain Adaptation Results - Clear Day Baseline Models:
                              clear_day  foggy   night   rainy   snowy   Average
Source/Model
BDD10k/dlv3+_clear_day         XX.X%    XX.X%   XX.X%   XX.X%   XX.X%    XX.X%
BDD10k/pspnet_clear_day        XX.X%    XX.X%   XX.X%   XX.X%   XX.X%    XX.X%
BDD10k/segformer_clear_day     XX.X%    XX.X%   XX.X%   XX.X%   XX.X%    XX.X%
...

Full vs Clear Day Comparison (Δ mIoU):
Source/Model                   clear_day   foggy   night   rainy   snowy
BDD10k/dlv3+                    +X.X%     +X.X%   +X.X%   +X.X%   +X.X%
...
```
