# Domain Adaptation Ablation Study: Cross-Dataset Generalization to ACDC

## Overview

This experiment evaluates the **cross-dataset domain adaptation** capability of models trained on traffic-focused datasets (BDD10k, IDD-AW, MapillaryVistas) when tested on the ACDC adverse weather benchmark.

### Motivation

The ACDC dataset contains reference images (`_ref_` suffix) with **mismatched labels** - labels from corresponding adverse weather images are incorrectly applied to clear-day reference images showing different content (different vehicles, pedestrians, camera angles). This makes ACDC unreliable for training/validation.

**Solution**: Use ACDC (train+test, excluding `_ref_` images) purely as a **domain adaptation evaluation target** for models trained on other traffic datasets.

## Experimental Setup

### Training Datasets (Source Domains)

Models trained on these datasets will be evaluated:

| Dataset | Train Images | Test Images | Label Format | Notes |
|---------|-------------|-------------|--------------|-------|
| BDD10k | ~7,000 | ~1,000 | TrainID (0-18+255) | Berkeley Driving Dataset (segmentation subset) |
| IDD-AW | ~1,800 | ~450 | TrainID (0-19+255) | India Driving Dataset - Adverse Weather |
| MapillaryVistas | ~18,000 | ~2,000 | RGB Color→TrainID | Global street-level imagery, diverse conditions |

### Evaluation Dataset (Target Domain)

**ACDC (Adverse Conditions Dataset with Correspondences)**

The complete ACDC dataset (train + test) is used for evaluation, excluding reference images:

| Domain | Train (non-ref) | Test (non-ref) | **Total** |
|--------|-----------------|----------------|-----------|
| foggy | 214 | 144 | **358** |
| rainy | 241 | 160 | **401** |
| snowy | 284 | 190 | **474** |
| night | 350 | 145 | **495** |
| cloudy | 246 | 110 | **356** |
| **Total** | **1,335** | **749** | **2,084** |

**Excluded from evaluation**:
- `clear_day` (0 non-ref images - ALL are problematic reference images)
- `dawn_dusk` (0 non-ref images - ALL are problematic reference images)
- `_ref_` suffix images in other domains (label mismatch issue)

### Models

All three segmentation architectures:
- **DeepLabV3+ ResNet-50** (deeplabv3plus_r50)
- **PSPNet ResNet-50** (pspnet_r50)
- **SegFormer MiT-B5** (segformer_mit-b5)

### Training Configurations

Using existing **baseline** checkpoints from 80k iterations:

```
/scratch/aaa_exchange/AWARE/WEIGHTS/baseline/{dataset}/{model}/iter_80000.pth
```

## Label Unification

### Critical Handling

All datasets use Cityscapes 19-class format, but with different internal representations:

1. **BDD10k, IDD-AW**: Already in trainID format (0-18), no transformation needed
2. **MapillaryVistas**: RGB color-encoded → needs `MapillarytoCityscapes` mapping
3. **ACDC**: Cityscapes labelID format (0-33) → needs `CityscapesID_to_TrainID` mapping

The evaluation pipeline must:
1. Transform ACDC labels from Cityscapes labelID → trainID
2. Filter out `_ref_` images from evaluation
3. Report per-domain (weather condition) metrics

## Research Questions

### Primary Questions

1. **Q1: Cross-Dataset Generalization**
   - How well do models trained on BDD10k/IDD-AW/MapillaryVistas generalize to ACDC adverse weather?
   - Which source dataset produces the best domain adaptation?

2. **Q2: Weather-Specific Performance**
   - Are certain weather conditions (fog, rain, snow, night) easier to adapt to?
   - Does the source dataset's weather diversity affect target performance?

3. **Q3: Architecture Sensitivity**
   - Do transformer-based models (SegFormer) generalize better than CNN-based (DeepLabV3+, PSPNet)?
   - Is there an architecture-dataset interaction effect?

### Secondary Questions

4. **Q4: Geographic Domain Gap**
   - MapillaryVistas (global) vs BDD10k (US) vs IDD-AW (India) → ACDC (Europe)
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

2. **IDD-AW** may perform surprisingly well on specific conditions due to:
   - Native adverse weather training data
   - Despite smaller size, focused on weather variation

3. **Night domain** likely hardest for all source datasets:
   - Limited nighttime samples in training data
   - Significant appearance shift

## Metrics

- **mIoU** (mean Intersection over Union) - primary metric
- **Per-class IoU** - to identify which semantic classes suffer most in adaptation
- **Per-domain mIoU** - breakdown by weather condition

## Implementation

### File Filtering

```python
def filter_acdc_files(file_list):
    """Exclude reference images from ACDC evaluation"""
    return [f for f in file_list if '_ref' not in f.name]

def filter_acdc_domains(domain_list):
    """Exclude domains with no valid images"""
    return [d for d in domain_list if d not in ['clear_day', 'dawn_dusk']]
```

### Label Transformation for ACDC

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

## Job Submission Matrix

| Source Dataset | Model | Checkpoint | Target Domain |
|---------------|-------|------------|---------------|
| BDD10k | deeplabv3plus_r50 | iter_80000.pth | ACDC (5 domains) |
| BDD10k | pspnet_r50 | iter_80000.pth | ACDC (5 domains) |
| BDD10k | segformer_mit-b5 | iter_80000.pth | ACDC (5 domains) |
| IDD-AW | deeplabv3plus_r50 | iter_80000.pth | ACDC (5 domains) |
| IDD-AW | pspnet_r50 | iter_80000.pth | ACDC (5 domains) |
| IDD-AW | segformer_mit-b5 | iter_80000.pth | ACDC (5 domains) |
| MapillaryVistas | deeplabv3plus_r50 | iter_80000.pth | ACDC (5 domains) |
| MapillaryVistas | pspnet_r50 | iter_80000.pth | ACDC (5 domains) |
| MapillaryVistas | segformer_mit-b5 | iter_80000.pth | ACDC (5 domains) |

**Total: 9 evaluation jobs** (each evaluates all 5 ACDC domains)

## Result Storage

Results will be saved to:
```
/scratch/aaa_exchange/AWARE/WEIGHTS/domain_adaptation_ablation/
└── {source_dataset}/
    └── {model}/
        ├── acdc_evaluation.json
        └── per_domain_metrics.csv
```

## Script Usage

```bash
# Submit all domain adaptation evaluation jobs
./scripts/submit_domain_adaptation_ablation.sh --all

# Submit single job
./scripts/submit_domain_adaptation_ablation.sh \
    --source-dataset BDD10k \
    --model deeplabv3plus_r50

# Dry run
./scripts/submit_domain_adaptation_ablation.sh --all --dry-run
```

## Analysis

After jobs complete, analyze with:
```bash
python analyze_domain_adaptation_ablation.py
```

This will generate:
- Cross-dataset performance heatmap
- Per-domain breakdown bar charts
- Architecture comparison plots
- Statistical significance tests
