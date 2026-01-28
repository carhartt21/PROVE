# PROVE Evaluation Concept

## Executive Summary

PROVE (Pipeline for Recognition & Object Vision Evaluation) evaluates the effectiveness of various data augmentation strategies for improving semantic segmentation model performance, particularly in adverse weather conditions. This document provides a comprehensive overview of the evaluation framework, training configurations, and ablation studies.

## 1. Training Configurations

### 1.1 Training Variants

Each augmentation strategy is trained in **two variants** to enable comparison between full dataset training and clear weather only training:

| Variant | Model Suffix | Training Data | Purpose |
|---------|--------------|---------------|---------|
| **Full Dataset** | `{model}` | All available images (all weather/lighting conditions) | Baseline with diverse training data |
| **Clear Day Only** | `{model}_clear_day` | Only clear weather subset | Study benefit of weather diversity |

#### Why Two Variants?

Having both variants is scientifically valuable for:

1. **Isolating the Training Data Diversity Effect**
   - Both variants are tested on the **same complete test set** (including all weather conditions)
   - Difference in performance reveals how much training diversity helps
   - Clear day models serve as a "control group" with minimal weather diversity

2. **Understanding Domain Gap**
   | Model | On Clear Weather | On Adverse Weather | Domain Gap |
   |-------|------------------|-------------------|------------|
   | Clear day trained | Best case (matched) | Worst case (unseen) | LARGE |
   | Full dataset trained | Good | Better (seen similar) | SMALLER |

3. **Practical Decision Making**
   - Clear day data is often easier to collect (more available, fewer annotations needed)
   - If clear_day models perform well enough → simpler, cheaper training pipeline
   - If full models are significantly better → diverse training is worth the extra effort

4. **Baseline for Augmentation Comparison**
   - Helps interpret whether augmentation strategies provide missing diversity
   - If augmentation X helps clear_day models more → X provides weather diversity
   - If augmentation X helps both equally → X provides orthogonal benefits

**How it works:**
1. **Training**: The `--domain-filter clear_day` flag restricts training data to the clear_day subdirectory
2. **Storage**: Clear day models are stored with `_clear_day` suffix: `{model}_clear_day/`
3. **Testing**: Both variants are tested on the **same test set** (full test data including all weather conditions)

**Training Command Examples:**
```bash
# Full dataset training (all weather conditions)
python unified_training.py --dataset ACDC --model deeplabv3plus_r50 --strategy gen_cycleGAN

# Clear day only training
python unified_training.py --dataset ACDC --model deeplabv3plus_r50 --strategy gen_cycleGAN --domain-filter clear_day
```

**Directory Structure:**
```
WEIGHTS/{strategy}/{dataset}/
├── {model}/                    # Full dataset model
│   ├── iter_80000.pth          # Final checkpoint
│   ├── configs/                # Training config
│   ├── test_results/           # Standard test metrics
│   └── test_results_detailed/  # Fine-grained per-domain/per-class metrics
└── {model}_clear_day/          # Clear day only model  
    ├── iter_80000.pth
    ├── configs/
    ├── test_results/
    └── test_results_detailed/
```

### 1.2 Test Data Structure

Both model variants are tested on the **complete test set**, which includes all weather domains:

```
FINAL_SPLITS/{dataset}/test/
├── images/
│   ├── {dataset}/
│   │   ├── clear_day/          # Clear weather images
│   │   ├── foggy/              # Foggy images  
│   │   ├── night/              # Night images
│   │   ├── rainy/              # Rainy images
│   │   └── snowy/              # Snowy images
└── labels/
    └── {dataset}/
        ├── clear_day/
        ├── foggy/
        ├── night/
        ├── rainy/
        └── snowy/
```

**Key Point:** Testing uses ALL test data regardless of training variant. This allows direct comparison:
- Full model tested on: clear_day + foggy + night + rainy + snowy
- Clear_day model tested on: clear_day + foggy + night + rainy + snowy (same test set)

### 1.3 Model Matrix

For each strategy/dataset combination, we train **6 models** (3 architectures × 2 variants):

| Architecture | Full Dataset | Clear Day Only |
|--------------|--------------|----------------|
| DeepLabV3+ ResNet-50 | `deeplabv3plus_r50` | `deeplabv3plus_r50_clear_day` |
| PSPNet ResNet-50 | `pspnet_r50` | `pspnet_r50_clear_day` |
| SegFormer MiT-B5 | `segformer_mit-b5` | `segformer_mit-b5_clear_day` |

### 1.2 Source Datasets

| Dataset | Total Images | Clear Day Subset | Format | Notes |
|---------|-------------|------------------|--------|-------|
| ACDC | ~4,000 | Variable by domain | Cityscapes | Adverse weather focused |
| BDD10k | ~7,000 | ~2,000 | Cityscapes | Berkeley Driving Dataset |
| IDD-AW | ~1,800 | ~400 | Cityscapes | India Driving - Adverse Weather |
| MapillaryVistas | ~18,000 | N/A | Mapillary | Global street imagery |
| OUTSIDE15k | ~15,000 | ~8,000 | Cityscapes | Diverse outdoor scenes |

### 1.3 Model Architectures

| Model | Backbone | Parameters | Notes |
|-------|----------|------------|-------|
| `deeplabv3plus_r50` | ResNet-50 | ~26M | Strong encoder-decoder architecture |
| `pspnet_r50` | ResNet-50 | ~23M | Pyramid pooling module |
| `segformer_mit-b5` | MiT-B5 | ~84M | Transformer-based, best performance |

### 1.4 Augmentation Strategies

#### Base Strategies
| Strategy | Description |
|----------|-------------|
| `baseline` | No augmentation (control) |
| `std_photometric_distort` | Random brightness, contrast, saturation |

#### Standard Augmentations
| Strategy | Description |
|----------|-------------|
| `std_cutmix` | CutMix: cut and paste image regions |
| `std_mixup` | MixUp: blend two images |
| `std_autoaugment` | AutoAugment: learned augmentation policy |
| `std_randaugment` | RandAugment: random augmentation policy |

#### Generative Augmentations
| Strategy | Type | Description |
|----------|------|-------------|
| `gen_automold` | Weather simulation | Synthetic weather effects |
| `gen_cycleGAN` | GAN | Unpaired image translation |
| `gen_CUT` | GAN | Contrastive unpaired translation |
| `gen_TSIT` | GAN | Spatially-adaptive translation |
| `gen_LANIT` | GAN | Language-guided translation |
| `gen_stargan_v2` | GAN | Multi-domain translation |
| `gen_StyleID` | Diffusion | Style-guided editing |
| `gen_IP2P` | Diffusion | InstructPix2Pix |
| `gen_Img2Img` | Diffusion | Image-to-image diffusion |
| `gen_EDICT` | Diffusion | EDICT inversion editing |
| `gen_NST` | Neural Style | Neural style transfer |
| `gen_flux1_kontext` | Diffusion | Flux Kontext editing |
| `gen_SUSTechGAN` | GAN | Weather synthesis |
| `gen_UniControl` | Diffusion | Unified controllable generation |
| `gen_Weather_Effect_Generator` | Traditional | Programmatic weather effects |
| `gen_Attribute_Hallucination` | Diffusion | Attribute-based editing |
| `gen_Qwen_Image_Edit` | VLM | Vision-language model editing |
| `gen_augmenters` | Traditional | General augmentation library |
| `gen_albumentations_weather` | Traditional | Albumentations weather effects |
| `gen_VisualCloze` | Diffusion | Visual cloze editing |
| `gen_cyclediffusion` | Diffusion | Cycle-consistent diffusion |
| `gen_step1x_new` | Diffusion | Step1X editing model |

---

## 2. Evaluation Framework

### 2.1 Testing Philosophy

**Both training variants (Full Dataset and Clear Day Only) are tested on the same complete test set.**

This enables direct comparison of:
1. **Clear Day Performance**: How well each variant performs on clear weather
2. **Adverse Weather Performance**: How well each variant generalizes to adverse conditions
3. **Domain Gap**: Performance drop from clear to adverse conditions

### 2.2 Standard Testing

Tests each model on the complete test set and reports aggregate metrics:

**Per-Configuration Metrics:**
- **mIoU**: Mean Intersection over Union (primary metric)
- **mAcc**: Mean Accuracy
- **aAcc**: Pixel Accuracy
- **fwIoU**: Frequency-weighted IoU

**Command:**
```bash
./scripts/test_unified.sh single --dataset ACDC --model deeplabv3plus_r50 --strategy gen_cycleGAN
# Tests both full and _clear_day variants
```

**Output:**
```
WEIGHTS/{strategy}/{dataset}/{model}/test_results/metrics.json
WEIGHTS/{strategy}/{dataset}/{model}_clear_day/test_results/metrics.json
```

### 2.3 Fine-Grained Testing

Provides detailed breakdown by weather domain and semantic class:

**Per-Domain Metrics:** Performance on each weather condition
- Clear day, foggy, night, rainy, snowy

**Per-Class Metrics:** Performance on each semantic class
- Road, sidewalk, building, vegetation, vehicle, person, etc.

**Command:**
```bash
./scripts/test_unified.sh detailed --dataset ACDC --model deeplabv3plus_r50 --strategy gen_cycleGAN --mode per-domain
```

**Output:**
```
WEIGHTS/{strategy}/{dataset}/{model}/test_results_detailed/
├── metrics_summary.json        # Overall metrics
├── metrics_per_domain.json     # Per-weather metrics
├── metrics_per_class.json      # Per-class IoU
└── test_report.txt             # Human-readable summary
```

### 2.4 Comparison Analysis

With both variants tested on all domains, we can compute:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Clear Day Gap** | mIoU(full, clear) - mIoU(clear_day, clear) | Benefit of diverse training on clear weather |
| **Adverse Weather Gap** | mIoU(full, adverse) - mIoU(clear_day, adverse) | Benefit of diverse training on adverse weather |
| **Domain Drop (Full)** | mIoU(full, clear) - mIoU(full, adverse) | Robustness of full model |
| **Domain Drop (Clear Day)** | mIoU(clear_day, clear) - mIoU(clear_day, adverse) | Robustness of clear-only model |

**Expected Findings:**
- Full dataset models should have smaller domain drop
- Clear day models may perform equally or slightly better on clear weather only
- Adverse weather gap should be significantly larger for clear day models

### 2.5 Key Research Questions

1. **Training Data Diversity**: Does training on diverse weather conditions improve adverse weather performance?
   - Compare: Full dataset vs. Clear day only models on adverse weather test domains

2. **Augmentation Effectiveness**: Which augmentation strategies best improve adverse weather robustness?
   - Compare: All strategies vs. baseline, focusing on adverse weather performance

3. **Domain-Specific Performance**: How does each strategy perform on specific weather domains?
   - Compare: Per-domain metrics across strategies

---

## 3. Ablation Studies

### 3.1 Ratio Ablation Study

**Research Question:** What is the optimal ratio of real to generated images for training?

**Methodology:**
- Test ratios from 0.0 (100% generated) to 1.0 (100% real) in 0.125 increments
- Use top-5 performing generative strategies:
  1. `gen_LANIT` (55.71 mIoU)
  2. `gen_step1x_new` (55.70 mIoU)
  3. `gen_automold` (55.62 mIoU)
  4. `gen_TSIT` (55.61 mIoU)
  5. `gen_NST` (55.55 mIoU)

**Tested Configurations:**

| Ratio | Real Images | Generated Images | Notes |
|-------|-------------|------------------|-------|
| 1.0 | 100% | 0% | Pure real (baseline) |
| 0.875 | 87.5% | 12.5% | |
| 0.75 | 75% | 25% | |
| 0.625 | 62.5% | 37.5% | |
| 0.5 | 50% | 50% | Standard training ✓ |
| 0.375 | 37.5% | 62.5% | |
| 0.25 | 25% | 75% | |
| 0.125 | 12.5% | 87.5% | |
| 0.0 | 0% | 100% | Pure synthetic |

**Data Location:**
```
WEIGHTS_RATIO_ABLATION/{strategy}_{ratio}/{dataset}/{model}/
```

**Analysis Script:** `analysis_scripts/analyze_ratio_ablation.py`

---

### 3.2 Domain Adaptation Ablation Study

**Research Question:** How well do models trained on one dataset generalize to unseen datasets?

**Important:** This ablation uses **baseline** models only (not augmented strategies), comparing how source dataset and training data diversity affect cross-dataset generalization.

#### Why Compare Full vs. Clear Day?

This comparison is crucial for understanding domain adaptation:

1. **Training Distribution vs. Test Distribution**
   - Clear day model: Trained on narrow distribution (clear weather only)
   - Full model: Trained on wider distribution (all weather conditions)
   - Test on Cityscapes (clear) + ACDC (adverse): Both seen and unseen conditions

2. **Measuring Robustness**
   - Clear day model performance on adverse weather = pure generalization ability
   - Full model performance on adverse weather = generalization + similar training experience
   - Difference = benefit of diverse training for domain adaptation

3. **Cross-Dataset Validity**
   - Source datasets (BDD10k, IDD-AW, MapillaryVistas) have different characteristics
   - Comparing both variants across sources reveals which factors matter most

**Methodology:**
- Train baseline models on source datasets (BDD10k, IDD-AW, MapillaryVistas)
- Test on target domains:
  - **Cityscapes** (Test set - clear day baseline)
  - **ACDC** (Train+Val - adverse weather: foggy, night, rainy, snowy)

**Training Configurations:**

| Configuration | Training Data | Model Suffix | Purpose |
|--------------|---------------|--------------|---------|
| Full Dataset | All weather conditions | `{model}` | Maximum data diversity |
| Clear Day Only | Clear weather subset | `{model}_clear_day` | Isolate clear weather impact |

**Source Model Location:**
```
# Full dataset baseline models
WEIGHTS/baseline/{source_dataset}/{model}/iter_80000.pth

# Clear day baseline models  
WEIGHTS/baseline/{source_dataset}/{model}_clear_day/iter_80000.pth
```

**Output Directory:**
```
WEIGHTS/domain_adaptation_ablation/
├── bdd10k/
│   ├── deeplabv3plus_r50/
│   │   └── domain_adaptation_evaluation.json
│   ├── deeplabv3plus_r50_clear_day/
│   │   └── domain_adaptation_evaluation.json
│   ├── pspnet_r50/
│   ├── pspnet_r50_clear_day/
│   ├── segformer_mit-b5/
│   └── segformer_mit-b5_clear_day/
├── idd-aw/
│   └── ... (same structure)
└── mapillaryvistas/
    └── ... (same structure)
```

**Evaluation Domains:**

| Domain | Source | Images | Condition |
|--------|--------|--------|-----------|
| clear_day | Cityscapes | 1,525 | ☀️ Clear daytime |
| foggy | ACDC | 500 | 🌫️ Dense fog |
| night | ACDC | 506 | 🌙 Nighttime |
| rainy | ACDC | 500 | 🌧️ Rain |
| snowy | ACDC | 500 | ❄️ Snow |

**Key Comparisons:**
1. **Source Dataset**: Which training dataset generalizes best?
2. **Full vs. Clear Day Training**: Does diverse weather training help generalization?
3. **Domain Gap**: Performance degradation from clear to adverse conditions

**Job Submission:**
```bash
# Submit all domain adaptation jobs (18 total)
bash scripts/submit_domain_adaptation_ablation.sh --all

# Submit only full dataset models (9 jobs)
bash scripts/submit_domain_adaptation_ablation.sh --all-full

# Submit only clear_day models (9 jobs)  
bash scripts/submit_domain_adaptation_ablation.sh --all-clear-day
```

**Analysis Script:** `analysis_scripts/analyze_domain_adaptation_ablation.py`

---

### 3.3 Combination Ablation Study

**Research Question:** Do combined augmentation strategies provide synergistic benefits?

**Combination Types:**

| Type | Description | Example |
|------|-------------|---------|
| Generative + Standard | GAN/diffusion + traditional | `gen_CUT+std_mixup` |
| Standard + Standard | Two traditional methods | `std_randaugment+std_mixup` |
| Baseline + Standard | Baseline + single augmentation | `baseline+std_cutmix` |

**Tested Combinations:**
- `gen_CUT+std_mixup`
- `gen_CUT+std_randaugment`
- `gen_cycleGAN+std_mixup`
- `gen_cycleGAN+std_randaugment`
- `gen_StyleID+std_mixup`
- `gen_StyleID+std_randaugment`
- `std_cutmix+std_autoaugment`
- `std_mixup+std_autoaugment`
- `std_mixup+std_cutmix`
- `std_randaugment+std_autoaugment`
- `std_randaugment+std_cutmix`
- `std_randaugment+std_mixup`

**Data Location:**
```
WEIGHTS_COMBINATIONS/{strategy1}+{strategy2}/{dataset}/{model}/
```

**Synergy Analysis:**
- **Positive synergy**: Combination > best single component
- **Neutral**: Combination ≈ best single component
- **Negative synergy**: Combination < best single component

**Analysis Script:** `analysis_scripts/analyze_combination_ablation.py`

---

### 3.4 Extended Training Ablation Study

**Research Question:** Does training beyond 80k iterations improve performance?

**Methodology:**
- Extend training from 80k to 160k iterations
- Test on select strategies that showed promise

**Tested Configurations:**
- `gen_flux1_kontext` on ACDC
- `gen_NST` on ACDC
- `gen_SUSTechGAN` on ACDC
- `std_cutmix+std_autoaugment` on ACDC
- gen_Attribute_Hallucination  
- gen_Img2Img     
- gen_SUSTechGAN  
- std_cutmix+std_autoaugment  
- std_randaugment+std_mixup
- gen_automold                 
- gen_LANIT       
- gen_TSIT        
- std_mixup+std_autoaugment
- gen_CUT                      
- gen_step1x_new  
- gen_UniControl  
- std_randaugment

**Data Location:**
```
WEIGHTS_EXTENDED/{strategy}/{dataset}/{model}/iter_160000.pth
```

**Analysis Script:** `analysis_scripts/analyze_extended_training.py`

---

## 4. Results Structure

### 4.1 Directory Overview

```
/scratch/aaa_exchange/AWARE/
├── FINAL_SPLITS/                    # Dataset splits
│   ├── {dataset}/
│   │   ├── train/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   └── test/
│   │       ├── images/
│   │       └── labels/
│   └── gen_{strategy}/              # Generated images
│       └── {dataset}/
│           └── train/
│               ├── images/
│               └── labels/
│
├── WEIGHTS/                          # Main training results
│   └── {strategy}/
│       └── {dataset}/
│           ├── {model}/              # Full dataset model
│           │   ├── iter_80000.pth
│           │   ├── configs/
│           │   ├── test_results/
│           │   └── test_results_detailed/
│           └── {model}_clear_day/    # Clear day model
│               └── ...
│
├── WEIGHTS_RATIO_ABLATION/           # Ratio ablation results
│   └── {strategy}_{ratio}/
│       └── ...
│
├── WEIGHTS_COMBINATIONS/             # Combination results
│   └── {strategy1}+{strategy2}/
│       └── ...
│
└── WEIGHTS_EXTENDED/                 # Extended training results
    └── {strategy}/
        └── ...
```

### 4.2 Metrics Files

**Standard Test Results (`test_results/metrics.json`):**
```json
{
  "mIoU": 55.23,
  "mAcc": 67.45,
  "aAcc": 92.18,
  "fwIoU": 87.34
}
```

**Detailed Test Results (`test_results_detailed/`):**
```json
// metrics_per_domain.json
{
  "clear_day": {"mIoU": 62.4, "mAcc": 75.2, ...},
  "foggy": {"mIoU": 48.3, "mAcc": 58.1, ...},
  "night": {"mIoU": 41.2, "mAcc": 52.8, ...},
  ...
}

// metrics_per_class.json
{
  "road": {"IoU": 95.2, "Acc": 97.8},
  "sidewalk": {"IoU": 78.3, "Acc": 85.4},
  "building": {"IoU": 88.7, "Acc": 93.2},
  ...
}
```

---

## 5. Analysis Workflow

### 5.1 Strategy Performance Analysis

```bash
# Generate strategy leaderboard
python analysis_scripts/generate_strategy_leaderboard.py

# Analyze strategy families (GAN, Diffusion, Traditional)
python analysis_scripts/analyze_strategy_families.py

# Visualize domain gap between clear and adverse conditions
python analysis_scripts/analyze_domain_gap_corrected.py
```

### 5.2 Ablation Study Analysis

```bash
# Ratio ablation analysis
python analysis_scripts/analyze_ratio_ablation.py
python analysis_scripts/visualize_ratio_ablation.py

# Domain adaptation analysis
python analysis_scripts/analyze_domain_adaptation_ablation.py

# Combination strategy analysis
python analysis_scripts/analyze_combination_ablation.py

# Extended training analysis
python analysis_scripts/analyze_extended_training.py
python analysis_scripts/visualize_extended_training.py
```

### 5.3 Report Generation

```bash
# Generate comprehensive analysis report
python analysis_scripts/generate_analysis_report.py
```

---

## 6. Key Insights

### 6.1 Training Variant Comparison

| Comparison | Expected Finding |
|------------|------------------|
| Full vs. Clear Day on clear test | Similar or Full slightly better |
| Full vs. Clear Day on adverse test | Full significantly better |
| Domain gap (Clear → Adverse) | Smaller for Full dataset training |

### 6.2 Strategy Family Comparison

| Family | Characteristics | Best For |
|--------|-----------------|----------|
| GAN-based | Realistic transformations, paired/unpaired | Weather simulation |
| Diffusion-based | High quality, controllable | Style transfer, editing |
| Traditional | Fast, deterministic | Quick augmentation |
| Standard | Simple, well-understood | Baseline improvement |

### 6.3 Ablation Study Insights

| Study | Key Finding |
|-------|-------------|
| Ratio Ablation | Optimal ratio varies by strategy (typically 0.25-0.75) |
| Domain Adaptation | Weather-diverse training improves generalization |
| Combination | Standard+Standard often outperforms Gen+Standard |
| Extended Training | Diminishing returns beyond 80k for most strategies |

---

## 7. Related Documentation

- [UNIFIED_TRAINING.md](UNIFIED_TRAINING.md) - Training system details
- [UNIFIED_TESTING.md](UNIFIED_TESTING.md) - Testing procedures
- [RATIO_ABLATION.md](RATIO_ABLATION.md) - Ratio ablation details
- [DOMAIN_ADAPTATION_ABLATION.md](DOMAIN_ADAPTATION_ABLATION.md) - Domain adaptation details
- [COMBINATION_ABLATION.md](COMBINATION_ABLATION.md) - Combination strategy details
- [EXTENDED_TRAINING.md](EXTENDED_TRAINING.md) - Extended training details
- [FAMILY_ANALYSIS.md](FAMILY_ANALYSIS.md) - Strategy family analysis
