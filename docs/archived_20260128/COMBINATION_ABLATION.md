# Combination Strategy Ablation Study

This document describes the combination strategy ablation study, which analyzes the effects of combining different data augmentation strategies.

## Overview

The combination ablation study investigates whether combining multiple augmentation strategies provides synergistic benefits over using individual strategies alone.

## Combination Types

| Type | Description | Examples |
|------|-------------|----------|
| **Generative + Standard** | GAN/diffusion + traditional augmentation | gen_CUT+std_mixup, gen_cycleGAN+std_randaugment |
| **Standard + Standard** | Two traditional augmentation methods | std_randaugment+std_mixup, std_cutmix+std_autoaugment |
| **Baseline + Standard** | Baseline + single augmentation | baseline+std_cutmix |

## Data Location

Combination strategy results are stored separately from single strategies:

```
${AWARE_DATA_ROOT}/WEIGHTS_COMBINATIONS/
├── baseline+std_cutmix/
├── gen_CUT+std_mixup/
├── gen_CUT+std_randaugment/
├── gen_cycleGAN+std_mixup/
├── gen_cycleGAN+std_randaugment/
├── gen_StyleID+std_mixup/
├── gen_StyleID+std_randaugment/
├── std_cutmix+std_autoaugment/
├── std_mixup+std_autoaugment/
├── std_mixup+std_cutmix/
├── std_randaugment+std_autoaugment/
├── std_randaugment+std_cutmix/
└── std_randaugment+std_mixup/
```

## Analysis Script

### `analyze_combination_ablation.py`

**Usage:**
```bash
mamba run -n prove python analyze_combination_ablation.py
```

**Output Directory:** `result_figures/combination_ablation/`

### Generated Files

| File | Description |
|------|-------------|
| `combination_overview.png` | Performance overview by combination type |
| `combination_ablation_summary.csv` | Summary statistics |
| `combination_results.csv` | Raw results data |
| `combination_ablation_report.txt` | Detailed text report |
| `synergy_analysis.png` | Synergy effects vs best component |
| `component_interaction.png` | Component interaction heatmap |

## Key Findings

### Performance Ranking

| Rank | Combination | Type | mIoU | N_Results |
|------|------------|------|------|-----------|
| 1 | std_randaugment+std_mixup | Standard + Standard | 56.06 | 29 |
| 2 | std_cutmix+std_autoaugment | Standard + Standard | 55.82 | 30 |
| 3 | std_mixup+std_autoaugment | Standard + Standard | 55.67 | 30 |
| 4 | gen_CUT+std_mixup | Generative + Standard | 55.27 | 30 |
| 5 | std_randaugment+std_autoaugment | Standard + Standard | 54.97 | 30 |

### Observations

1. **Standard + Standard** combinations generally outperform **Generative + Standard**
2. Best combination (std_randaugment+std_mixup) beats the best single strategy
3. gen_StyleID combinations perform poorly (34-36 mIoU) - requires investigation
4. Mixing methods (cutmix, mixup) combine well with augmentation policies

## Synergy Analysis

The script analyzes whether combinations provide **synergistic** (better than best component), **neutral** (equal to best), or **negative** (worse than best) effects.

### Synergy Metrics

| Metric | Description |
|--------|-------------|
| `improvement_over_best` | mIoU(combination) - mIoU(best_component) |
| `improvement_over_avg` | mIoU(combination) - mean(mIoU(components)) |
| `synergy` | Positive/Neutral/Negative classification |

## How to Add New Combinations

### Using the Combination Training Script (Recommended)

For systematic combination ablation studies:

```bash
# List all combinations (top 3 gen × top 3 std × datasets × models)
./scripts/submit_combination_training.sh --list

# Preview bsub commands without submitting
./scripts/submit_combination_training.sh --dry-run

# Submit all combination training jobs
./scripts/submit_combination_training.sh

# Submit with limit
./scripts/submit_combination_training.sh --limit 10
```

**Default Configuration:**
- **Gen strategies:** gen_cyclediffusion, gen_TSIT, gen_cycleGAN
- **Std strategies:** std_randaugment, std_mixup, std_cutmix
- **Datasets:** MapillaryVistas, IDD-AW
- **Models:** SegFormer, PSPNet
- **Total:** 36 combinations (3 × 3 × 2 × 2)

### Manual Single Job Submission

1. **Train with combined strategy:**
   ```bash
   python unified_training.py --strategy gen_XXX --std-strategy std_YYY \
       --dataset acdc --model deeplabv3plus_r50
   ```

2. **Or use the generic template:**
   ```bash
   ./scripts/submit_training.sh --dataset BDD10k --model segformer_mit-b5 \
       --strategy gen_cycleGAN --std-strategy std_mixup
   ```

3. **Move to combinations directory (optional):**
   ```bash
   mv ${AWARE_DATA_ROOT}/WEIGHTS/gen_XXX+std_YYY ${AWARE_DATA_ROOT}/WEIGHTS_COMBINATIONS/
   ```

4. **Re-run analysis:**
   ```bash
   python analyze_combination_ablation.py
   ```

## Related Documentation

- [FAMILY_ANALYSIS.md](FAMILY_ANALYSIS.md) - Single strategy family analysis
- [UNIFIED_TRAINING.md](UNIFIED_TRAINING.md) - Training procedures
- [ABLATION_EVALUATION.md](ABLATION_EVALUATION.md) - Other ablation studies
