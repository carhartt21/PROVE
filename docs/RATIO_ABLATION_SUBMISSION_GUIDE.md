# Ratio Ablation Study Submission Guide

## Overview

The `batch_training_submission.py` script now supports ratio ablation studies through the `--stage ratio` option.

## Usage

### Basic Command

```bash
python scripts/batch_training_submission.py \
    --stage ratio \
    --ratios 0.0 0.25 0.5 \
    --strategies gen_stargan_v2 gen_cycleGAN \
    --datasets BDD10k IDD-AW \
    --models pspnet_r50 \
    --dry-run
```

### Options

| Option | Description | Example |
|--------|-------------|---------|
| `--stage ratio` | Enable ratio ablation mode | Required |
| `--ratios` | List of real/gen ratios to test | `0.0 0.12 0.25 0.5 0.75 1.0` |
| `--strategies` | Which generators to test | `gen_stargan_v2 gen_cycleGAN` |
| `--datasets` | Which datasets to use | `BDD10k IDD-AW` |
| `--models` | Which models to train | `pspnet_r50 deeplabv3plus_r50` |
| `--dry-run` | Preview without submitting | Always use first! |
| `--limit` | Max jobs to submit | `--limit 20` |

### Output Directory

Jobs are submitted to:
```
${AWARE_DATA_ROOT}/WEIGHTS_RATIO_ABLATION/{strategy}/{dataset}/{model}_ratio{XX}p{YY}/
```

Examples:
- `${AWARE_DATA_ROOT}/WEIGHTS_RATIO_ABLATION/gen_stargan_v2/bdd10k/pspnet_r50_ratio0p00/`
- `${AWARE_DATA_ROOT}/WEIGHTS_RATIO_ABLATION/gen_stargan_v2/bdd10k/pspnet_r50_ratio0p25/`
- `${AWARE_DATA_ROOT}/WEIGHTS_RATIO_ABLATION/gen_cycleGAN/iddaw/pspnet_r50_ratio0p50/`

**Note**: Uses `iddaw` (no hyphen) for IDD-AW dataset directory, consistent with existing ratio ablation convention.

## Example Workflows

### Full Ratio Sweep (All Standard Ratios)

```bash
# Dry run first
python scripts/batch_training_submission.py \
    --stage ratio \
    --ratios 0.00 0.12 0.25 0.38 0.50 0.62 0.75 0.88 \
    --strategies gen_stargan_v2 gen_cycleGAN gen_Attribute_Hallucination \
    --datasets BDD10k IDD-AW \
    --models pspnet_r50 \
    --dry-run

# If output looks correct, remove --dry-run and submit
```

**Job count**: 3 strategies × 2 datasets × 1 model × 8 ratios = **48 jobs**

### Control Test (Pure Synthetic vs Baseline)

```bash
# Test ratio=0.0 (100% generated) for key generators
python scripts/batch_training_submission.py \
    --stage ratio \
    --ratios 0.0 \
    --strategies gen_stargan_v2 gen_cycleGAN gen_Attribute_Hallucination \
    --datasets BDD10k IDD-AW \
    --models pspnet_r50 \
    --dry-run
```

**Job count**: 3 strategies × 2 datasets × 1 model × 1 ratio = **6 jobs**

### Fine-Grained Low Ratio Study

```bash
# Explore the 0.0-0.3 range where degradation is steep
python scripts/batch_training_submission.py \
    --stage ratio \
    --ratios 0.00 0.05 0.10 0.15 0.20 0.25 0.30 \
    --strategies gen_stargan_v2 \
    --datasets BDD10k \
    --models pspnet_r50 \
    --dry-run
```

**Job count**: 1 strategy × 1 dataset × 1 model × 7 ratios = **7 jobs**

### Multi-Model Comparison

```bash
# Test if ratio effects are model-dependent
python scripts/batch_training_submission.py \
    --stage ratio \
    --ratios 0.0 0.25 0.5 \
    --strategies gen_stargan_v2 \
    --datasets BDD10k \
    --models deeplabv3plus_r50 pspnet_r50 segformer_mit-b3 \
    --dry-run
```

**Job count**: 1 strategy × 1 dataset × 3 models × 3 ratios = **9 jobs**

## Pre-Flight Checks

The script automatically:
- ✅ Skips if checkpoint already exists (won't overwrite)
- ✅ Checks training lock (prevents duplicates)
- ✅ Verifies generated images exist for strategy+dataset

To bypass checks (use with caution):
```bash
--no-check-existing  # Ignore existing results
--no-check-locks     # Ignore training locks
```

## Key Differences: Stage 1 vs Stage 2 vs Ratio

| Feature | Stage 1 | Stage 2 | Ratio Ablation |
|---------|---------|---------|----------------|
| **Directory** | `WEIGHTS/` | `WEIGHTS_STAGE_2/` | `WEIGHTS_RATIO_ABLATION/` |
| **Domain Filter** | ✅ `--domain-filter clear_day` | ❌ No filter | ❌ No filter |
| **Ratios** | Fixed 0.5 | Fixed 0.5 | **User-specified (multiple)** |
| **Purpose** | Cross-domain robustness | Domain-inclusive performance | Ratio sensitivity analysis |

## Monitoring Jobs

```bash
# Check job status
bjobs -w | grep gen_stargan_v2

# View live training log
bpeek <job_id>

# Check completed training
ls -lh ${AWARE_DATA_ROOT}/WEIGHTS_RATIO_ABLATION/gen_stargan_v2/bdd10k/*/iter_10000.pth
```

## Analysis After Completion

Once jobs finish, analyze with:

```bash
# Update downstream results cache
python scripts/update_testing_tracker.py --stage ratio

# Generate ratio ablation analysis
python analysis_scripts/analyze_ratio_ablation.py \
    --strategy gen_stargan_v2 \
    --dataset BDD10k \
    --model pspnet_r50
```

## Current Status (2026-01-30)

### Jobs Already Running (Buggy ratio~0.125)

| Job ID | Strategy | Dataset | Model | Ratio | Status |
|--------|----------|---------|-------|-------|--------|
| 799816 | gen_stargan_v2 | BDD10k | pspnet_r50 | ~0.125 (buggy) | Running @ iter 2050 |
| 799817 | gen_stargan_v2 | IDD-AW | pspnet_r50 | ~0.125 (buggy) | Running @ iter 2050 |

**Note**: These were submitted before the batch composition bug fix and use incorrect ratio (1 real + 7 gen instead of 0 real + 8 gen). Keeping for comparison.

### Recommended Next Submission

Once bug fix is verified and user confirms:

```bash
# Submit true ratio=0.0 control tests (fixed composition)
python scripts/batch_training_submission.py \
    --stage ratio \
    --ratios 0.0 \
    --strategies gen_stargan_v2 gen_cycleGAN gen_Attribute_Hallucination \
    --datasets BDD10k IDD-AW \
    --models pspnet_r50 \
    --dry-run

# If approved, remove --dry-run and submit
```

This will create:
- `${AWARE_DATA_ROOT}/WEIGHTS_RATIO_ABLATION/gen_stargan_v2/bdd10k/pspnet_r50_ratio0p00/` ✅ New (fixed)
- `${AWARE_DATA_ROOT}/WEIGHTS_RATIO_ABLATION/gen_stargan_v2/iddaw/pspnet_r50_ratio0p00/` ✅ New (fixed)
- etc.

**Different from running jobs**: These use `WEIGHTS_RATIO_ABLATION/` path, won't conflict with current jobs in `WEIGHTS/` directory.

---

**Status**: ✅ Ready for submission  
**Awaiting**: User confirmation to proceed

