# PROVE Unified Testing Documentation

## Overview

The PROVE testing pipeline uses `fine_grained_test.py` for per-domain/per-class evaluation and `batch_test_submission.py` / `auto_submit_tests.py` for batch job submission to the LSF cluster.

> **Note:** The legacy `test_unified.sh` shell script has been superseded by the Python-based testing tools described below.

## Components

| Component | Purpose |
|-----------|---------|
| `fine_grained_test.py` | Core testing script: per-domain, per-class metrics with optimized inference |
| `scripts/batch_test_submission.py` | **Preferred** batch test job submission (pre-flight checks, duplicate detection) |
| `scripts/auto_submit_tests.py` | Auto-submit missing tests for completed training runs |

## Quick Start

```bash
# Test a single model
python fine_grained_test.py --config /path/training_config.py \
    --checkpoint /path/iter_80000.pth \
    --dataset BDD10k --output-dir /path/test_results_detailed

# Auto-submit all missing tests (Stage 1)
python scripts/auto_submit_tests.py --stage 1 --dry-run

# Batch test submission
python scripts/batch_test_submission.py --stage cityscapes-gen --dry-run
python scripts/batch_test_submission.py --stage cityscapes-gen -y
```

## fine_grained_test.py

The core testing script performs evaluation with per-domain and per-class breakdown.

### Usage

```bash
python fine_grained_test.py \
    --config ${AWARE_DATA_ROOT}/WEIGHTS/baseline/bdd10k/pspnet_r50/training_config.py \
    --checkpoint ${AWARE_DATA_ROOT}/WEIGHTS/baseline/bdd10k/pspnet_r50/iter_80000.pth \
    --dataset BDD10k \
    --output-dir ${AWARE_DATA_ROOT}/WEIGHTS/baseline/bdd10k/pspnet_r50/test_results_detailed
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--config <path>` | Path to training config file | Required |
| `--checkpoint <path>` | Path to checkpoint file (.pth) | Required |
| `--dataset <name>` | Dataset name (BDD10k, ACDC, IDD-AW, MapillaryVistas, OUTSIDE15k) | Required |
| `--output-dir <path>` | Output directory for results | Required |
| `--test-type <type>` | Test type (e.g., `acdc` for cross-domain) | Dataset default |

### Output Format

Results are saved as `results.json`:

```json
{
    "overall": {"mIoU": 45.23, "aAcc": 92.1},
    "per_domain": {"clear_day": {"mIoU": 48.5}, "rainy": {"mIoU": 42.1}},
    "per_class": {"road": {"IoU": 95.2}, "sidewalk": {"IoU": 72.3}}
}
```

## Batch Test Submission

### scripts/batch_test_submission.py (Preferred)

```bash
# Cityscapes-gen stage
python scripts/batch_test_submission.py --stage cityscapes-gen --dry-run
python scripts/batch_test_submission.py --stage cityscapes-gen -y

# With specific test type (e.g., ACDC cross-domain)
python scripts/batch_test_submission.py --stage cityscapes-gen --test-type acdc --dry-run
```

### scripts/auto_submit_tests.py

Automatically finds completed training runs without test results and submits test jobs:

```bash
# Stage 1 tests
python scripts/auto_submit_tests.py --stage 1 --dry-run
python scripts/auto_submit_tests.py --stage 1 --limit 20

# Stage 2 tests
python scripts/auto_submit_tests.py --stage 2 --dry-run

# Cityscapes tests
python scripts/auto_submit_tests.py --stage cityscapes --dry-run
```

## Metrics

### Segmentation Metrics

| Metric | Description |
|--------|-------------|
| `mIoU` | Mean Intersection over Union — primary metric |
| `aAcc` | Average Accuracy — overall pixel accuracy |
| `mAcc` | Mean Accuracy — average per-class accuracy |
| `fwIoU` | Frequency Weighted IoU — IoU weighted by class frequency |

### Available Domains by Dataset

| Dataset | Domains |
|---------|---------|
| ACDC | clear_day, cloudy, dawn_dusk, foggy, night, rainy, snowy |
| BDD10k | daytime, night, dawn_dusk |
| IDD-AW | clear, rainy, foggy, hazy, low_light |
| MapillaryVistas | all (no domain split) |
| OUTSIDE15k | all (no domain split) |

## Update Trackers After Testing

```bash
python scripts/update_testing_tracker.py --stage all
```

## See Also

- [UNIFIED_TRAINING.md](UNIFIED_TRAINING.md) - Training documentation
- [EVALUATION_STAGE_STATUS.md](EVALUATION_STAGE_STATUS.md) - Overall progress tracking
- [LABEL_HANDLING.md](LABEL_HANDLING.md) - Dataset label formats
