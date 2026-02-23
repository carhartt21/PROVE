# PROVE Repository Instructions

This document provides essential information for working with the PROVE repository, especially when using AI agents.

## Repository Overview

PROVE (PRobustness under adVErse weather) is a research project for training and evaluating semantic segmentation models under adverse weather conditions.

### Key Directories

```
${HOME}/repositories/PROVE/          # Repository root
├── scripts/                                # Job submission and utility scripts
├── docs/                                   # Documentation and tracker files
├── unified_training.py                     # Main training script
├── fine_grained_test.py                    # Main testing script
└── unified_training_config.py              # Configuration builder

${AWARE_DATA_ROOT}/WEIGHTS/        # Stage 1 models (clear_day training)
${AWARE_DATA_ROOT}/WEIGHTS_STAGE_2/# Stage 2 models (all_domains training)
```

### Training Stages

| Stage | Domain Filter | Directory | Description |
|-------|---------------|-----------|-------------|
| **Stage 1** | `clear_day` | `WEIGHTS/` | Train on clear weather only, test cross-domain |
| **Stage 2** | None (all) | `WEIGHTS_STAGE_2/` | Train on all weather conditions |

---

## Job Submission

### LSF Cluster Commands

```bash
# Check running jobs
bjobs -w
bjobs -u ${USER}

# Check job history
bhist -n 20

# Kill a job
bkill <job_id>
```

### Training Job Submission

**Using unified_training.py directly:**
```bash
# Stage 1 (clear_day)
python unified_training.py \
    --dataset BDD10k \
    --model deeplabv3plus_r50 \
    --strategy baseline \
    --domain-filter clear_day

# Stage 2 (all domains)
python unified_training.py \
    --dataset BDD10k \
    --model deeplabv3plus_r50 \
    --strategy baseline

# With generative augmentation
python unified_training.py \
    --dataset BDD10k \
    --model deeplabv3plus_r50 \
    --strategy gen_cycleGAN \
    --real-gen-ratio 0.5 \
    --domain-filter clear_day

# Add auxiliary loss (CE remains primary)
python unified_training.py \
    --dataset BDD10k \
    --model deeplabv3plus_r50 \
    --strategy baseline \
    --aux-loss focal
```

**Using submit_training.sh:**
```bash
./scripts/submit_training.sh \
    --dataset BDD10k \
    --model deeplabv3plus_r50 \
    --strategy baseline \
    --domain-filter clear_day

# Dry run to preview
./scripts/submit_training.sh \
    --dataset BDD10k \
    --model deeplabv3plus_r50 \
    --strategy gen_cycleGAN \
    --ratio 0.5 \
    --dry-run
```

**Batch submission with auxiliary loss:**
```bash
python scripts/batch_training_submission.py --stage 1 \
    --strategies baseline gen_cycleGAN \
    --ratios 0.5 \
    --aux-loss focal \
    -y
```

### Testing Job Submission

**Using fine_grained_test.py:**
```bash
python fine_grained_test.py \
    --config /path/to/training_config.py \
    --checkpoint /path/to/iter_80000.pth \
    --dataset BDD10k \
    --output-dir /path/to/test_results_detailed
```

**Using submit_testing.sh:**
```bash
./scripts/submit_testing.sh \
    --checkpoint /path/to/iter_80000.pth \
    --dataset BDD10k
```

**Auto-submit missing tests:**
```bash
python scripts/auto_submit_tests.py --dry-run    # Preview
python scripts/auto_submit_tests.py --limit 20   # Submit up to 20
```

---

## Available Models and Strategies

### Models
| Model | ID |
|-------|-----|
| DeepLabV3+ | `deeplabv3plus_r50` |
| PSPNet | `pspnet_r50` |
| SegFormer | `segformer_mit-b5` |

### Datasets
| Dataset | Expected Classes |
|---------|------------------|
| ACDC | 19 (Cityscapes) |
| BDD10k | 19 (Cityscapes) |
| IDD-AW | 19 (Cityscapes) |
| MapillaryVistas | 66 (native) |
| OUTSIDE15k | 24 (native) |

### Strategies
- **Baseline:** `baseline`
- **Standard Augmentation:** `std_minimal`, `std_autoaugment`, `std_cutmix`, `std_mixup`, `std_randaugment`, `std_photometric_distort`
- **Generative:** `gen_cycleGAN`, `gen_cyclediffusion`, `gen_CUT`, `gen_IP2P`, `gen_step1x_new`, `gen_flux_kontext`, etc.

---

## Tracking Progress

### Update Trackers

```bash
# Update Stage 1 training tracker
python scripts/update_training_tracker.py --stage 1

# Update Stage 2 training tracker
python scripts/update_training_tracker.py --stage 2

# Update testing tracker
python scripts/update_testing_tracker.py
```

### Key Tracker Files
- `docs/TRAINING_TRACKER_STAGE1.md` - Stage 1 training progress
- `docs/TRAINING_TRACKER_STAGE2.md` - Stage 2 training progress
- `docs/TESTING_TRACKER.md` - Testing coverage
- `TODO.md` - Task list and status summary

### Status Indicators
| Emoji | Status |
|-------|--------|
| ✅ | Complete |
| 🔶 | Partial (e.g., 2/3 models done) |
| 🔄 | Running |
| ⏳ | Pending |
| ❌ | Failed |
| ➖ | Skipped |

---

## Directory Structure

### Weights Directory Structure
```
WEIGHTS/                           # Stage 1 (clear_day)
├── baseline/
│   ├── bdd10k/
│   │   └── deeplabv3plus_r50/
│   │       ├── iter_80000.pth     # Trained model
│   │       ├── training_config.py # Config used for training
│   │       └── test_results_detailed/
│   │           └── 20260116_123456/
│   │               └── results.json
│   ├── idd-aw/
│   ├── mapillaryvistas/
│   └── outside15k/
├── gen_cycleGAN/
│   ├── bdd10k/
│   │   └── deeplabv3plus_r50_ratio0p50/
│   ...
```

### Test Results Format (results.json)
```json
{
    "overall": {
        "mIoU": 45.23,
        "aAcc": 92.1
    },
    "per_domain": {
        "clear_day": {"mIoU": 48.5, "aAcc": 93.2},
        "rainy": {"mIoU": 42.1, "aAcc": 90.8}
    },
    "per_class": {
        "road": {"IoU": 95.2},
        "sidewalk": {"IoU": 72.3}
    }
}
```

---

## Common Tasks

### Check Model num_classes
```python
import torch

ckpt = torch.load('iter_80000.pth', map_location='cpu')
state_dict = ckpt.get('state_dict', ckpt)

for key in state_dict:
    if 'conv_seg' in key and 'weight' in key:
        print(f"num_classes = {state_dict[key].shape[0]}")
        break
```

### Analyze Test Results
```bash
python test_result_analyzer.py --root ${AWARE_DATA_ROOT}/WEIGHTS --comprehensive
```

---

## Important Notes

### ⚠️ Native Classes vs Cityscapes Classes
- **Default behavior (since Jan 2026):** Native classes are used automatically
  - MapillaryVistas: 66 classes
  - OUTSIDE15k: 24 classes
  - ACDC, BDD10k, IDD-AW: 19 Cityscapes classes
- To force Cityscapes 19 classes, use `--no-native-classes`

**Example (forcing 19 classes, NOT recommended):**
```bash
python unified_training.py \
    --dataset MapillaryVistas \
    --model deeplabv3plus_r50 \
    --strategy baseline \
    --domain-filter clear_day \
    --no-native-classes  # NOT recommended!
```

### Directory Naming
- **No more `_cd` or `_ad` suffixes** in dataset directories
- Stage is determined by root directory (WEIGHTS vs WEIGHTS_STAGE_2)
- Keep hyphen in `idd-aw` (not `iddaw`)

---

## Updating TODO.md

**Always update TODO.md after completing tasks:**
1. Update status numbers
2. Move completed items to "Recently Completed"
3. Update "Last updated" timestamp

**Template for Recently Completed:**
```markdown
### Task Category (Date)
- ✅ **Task description**
  - Detail 1
  - Detail 2
```

---

## Git Workflow

```bash
# Check status
git status

# Stage and commit
git add <files>
git commit -m "type: Short description

Longer description if needed"

# Push
git push

# If rejected (remote has changes)
git stash
git pull --rebase
git stash pop
git push
```

**Commit Types:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `refactor:` - Code restructuring
- `chore:` - Maintenance tasks
