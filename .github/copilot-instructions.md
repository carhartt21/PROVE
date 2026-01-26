# PROVE Repository Instructions

## Architecture Overview

PROVE evaluates semantic segmentation models under adverse weather conditions using MMSegmentation. The pipeline has **two training stages**:

| Stage | Domain Filter | Weights Dir | Purpose |
|-------|--------------|-------------|---------|
| **Stage 1** | `clear_day` | `WEIGHTS/` | Train clear-only, test cross-domain robustness |
| **Stage 2** | None (all) | `WEIGHTS_STAGE_2/` | Train all conditions, evaluate domain-inclusive |

**External Data Paths:**
```
/scratch/aaa_exchange/AWARE/WEIGHTS/           # Stage 1 weights
/scratch/aaa_exchange/AWARE/WEIGHTS_STAGE_2/   # Stage 2 weights
/scratch/aaa_exchange/AWARE/WEIGHTS_RATIO_ABLATION/  # Ratio ablation study
/scratch/aaa_exchange/AWARE/WEIGHTS_EXTENDED/        # Extended training study
```

**Core Data Flow:**
```
unified_training.py → unified_training_config.py → MMSegmentation training
                           ↓
                    custom_transforms.py (label mapping: RGB→class ID)
                           ↓
                    fine_grained_test.py → results.json (per-domain/per-class metrics)
```

## Critical Conventions

### ⚠️ Label Handling (Common Bug Source)
- **mmcv.imfrombytes() returns BGR**, not RGB - always account for channel order
- MapillaryVistas/OUTSIDE15k use RGB-encoded labels requiring `MapillaryRGBToClassId` transform
- BDD10k/ACDC/IDD-AW use Cityscapes train IDs (single-channel, 19 classes)
- **Always use `--use-native-classes`** for MapillaryVistas (66 classes) and OUTSIDE15k (24 classes)

### ⚠️ Stage 1 vs Stage 2 Training (Critical Difference)
| Aspect | Stage 1 | Stage 2 |
|--------|---------|---------|
| **Domain Filter** | `--domain-filter clear_day` | NO domain filter |
| **Output Dir** | `WEIGHTS/` | `WEIGHTS_STAGE_2/` |
| **Purpose** | Cross-domain robustness | Domain-inclusive training |
| **Auto Scripts** | `auto_submit_tests.py` | `auto_submit_tests_stage2.py` |

**Common mistake:** Forgetting `--domain-filter clear_day` for Stage 1 trains on ALL domains (wrong!).

### Directory Naming
- Lowercase dataset dirs: `bdd10k`, `idd-aw`, `mapillaryvistas`, `outside15k`
- Keep hyphen in `idd-aw` (not `iddaw`)
- No `_cd`/`_ad` suffixes - stage determined by root directory

### Weights Structure
```
WEIGHTS/{strategy}/{dataset}/{model}[_ratio0p50]/
├── iter_80000.pth              # Final checkpoint (80k iterations)
├── training_config.py          # Config used for training
└── test_results_detailed/{timestamp}/results.json
```

## Essential Commands

### LSF Cluster (HPC Job Management)
```bash
bjobs -w                          # List all jobs with full names
bjobs -u mima2416 | grep RUN      # Running jobs only
bjobs -u mima2416 | wc -l         # Count total jobs
bkill <job_id>                    # Kill specific job
bkill 0                           # Kill ALL your jobs
bhist -n 20                       # Recent job history
bpeek <job_id>                    # View running job output
```

### Training
```bash
# Stage 1 (REQUIRES --domain-filter clear_day)
python unified_training.py --dataset BDD10k --model deeplabv3plus_r50 \
    --strategy baseline --domain-filter clear_day

# Stage 2 (NO domain filter - trains on all conditions)
python unified_training.py --dataset BDD10k --model deeplabv3plus_r50 \
    --strategy gen_cycleGAN --real-gen-ratio 0.5

# MapillaryVistas REQUIRES --use-native-classes (66 classes, not 19)
python unified_training.py --dataset MapillaryVistas --model deeplabv3plus_r50 \
    --strategy baseline --use-native-classes --domain-filter clear_day

# Custom batch size and learning rate (for ablation studies)
python unified_training.py --dataset BDD10k --model deeplabv3plus_r50 \
    --strategy baseline --batch-size 4 --lr 0.02 --warmup-iters 500

# Submit as LSF job instead of running locally
python unified_training.py --dataset BDD10k --model deeplabv3plus_r50 \
    --strategy baseline --domain-filter clear_day --submit-job
```

### Testing
```bash
python fine_grained_test.py --config /path/config.py --checkpoint /path/iter_80000.pth \
    --dataset BDD10k --output-dir /path/test_results_detailed

# Auto-submit missing tests (always dry-run first!)
python scripts/auto_submit_tests.py --dry-run        # Stage 1
python scripts/auto_submit_tests_stage2.py --dry-run # Stage 2
python scripts/auto_submit_tests_stage2.py --limit 20
```

### Update Trackers (after job completion)
```bash
python scripts/update_training_tracker.py --stage 1
python scripts/update_training_tracker.py --stage 2
python scripts/update_testing_tracker.py --stage 1
python scripts/update_testing_tracker.py --stage 2
```

## Key Files

| File | Purpose |
|------|---------|
| `unified_training.py` | Main training entry point, handles job submission |
| `fine_grained_test.py` | Per-domain/per-class evaluation with optimized inference |
| `unified_training_config.py` | Generates MMSeg configs from CLI args |
| `custom_transforms.py` | Label transforms including `MapillaryRGBToClassId` |
| `TODO.md` | **Check first** - current status, active jobs, known issues |
| `docs/EVALUATION_STAGE_STATUS.md` | Training/testing progress overview |

## Models, Datasets & Strategies

**Models:** `deeplabv3plus_r50`, `pspnet_r50`, `segformer_mit-b5`

**Datasets:**
| Dataset | Classes | Label Format |
|---------|---------|--------------|
| ACDC, BDD10k, IDD-AW | 19 | Cityscapes train IDs |
| MapillaryVistas | 66 | RGB-encoded |
| OUTSIDE15k | 24 | RGB-encoded |

**Strategies:** `baseline`, `std_autoaugment`, `std_cutmix`, `std_mixup`, `std_randaugment`, `photometric_distort`, `gen_cycleGAN`, `gen_IP2P`, `gen_flux_kontext`, `gen_step1x_new`, etc.

## Ablation Studies

| Study | Directory | Purpose |
|-------|-----------|---------|
| Ratio Ablation | `WEIGHTS_RATIO_ABLATION/` | Real/gen ratios: 0.00, 0.12, 0.25, 0.38, 0.50, 0.62, 0.75, 0.88 |
| Extended Training | `WEIGHTS_EXTENDED/` | Iterations: 40k→160k (20k steps) + 320k |
| Batch Size Ablation | `WEIGHTS_BATCH_SIZE_ABLATION/` | Batch sizes: 2, 4, 8, 16 with LR scaling |

Analysis scripts: `analysis_scripts/analyze_ratio_ablation.py`, `analysis_scripts/analyze_extended_training.py`

### Batch Size Configuration
Default batch size is 2. When increasing batch size, use linear learning rate scaling:
- BS=4: LR=0.02, warmup=500
- BS=8: LR=0.04, warmup=1000
- BS=16: LR=0.08, warmup=1500

Run batch size ablation: `python scripts/batch_size_ablation.py --analyze`

## Documentation Updates

After completing tasks, always:
1. Update `TODO.md` status numbers and "Last Updated" timestamp
2. Move completed items to "Recently Completed" section
3. Use status emojis: ✅ Complete | 🔄 Running | ⏳ Pending | ❌ Failed | 🔶 Partial

## Git Conventions

```bash
git commit -m "type: Short description"
```
Types: `feat:`, `fix:`, `docs:`, `refactor:`, `chore:`

## Debugging

### Check model num_classes from checkpoint
```python
import torch
ckpt = torch.load('iter_80000.pth', map_location='cpu')
for k, v in ckpt.get('state_dict', ckpt).items():
    if 'conv_seg' in k and 'weight' in k:
        print(f"num_classes = {v.shape[0]}")
```

### Analyze all test results
```bash
python test_result_analyzer.py --root /scratch/aaa_exchange/AWARE/WEIGHTS --comprehensive
```

### Test results format (results.json)
```json
{
    "overall": {"mIoU": 45.23, "aAcc": 92.1},
    "per_domain": {"clear_day": {"mIoU": 48.5}, "rainy": {"mIoU": 42.1}},
    "per_class": {"road": {"IoU": 95.2}, "sidewalk": {"IoU": 72.3}}
}
```
