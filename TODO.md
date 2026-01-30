# PROVE Project TODO

**Last Updated:** 2026-01-30 (08:15)

---

## 🔧 Current Status

### Active Jobs
- **Total:** 33 jobs (0 running, 33 pending) - all Lovasz loss
- **chge7185:** 6 running Stage 2 Lovasz jobs, many pending

### Current Training Configuration (2026-01-30)
| Setting | Value |
|---------|-------|
| **Batch Size** | 16 |
| **Max Iterations** | 80,000 |
| **Warmup Iterations** | 1,000 |
| **Loss Function** | Lovasz (under evaluation) |
| **Checkpoint Interval** | 5,000 |
| **Eval Interval** | 5,000 |
| **Early Stop Patience** | 5 validations (25k iters) |
| **LR Scale Factor** | 8.0 (batch_size=16 / base=2) |
| **Best Checkpoint** | Saved based on val/mIoU |
| **Keep Checkpoints** | ALL |

---

## ⚠️ Lovasz Loss Findings (2026-01-30)

### Investigation Summary
Analyzed chge7185's Stage 2 training (baseline_bdd10k_deeplabv3plus with Lovasz loss)

### Validation mIoU Progression
```
Iter    mIoU    Best?
5k      35.43   
10k     35.86   ↑ best
15k     34.68   
20k     36.35   ↑ best
25k     35.18   
30k     34.09   
35k     37.56   ↑ best
40k     37.55   
45k     35.81   
50k     33.49   
55k     39.15   ↑ best
60k     35.61   
65k     33.94   
70k     39.69   ↑ best (current best)
75k     39.63   
```

### Key Observations
1. **Oscillating mIoU (33-40%)** - No clear convergence trend
2. **Early stopping not triggered** - keeps finding new bests intermittently
3. **Per-class issues:** 8 classes at 0% IoU (wall, rider, bus, train, motorcycle, bicycle)
4. **Historical CE baseline:** 44-48% mIoU (more stable)

### Possible Causes
- `classes='present'` in Lovasz loss (uneven gradients for rare classes)
- High LR with batch size 16
- Multi-domain training complexity (Stage 2)

### Recommendation
Consider comparing with CrossEntropy loss baseline to determine which is more stable.

### Optimizer & Scheduler
| Model Type | Optimizer | Base LR | Scaled LR (BS=16) |
|------------|-----------|---------|-------------------|
| DeepLabV3+/PSPNet (CNN) | SGD (momentum=0.9) | 0.01 | 0.08 |
| SegFormer (Transformer) | AdamW | 0.00006 | 0.00048 |

---

## ✅ Recently Completed

### 2026-01-29
- [x] **Cleared all old weights** (996 GB removed)
  - WEIGHTS/: 955 GB cleared
  - WEIGHTS_STAGE_2/: 41 GB cleared
- [x] **Killed all running/pending jobs** (54 jobs terminated)
- [x] **Updated training configuration:**
  - batch_size: 8 → 16
  - max_iters: 10,000 → 80,000
  - warmup_iters: 500 → 1,000
  - lr_scale_factor: 4.0 → 8.0
- [x] **Added best checkpoint saving logic:**
  - save_best='val/mIoU' with rule='greater'
  - max_keep_ckpts=-1 (keep all)
- [x] **Aligned checkpoint/eval intervals:**
  - Both at 10,000 iterations for proper best checkpoint selection
- [x] **Submitted Stage 1 baseline jobs with Lovasz loss:**
  - 12 jobs: 4 datasets × 3 models
  - Job IDs: 895295-895306

---

## 📋 Evaluation Completion Checklist

### Phase 1: Clear Old Data ✅
- [x] Delete old WEIGHTS/ (955 GB cleared)
- [x] Delete old WEIGHTS_STAGE_2/ (41 GB cleared)
- [x] Kill all old jobs (54 jobs terminated)

### Phase 2: Stage 1 Training (Clear Day Only) 🔄 IN PROGRESS

**New Configuration: batch_size=16, max_iters=80,000, warmup=1,000, Lovasz loss**

| Strategy | Progress | Status |
|----------|----------|--------|
| baseline | 0/12 (0%) | 🔄 12 jobs pending |
| std_* | 0/42 | ⏳ Pending |
| gen_* | 0/57+ | ⏳ Pending |

**Submitted Jobs (2026-01-29):**
- baseline_bdd10k_* (895295-895297)
- baseline_iddaw_* (895298-895300)
- baseline_mapillaryvistas_* (895301-895303)
- baseline_outside15k_* (895304-895306)

### Phase 3: Stage 2 Training (All Domains) 🔄 IN PROGRESS (chge7185)

| Model | Dataset | Progress | mIoU | Status |
|-------|---------|----------|------|--------|
| DeepLabV3+ | BDD10k | ~84% (67k/80k) | 39.69% | 🔄 Running |
| SegFormer | BDD10k | ~59% (47k/80k) | ~37% | 🔄 Running |
| PSPNet | BDD10k | ~37% (30k/80k) | ~36% | 🔄 Running |

**⚠️ Note:** Stage 2 mIoU is lower than Stage 1 because it trains on ALL weather conditions, including challenging night/rain domains (night domain: ~22.8% mIoU).

---

## 🎯 Proposed Next Steps

### ⚠️ Decision Pending: Loss Function
Based on Lovasz instability findings, decide between:
- **Option A:** Add CrossEntropy comparison jobs (run both)
- **Option B:** Kill Lovasz jobs, switch to CrossEntropy only
- **Option C:** Let Lovasz jobs finish, analyze final results

### Immediate
1. **Monitor baseline training** - 33 jobs pending (Lovasz)
   - Expected duration: ~24-48 hours per job
   - Use `bjobs -w -u mima2416` to check status

2. **Monitor chge7185 Stage 2 jobs** - 6 running
   - DeepLabV3+ at 84% - near completion
   - Watch final mIoU to assess Lovasz convergence

### After Baseline Complete
3. **Submit std_* strategies** (after loss decision)
   - `python scripts/batch_training_submission.py --stage 1 --strategy-type std --seg-loss lovasz`

4. **Submit gen_* strategies**
   - `python scripts/batch_training_submission.py --stage 1 --strategy-type gen --seg-loss lovasz`

4. **Run testing on completed models**
   - `python scripts/auto_submit_tests.py --stage 1 --dry-run`

---

## 🎯 Strategy Lists

### STD_STRATEGIES (7)
| Strategy | Augmentation | Description |
|----------|--------------|-------------|
| baseline | None | Pure baseline, no augmentation |
| std_minimal | RandomCrop + RandomFlip | Geometric augmentation only |
| std_photometric_distort | PhotoMetricDistortion | Color augmentation only |
| std_autoaugment | AutoAugment | Policy-based augmentation |
| std_cutmix | CutMix | Cut-paste augmentation |
| std_mixup | MixUp | Linear interpolation |
| std_randaugment | RandAugment | Random augmentation policy |

### GEN_STRATEGIES (19)
| Rank | Strategy | CQS | FID | mIoU |
|------|----------|-----|-----|------|
| 1 | gen_cycleGAN | -0.78 | 92.65 | 46.76 |
| 2 | gen_flux_kontext | -0.66 | 80.30 | 40.01 |
| 3 | gen_step1x_new | -0.47 | 86.64 | 35.92 |
| 4 | gen_LANIT | -0.29 | 106.24 | 44.48 |
| 5 | gen_albumentations_weather | 0.07 | 123.94 | 43.99 |
| 6 | gen_automold | 0.16 | 121.12 | 33.50 |
| 7 | gen_step1x_v1p2 | 0.18 | 91.63 | 24.61 |
| 8 | gen_VisualCloze | 0.26 | 99.34 | 24.02 |
| 9 | gen_SUSTechGAN | 0.36 | 147.49 | 49.13 |
| 10 | gen_cyclediffusion | 0.39 | 138.77 | 33.18 |
| 11 | gen_IP2P | 0.41 | 114.22 | 28.52 |
| 12 | gen_Attribute_Hallucination | 0.64 | 117.95 | 21.60 |
| 13 | gen_UniControl | 0.73 | 114.90 | 22.51 |
| 14 | gen_CUT | 0.78 | 119.38 | 18.11 |
| 15 | gen_Img2Img | 0.84 | 120.25 | 15.04 |
| 16 | gen_Qwen_Image_Edit | 0.85 | 111.41 | 17.18 |
| 17 | gen_CNetSeg | 1.43 | 120.77 | 13.00 |
| 18 | gen_stargan_v2 | 1.52 | 100.28 | 4.84 |
| 19 | gen_Weather_Effect_Generator | - | - | - |

*Excluded: gen_EDICT, gen_StyleID (no generated images available)*

---

## 🚀 Batch Training Submission

\`\`\`bash
# ============================================
# ALWAYS DRY-RUN FIRST
# ============================================
python scripts/batch_training_submission.py --stage 1 --dry-run

# ============================================
# Submit by Strategy Type
# ============================================
# STD strategies only (84 jobs: 7 × 4 × 3)
python scripts/batch_training_submission.py --stage 1 --strategy-type std

# GEN strategies only (228 jobs: 19 × 4 × 3)
python scripts/batch_training_submission.py --stage 1 --strategy-type gen

# All strategies (312 jobs: 26 × 4 × 3)
python scripts/batch_training_submission.py --stage 1 --strategy-type all

# ============================================
# Submit with Limit
# ============================================
python scripts/batch_training_submission.py --stage 1 --limit 50

# ============================================
# Submit Specific Strategies
# ============================================
python scripts/batch_training_submission.py --stage 1 \\
    --strategies baseline std_minimal gen_cycleGAN

# ============================================
# Submit for Specific Dataset/Model
# ============================================
python scripts/batch_training_submission.py --stage 1 \\
    --datasets BDD10k --models deeplabv3plus_r50
\`\`\`

---

## 📁 File Organization

\`\`\`
/scratch/aaa_exchange/AWARE/
├── WEIGHTS/                    # Stage 1 outputs (DELETE & RETRAIN)
├── WEIGHTS_STAGE_2/            # Stage 2 outputs (DELETE & RETRAIN)  
├── WEIGHTS_RATIO_ABLATION/     # Ratio ablation study
├── WEIGHTS_EXTENDED/           # Extended training study
├── WEIGHTS_COMBINATIONS/       # Combination strategies
├── GENERATED_IMAGES/           # Synthetic images (KEEP)
└── training_locks/             # Lock files for multi-user safety
\`\`\`

---

## ⏱️ Time Estimates

| Phase | Jobs | Time/Job | Parallel | Total |
|-------|------|----------|----------|-------|
| Stage 1 Training | ~100 | ~24-48 hours | 6-10 | ~1-2 weeks |
| Stage 2 Training | ~60 | ~24-48 hours | 6-10 | ~3-5 days |
| Testing | ~160 | ~15 min | 20 | ~2 hours |
| **Total** | | | | **~2 weeks** |

**Note:** Training time increased due to batch_size=16, max_iters=80,000 (vs previous 10,000)

---

## 📝 Quick Commands

\`\`\`bash
# Job Management (LSF)
bjobs -w                          # List all jobs
bjobs -u mima2416 | grep RUN      # Running jobs only
bkill <job_id>                    # Kill specific job
bkill 0                           # Kill ALL jobs

# Testing
python fine_grained_test.py --config path/config.py \\
    --checkpoint path/iter_80000.pth --dataset BDD10k

# Auto-submit tests
python scripts/auto_submit_tests.py --dry-run

# Update trackers
python scripts/update_training_tracker.py --stage 1
python scripts/update_testing_tracker.py --stage 1
\`\`\`

---

## ⚠️ Important Notes

1. **Always dry-run before submitting batch jobs**
2. **Pre-flight checks automatically skip:**
   - Jobs with existing checkpoints
   - Jobs locked by another user
   - gen_* strategies without generated images
3. **File permissions set to 775/664 for multi-user access**
4. **Testing runs automatically after training completes**
