# PROVE Project TODO

**Last Updated:** 2026-01-29 (22:30)

---

## 🔧 Current Status

### Active Jobs
- **Total:** 12 jobs (0 running, 12 pending)
- **Training:** Stage 1 baseline with Lovasz loss

### New Training Configuration (2026-01-29)
| Setting | Value |
|---------|-------|
| **Batch Size** | 16 |
| **Max Iterations** | 80,000 |
| **Warmup Iterations** | 1,000 |
| **Loss Function** | Lovasz |
| **Checkpoint Interval** | 10,000 |
| **Eval Interval** | 10,000 |
| **LR Scale Factor** | 8.0 (batch_size=16 / base=2) |
| **Best Checkpoint** | Saved based on val/mIoU |
| **Keep Checkpoints** | ALL |

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

### Phase 3: Stage 2 Training (All Domains) ⏳ PENDING

After Stage 1 completes, select top strategies for Stage 2.

---

## 🎯 Proposed Next Steps

### Immediate
1. **Monitor baseline training** - 12 jobs pending
   - Expected duration: ~24-48 hours per job
   - Use `bjobs -w -u mima2416` to check status

2. **Submit remaining std_* strategies** after baseline completes
   - `python scripts/batch_training_submission.py --stage 1 --strategy-type std --seg-loss lovasz`

### After Baseline Complete
3. **Submit gen_* strategies**
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
