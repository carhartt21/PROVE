# PROVE Project TODO

**Last Updated:** 2026-01-29 (16:30)

---

## 🔧 Current Status

### Augmentation Pipeline - FIXED ✅

Each augmentation strategy now applies ONLY its specific augmentation technique:

| Strategy | Pipeline Augmentations | Type |
|----------|----------------------|------|
| **baseline** | NONE | Baseline (no augmentation) |
| **std_minimal** | RandomCrop + RandomFlip | Geometric only |
| **std_photometric_distort** | PhotoMetricDistortion | Color only |
| **std_autoaugment** | AutoAugment (batch-level) | Batch hook |
| **std_cutmix** | CutMix (batch-level) | Batch hook |
| **std_mixup** | MixUp (batch-level) | Batch hook |
| **std_randaugment** | RandAugment (batch-level) | Batch hook |
| **gen_*** | NONE (uses synthetic images) | Generated |

**Previous models trained with INCORRECT augmentation (RandomCrop+RandomFlip+PhotoMetric on ALL strategies).**
**All previous results need to be retrained.**

---

## 📋 Evaluation Completion Checklist

### Phase 1: Clear Old Data & Retrain From Scratch
- [ ] Delete WEIGHTS/ (313 GB) - trained with incorrect augmentation
- [ ] Delete WEIGHTS_STAGE_2/ (315 GB) - trained with incorrect augmentation
- [ ] Submit Stage 1 training jobs (312 jobs)
- [ ] Submit Stage 2 training jobs (312 jobs)

### Phase 2: Stage 1 Training (Clear Day Only)

**26 strategies × 4 datasets × 3 models = 312 jobs**

| Type | Strategies | Jobs |
|------|------------|------|
| STD (7) | baseline, std_minimal, std_photometric_distort, std_autoaugment, std_cutmix, std_mixup, std_randaugment | 84 |
| GEN (19) | gen_cycleGAN, gen_flux_kontext, gen_step1x_new, ... | 228 |
| **Total** | **26** | **312** |

### Phase 3: Stage 2 Training (All Domains)

**Top 10 strategies × 4 datasets × 3 models = 120 jobs**

Selection criteria after Stage 1:
- Best mIoU improvement over baseline
- Best cross-domain robustness
- At least 1-2 from each generator family

### Phase 4: Testing & Analysis
- Run fine_grained_test.py on all trained models
- Generate per-domain metrics
- Create visualizations for paper

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
| Stage 1 Training | 312 | ~1 hour | 20 | ~16 hours |
| Stage 2 Training | 120 | ~1 hour | 20 | ~6 hours |
| Testing | 432 | ~15 min | 20 | ~6 hours |
| **Total** | | | | **~28 hours** |

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
