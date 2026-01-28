# PROVE Project TODO

**Last Updated:** 2026-01-28 (15:00)

---

## 🔧 Current Status

### MixedDataLoader Bug - FIXED ✅

The bug where generated images were never loaded has been fixed:
- Each batch now contains exactly N real + M generated samples
- Verified: Job 524893 running with 4 real + 4 generated per batch
- Training loss decreasing normally (2.47 → 0.73)

**All previous gen_* results are INVALID and have been deleted (~4.5 TB).**

---

## 📋 Evaluation Completion Checklist

### Phase 1: Verify Fix (In Progress)
- [x] Implement batch-level ratio enforcement
- [x] Submit test job (524893 - BDD10k/deeplabv3plus_r50/gen_cycleGAN)
- [ ] Wait for job 524893 to complete (~45 min remaining)
- [ ] Run test evaluation on trained model
- [ ] Verify mIoU is reasonable

### Phase 2: Retrain gen_* Strategies

#### Stage 1 (Clear Day Training) - 19 gen_* strategies
**19 strategies × 4 datasets × 3 models = 228 jobs**

| Dataset | Models | Strategies | Total Jobs |
|---------|--------|------------|------------|
| BDD10k | 3 | 19 gen_* | 57 |
| IDD-AW | 3 | 19 gen_* | 57 |
| MapillaryVistas | 3 | 19 gen_* | 57 |
| OUTSIDE15k | 3 | 19 gen_* | 57 |
| **Total Stage 1** | | | **228** |

#### Stage 2 (All Domain Training) - TOP 10 strategies only
**10 strategies × 4 datasets × 3 models = 120 jobs**

Top 10 strategies will be selected based on Stage 1 results:
- Best performers per strategy family (CycleGAN, IP2P, FLUX, Step1X, etc.)
- Aim to include at least 1-2 from each generator family

| Dataset | Models | Strategies | Total Jobs |
|---------|--------|------------|------------|
| BDD10k | 3 | 10 gen_* | 30 |
| IDD-AW | 3 | 10 gen_* | 30 |
| MapillaryVistas | 3 | 10 gen_* | 30 |
| OUTSIDE15k | 3 | 10 gen_* | 30 |
| **Total Stage 2** | | | **120** |

**Grand Total: 348 training jobs** (228 Stage 1 + 120 Stage 2)

### Phase 3: Ratio Ablation (After Phase 2)
**Ratios: 0.00, 0.12, 0.25, 0.38, 0.50, 0.62, 0.75, 0.88 (8 values)**

Select subset for ablation:
- 2-3 best gen_* strategies from Phase 2
- 2 datasets (BDD10k, IDD-AW)
- 2 models (deeplabv3plus_r50, segformer_mit-b5)

**Estimated: 48-96 jobs**

### Phase 4: Testing
- Run `fine_grained_test.py` on all trained models
- Generate per-domain metrics (JSON output)
- Compile results into analysis CSVs

**Estimated: 348+ test jobs (one per trained model)**

### Phase 5: Analysis & Visualization
- Run analysis scripts to generate leaderboards
- Create visualizations for paper
- Update documentation with new findings

---

## 🎯 gen_* Strategy List (19 active)

Strategies ranked by CQS (Composite Quality Score) from [results/generative_quality/generative_quality.csv](results/generative_quality/generative_quality.csv):

| Rank | Strategy | CQS | FID↓ | mIoU↑ | Notes |
|------|----------|-----|------|-------|-------|
| 1 | gen_cycleGAN | -0.78 | 92.65 | 46.76 | Best overall |
| 2 | gen_flux_kontext | -0.66 | 80.30 | 40.01 | Best FID |
| 3 | gen_step1x_new | -0.47 | 86.64 | 35.92 | |
| 4 | gen_LANIT | -0.29 | 106.24 | 44.48 | |
| 5 | gen_albumentations_weather | 0.07 | 123.94 | 43.99 | |
| 6 | gen_automold | 0.16 | 121.12 | 33.50 | |
| 7 | gen_step1x_v1p2 | 0.18 | 91.63 | 24.61 | |
| 8 | gen_VisualCloze | 0.26 | 99.34 | 24.02 | |
| 9 | gen_SUSTechGAN | 0.36 | 147.49 | 49.13 | Best mIoU |
| 10 | gen_cyclediffusion | 0.39 | 138.77 | 33.18 | |
| 11 | gen_IP2P | 0.41 | 114.22 | 28.52 | |
| 12 | gen_Attribute_Hallucination | 0.64 | 117.95 | 21.60 | |
| 13 | gen_UniControl | 0.73 | 114.90 | 22.51 | |
| 14 | gen_CUT | 0.78 | 119.38 | 18.11 | |
| 15 | gen_Img2Img | 0.84 | 120.25 | 15.04 | |
| 16 | gen_Qwen_Image_Edit | 0.85 | 111.41 | 17.18 | |
| 17 | gen_CNetSeg | 1.43 | 120.77 | 13.00 | |
| 18 | gen_stargan_v2 | 1.52 | 100.28 | 4.84 | Worst |
| 19 | gen_Weather_Effect_Generator | - | - | - | Added manually |

*Excluded: gen_EDICT, gen_StyleID (no generated images available)*

**Top 10 Selection Criteria (after Stage 1):**
1. Best mIoU improvement over baseline
2. Best cross-domain robustness
3. Consider CQS ranking as prior

---

## 📊 Valid Results (Keep These)

The following results remain valid (no generated images used):

| Strategy | Stage 1 | Stage 2 | Notes |
|----------|---------|---------|-------|
| baseline | ✅ 12 configs | ✅ 12 configs | Real images only |
| std_autoaugment | ✅ 12 configs | ✅ 12 configs | Pipeline augmentation |
| std_cutmix | ✅ 12 configs | ✅ 12 configs | Pipeline augmentation |
| std_mixup | ✅ 12 configs | ✅ 12 configs | Pipeline augmentation |
| std_randaugment | ✅ 12 configs | ✅ 12 configs | Pipeline augmentation |
| std_photometric_distort | ✅ 12 configs | ✅ 12 configs | Pipeline augmentation |

**Total valid: 144 trained models**

---

## 🚀 Recommended Execution Plan

### Step 1: Verify Current Test Job (Today)
```bash
# Monitor job 524893
bjobs 524893
tail -f /scratch/.../train_524893.out

# When complete, run test
python fine_grained_test.py --config ... --checkpoint iter_10000.pth
```

### Step 2: Create Batch Training Script
```python
# Generate all training jobs for Phase 2
# Use existing unified_training.py with --submit-job
```

### Step 3: Submit in Batches
- Submit 50-100 jobs at a time
- Monitor completion
- Submit next batch

### Step 4: Automated Testing
```bash
# Use auto_submit_tests.py when training completes
python scripts/auto_submit_tests.py --dry-run
python scripts/auto_submit_tests.py --limit 50
```

---

## 📁 File Organization

```
WEIGHTS/                    # Stage 1 (valid: baseline + std_*)
WEIGHTS_STAGE_2/            # Stage 2 (valid: baseline + std_*)
WEIGHTS_RATIO_ABLATION/     # Empty - needs re-run
WEIGHTS_EXTENDED/           # baseline only valid
WEIGHTS_COMBINATIONS/       # std_* combinations valid
```

---

## ⏱️ Time Estimates

| Phase | Jobs | Time per Job | Parallel Jobs | Total Time |
|-------|------|--------------|---------------|------------|
| Phase 2 Training | 348 | 1 hour | 20 | ~17 hours |
| Phase 3 Ratio | 48-96 | 1 hour | 20 | ~5 hours |
| Phase 4 Testing | 400+ | 15 min | 20 | ~5 hours |
| **Total** | | | | **~27 hours** |

*Assuming 20 GPU jobs can run concurrently on cluster*

---

## 📝 Commands Reference

```bash
# ============================================
# Batch Training Submission (NEW)
# ============================================

# Dry run - see what jobs would be submitted
python scripts/batch_training_submission.py --stage 1 --dry-run

# Submit Stage 1 jobs (all gen_* strategies)
python scripts/batch_training_submission.py --stage 1 --limit 50

# Submit Stage 2 jobs (top 10 strategies)
python scripts/batch_training_submission.py --stage 2 --strategies gen_cycleGAN gen_flux_kontext ...

# Submit for specific dataset/model
python scripts/batch_training_submission.py --stage 1 --datasets BDD10k --models deeplabv3plus_r50

# ============================================
# Single Training Submission
# ============================================
python unified_training.py --dataset BDD10k --model deeplabv3plus_r50 \
    --strategy gen_cycleGAN --real-gen-ratio 0.5 --domain-filter clear_day --submit-job

# ============================================
# Testing
# ============================================
python fine_grained_test.py --config path/training_config.py \
    --checkpoint path/iter_10000.pth --dataset BDD10k --output-dir path/test_results

# Auto-submit tests
python scripts/auto_submit_tests.py --dry-run
python scripts/auto_submit_tests.py --limit 50

# ============================================
# Job Management (LSF)
# ============================================
bjobs -w -u mima2416 | grep RUN | wc -l
bjobs -w -u mima2416 | head -30
bkill <job_id>                  # Kill specific job
bkill 0                         # Kill ALL your jobs

# ============================================
# Update Trackers
# ============================================
python scripts/update_training_tracker.py --stage 1
python scripts/update_testing_tracker.py --stage 1
```
