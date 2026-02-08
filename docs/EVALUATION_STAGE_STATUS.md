# Evaluation Stage Status

**Last Updated:** 2026-02-09 (00:30)

---

## ⚠️ CRITICAL WARNING: Pre-Fix gen_* Results Invalid

> **MixedDataLoader Bug (Jan 28, 2026):** Generated images were **NEVER LOADED** during training.
> All `gen_*` strategy results from the original training round are **INVALID** — only pipeline augmentation was used.
>
> **Bug Status:** ✅ FIXED | **Retraining:** 🔄 In Progress (50.4% Stage 1)
>
> See [BUG_REPORT](BUG_REPORT_CROSS_DATASET_CONTAMINATION.md) for details.

---

## Overview

| Stage | Training | Testing | Status |
|-------|----------|---------|--------|
| **Stage 1** (retrained) | 🔄 339/672 (50.4%) | 332 results (26 strategies) | 🔄 Active retraining |
| **Stage 2** (retrained) | 🔶 216/288 (75.0%) — stale | Pending retraining | ⏳ Awaiting Stage 1 |
| **Cityscapes Replication** | ✅ Complete (5 models) | ✅ Complete | ✅ Pipeline verified |
| **Cityscapes-Gen** | 🔄 ~9/108 complete | 0 valid (9 buggy retests pending) | 🔄 Active training |

**Active LSF Jobs:** 151 total (7 RUN, 144 PEND)  
**Job Breakdown:** 30 Stage 1 gen | 61 Cityscapes-gen training | 28 Cityscapes-gen retests | 32 Stage 1 fine-grained tests

---

## Stage 1: Clear-Day Domain Training

**Status: 🔄 RETRAINING (50.4% complete)**

### Description
- **Training Domain Filter:** `clear_day` only
- **Weights Directory:** `/scratch/aaa_exchange/AWARE/WEIGHTS/`
- **Purpose:** Train models on clear weather conditions, evaluate cross-domain robustness
- **New strategy count:** 28 strategies × 4 datasets × 6 models = 672 configurations

### Coverage

| Metric | Count | Percentage |
|--------|-------|------------|
| Training Complete | 339/672 | 🔄 50.4% |
| Per-Config Complete (4/4 datasets) | 36/111 | 🔶 32.4% |
| Individual models trained | 327/444 | 🔄 73.6% |
| Testing Complete | 332 results | — |

#### Per-Dataset Training

| Dataset | Complete | Running | Missing |
|---------|----------|---------|---------|
| BDD10K | 98/168 (58.3%) | 0 | 70 |
| IDD-AW | 97/168 (57.7%) | 0 | 71 |
| MapillaryVistas | 73/168 (43.5%) | 0 | 95 |
| OUTSIDE15k | 71/168 (42.3%) | 0 | 97 |

### Strategies (26 active, 2 removed)

| Category | Count | Strategies |
|----------|-------|------------|
| Generative | 21 | gen_Attribute_Hallucination, gen_augmenters, gen_automold, gen_CNetSeg, gen_CUT, gen_cyclediffusion, gen_cycleGAN, gen_flux_kontext, gen_Img2Img, gen_IP2P, gen_LANIT, gen_Qwen_Image_Edit, gen_stargan_v2, gen_step1x_new, gen_step1x_v1p2, gen_SUSTechGAN, gen_TSIT, gen_UniControl, gen_VisualCloze, gen_Weather_Effect_Generator, gen_albumentations_weather |
| Standard | 4 | std_autoaugment, std_cutmix, std_mixup, std_randaugment |
| Baseline | 1 | baseline |

**Note:** `std_photometric_distort` removed (redundant — applied to all strategies as default pipeline augmentation).

### Leaderboard (Fair Comparison Mode — 6 complete configs)

The `--fair` flag filters to (dataset, model) configurations where **all 26 strategies** have test results, ensuring unbiased comparison. Currently 6 of 17 possible configs qualify.

| Rank | Strategy | mIoU | Gain vs Baseline | Num Tests |
|------|----------|------|------------------|-----------|
| 1 | gen_Attribute_Hallucination | 42.56% | +2.90 | 6 |
| 2 | gen_UniControl | 42.51% | +2.84 | 6 |
| 3 | gen_VisualCloze | 42.46% | +2.80 | 6 |
| 4 | gen_automold | 42.45% | +2.79 | 6 |
| 5 | gen_CNetSeg | 42.44% | +2.78 | 6 |
| 6 | gen_Qwen_Image_Edit | 42.38% | +2.72 | 6 |
| 7 | gen_stargan_v2 | 42.32% | +2.66 | 6 |
| 8 | std_autoaugment | 42.25% | +2.59 | 6 |
| 9 | gen_IP2P | 42.23% | +2.56 | 6 |
| 10 | gen_SUSTechGAN | 42.17% | +2.50 | 6 |
| ... | ... | ... | ... | ... |
| 25 | gen_step1x_v1p2 | 40.72% | +1.06 | 6 |
| — | **baseline** | **39.67%** | — | 6 |

**Key Finding:** All 25 non-baseline strategies beat baseline (gains +1.06 to +2.90 pp).

#### Fair vs Default Leaderboard Comparison

> **Warning:** The default (unfair) leaderboard has significant sampling bias. See analysis below.

| Aspect | Default Mode | Fair Mode |
|--------|-------------|-----------|
| Total results used | 332 (6–17 per strategy) | 156 (6 per strategy) |
| Baseline mIoU | 33.63% | 39.67% |
| Max gain over baseline | +8.14 pp | +2.90 pp |
| All strategies beat baseline? | ✅ Yes (25/25) | ✅ Yes (25/25) |

**Why the discrepancy:** In default mode, baseline has 17 tests including weaker model architectures (PSPNet, HRNet on MapillaryVistas/OUTSIDE15k). Strategies with only 6–7 tests (e.g., gen_cyclediffusion, gen_albumentations_weather) happen to only have results for stronger models (Mask2Former, SegFormer), inflating their apparent advantage by ~3×.

**Recommendation:** Always use `--fair` for publication-quality comparisons. As more training completes, the fair config pool will grow beyond 6.

### Stage 1 Key Files
- Training Tracker: [TRAINING_TRACKER_STAGE1.md](TRAINING_TRACKER_STAGE1.md)
- Training Coverage: [TRAINING_COVERAGE_STAGE1.md](TRAINING_COVERAGE_STAGE1.md)
- Testing Tracker: [TESTING_TRACKER.md](TESTING_TRACKER.md)
- Testing Coverage: [TESTING_COVERAGE.md](TESTING_COVERAGE.md)
- Leaderboard: `result_figures/leaderboard/STRATEGY_LEADERBOARD_MIOU_FAIR.md`

---

## Stage 2: All-Domains Training

**Status: ⏳ AWAITING STAGE 1 RETRAINING COMPLETION**

### Description
- **Training Domain Filter:** None (all domains)
- **Weights Directory:** `/scratch/aaa_exchange/AWARE/WEIGHTS_STAGE_2/`
- **Purpose:** Train models on all weather conditions, evaluate domain-inclusive performance

### Coverage (from pre-fix round — stale, Jan 22)

| Metric | Count | Percentage |
|--------|-------|------------|
| Training Complete | 216/288 | 🔶 75.0% (stale) |
| BDD10K | 72/72 | ✅ 100% |
| IDD-AW | 72/72 | ✅ 100% |
| MapillaryVistas | 0/72 | ❌ 0% |
| OUTSIDE15k | 72/72 | ✅ 100% |

**Note:** Stage 2 coverage numbers are from the **pre-fix** round (Jan 22). These models used the buggy MixedDataLoader and will need to be retrained once Stage 1 retraining stabilizes. Priority is currently on Stage 1.

### Stage 2 Key Files
- Training Tracker: [TRAINING_TRACKER_STAGE2.md](TRAINING_TRACKER_STAGE2.md)
- Training Coverage: [TRAINING_COVERAGE_STAGE2.md](TRAINING_COVERAGE_STAGE2.md)
- Testing Tracker: [TESTING_TRACKER_STAGE2.md](TESTING_TRACKER_STAGE2.md)
- Testing Coverage: [TESTING_COVERAGE_STAGE2.md](TESTING_COVERAGE_STAGE2.md)

---

## Cityscapes Pipeline Verification

**Status: ✅ COMPLETE**

### Description
- **Weights Directory:** `/scratch/aaa_exchange/AWARE/WEIGHTS_CITYSCAPES/`
- **Iterations:** 160,000 (baseline, BS=2)
- **Purpose:** Verify PROVE unified training config reproduces reference mmseg results

### Results

| Model | PROVE mIoU | Reference mIoU | Difference |
|-------|-----------|----------------|------------|
| pspnet_r50 | 57.64% | — | Reference |
| deeplabv3plus_r50 | 58.02% | — | Reference |
| hrnet_hr48 | 65.67% | — | Reference |
| segformer_b3 | 79.98% | — | Reference |
| segnext_mscan_b | 81.13% | — | Reference |

**Conclusion:** Pipeline produces expected results on Cityscapes benchmark.

### Cross-Domain Transfer (Cityscapes → ACDC)

See [CITYSCAPES_ACDC_CROSS_DOMAIN_RESULTS.md](CITYSCAPES_ACDC_CROSS_DOMAIN_RESULTS.md) for per-domain breakdown on ACDC (fog, night, rain, snow).

---

## Cityscapes-Gen: Strategy Augmentation on Cityscapes

**Status: 🔄 TRAINING IN PROGRESS**

### Description
- **Weights Directory:** `/scratch/aaa_exchange/AWARE/WEIGHTS_CITYSCAPES/`
- **Iterations:** 160,000
- **Purpose:** Test all 26 strategies on Cityscapes benchmark, then evaluate cross-domain transfer to ACDC

### Coverage

| Metric | Count | Status |
|--------|-------|--------|
| Training configs | 27 × 4 models = 108 | 🔄 ~9 complete |
| Testing (valid results) | 0/9 | ⚠️ 9 buggy (mIoU < 5%), retests submitted |
| Active training jobs | 5 RUN + 56 PEND | 🔄 |
| Retest jobs | 28 PEND | ⏳ |

### Active Running Jobs
| Job | Strategy | Model | Since |
|-----|----------|-------|-------|
| Cityscapes-gen | std_mixup | mask2former | Feb 8 |
| Cityscapes-gen | std_randaugment | pspnet | Feb 8 |
| Cityscapes-gen | std_randaugment | mask2former | Feb 8 |
| Cityscapes-gen | std_randaugment | segnext | Feb 8 |
| Cityscapes-gen | gen_flux_kontext | pspnet | Feb 8 |

### Cityscapes-Gen Key Files
- Training Tracker: [TRAINING_TRACKER_CITYSCAPES_GEN.md](TRAINING_TRACKER_CITYSCAPES_GEN.md)
- Testing Tracker: [TESTING_TRACKER_CITYSCAPES_GEN.md](TESTING_TRACKER_CITYSCAPES_GEN.md)
- Testing Coverage: [TESTING_COVERAGE_CITYSCAPES_GEN.md](TESTING_COVERAGE_CITYSCAPES_GEN.md)

---

## Ablation Studies

### 1. Ratio Ablation Study
**Status:** 🔄 ACTIVE TRAINING

- **Location:** `/scratch/aaa_exchange/AWARE/WEIGHTS_RATIO_ABLATION/`
- **Ratios:** 0.00, 0.12, 0.25, 0.38, 0.50, 0.62, 0.75, 0.88
- **Models:** pspnet_r50, segformer_mit-b5
- **Datasets:** BDD10k, IDD-AW
- **Key Finding (preliminary):** Higher ratios (0.62–0.88) slightly outperform lower ratios
- **Guide:** [RATIO_ABLATION_SUBMISSION_GUIDE.md](RATIO_ABLATION_SUBMISSION_GUIDE.md)

### 2. Extended Training Study
**Status:** ✅ COMPLETE

- **Location:** `/scratch/aaa_exchange/AWARE/WEIGHTS_EXTENDED/`
- **Iterations:** 40k → 160k (20k increments) + 320k
- **Key Finding:** 160k iterations = 75% of gains at 50% compute cost
- **Docs:** [EXTENDED_TRAINING.md](EXTENDED_TRAINING.md)

### 3. Batch Size Ablation
**Status:** ✅ COMPLETE

- **Location:** `/scratch/aaa_exchange/AWARE/WEIGHTS_BATCH_SIZE_ABLATION/`
- **Batch Sizes:** 2, 4, 8, 16 with linear LR scaling

### 4. Loss Function Ablation
**Status:** ✅ COMPLETE

- **Location:** `/scratch/aaa_exchange/AWARE/WEIGHTS_LOSS_ABLATION/`
- **Variants:** aux-lovasz, aux-boundary, aux-focal, loss-lovasz
- **Docs:** [LOSS_CONFIGURATION.md](LOSS_CONFIGURATION.md)

---

## Stage 1 Baseline Analysis

**Publication-ready analysis generated:** 2026-01-23

| Output | Description |
|--------|-------------|
| **Location** | `result_figures/baseline_consolidated/stage1_baseline_output/` |
| **Script** | `result_figures/baseline_consolidated/generate_stage1_baseline.py` |

### Key Findings (pre-fix round, now being validated with retrained models)
| Metric | Value | Notes |
|--------|-------|-------|
| Overall mIoU | 33.3% | Average across 12 configs |
| Domain Gap | 10.1% | Clear Day − Adverse Avg |
| Most Robust Model | SegFormer | Gap 8.7% |
| Hardest Domain | Night | −14.9% from Clear Day |
| Largest Dataset Gap | IDD-AW | 17.6% domain gap |
| Smallest Dataset Gap | Mapillary | 2.6% domain gap |

---

## Useful Commands

### Training Submission
```bash
# ALWAYS use batch_training_submission.py (handles locks, checks, parameters)
python scripts/batch_training_submission.py --stage 1 --dry-run
python scripts/batch_training_submission.py --stage 1 --strategies baseline
python scripts/batch_training_submission.py --stage cityscapes --dry-run
```

### Auto-Submit Tests
```bash
python scripts/auto_submit_tests.py --dry-run        # Stage 1
python scripts/auto_submit_tests_stage2.py --dry-run  # Stage 2
```

### Update Trackers
```bash
python scripts/update_training_tracker.py --stage 1
python scripts/update_training_tracker.py --stage 1 -c   # With coverage report
python scripts/update_training_tracker.py --stage 2
python scripts/update_testing_tracker.py --stage 1
python scripts/update_testing_tracker.py --stage 2
python scripts/update_testing_tracker.py --stage cityscapes-gen
```

### Generate Leaderboards
```bash
python analysis_scripts/generate_stage1_leaderboard.py              # Default (all results)
python analysis_scripts/generate_stage1_leaderboard.py --fair       # Fair comparison (complete configs only)
python analysis_scripts/generate_stage2_leaderboard.py
```

---

## Next Steps

1. **Complete Stage 1 retraining** — 333 remaining configurations (49.6%)
   - Priority: Complete all models for more datasets to expand fair comparison pool beyond 6 configs
2. **Cityscapes-Gen training** — 61 jobs queued, 5 running
   - After completion: retest (28 retests already queued) and generate cross-domain ACDC results
3. **Stage 2 retraining** — Blocked on Stage 1 completion and cluster availability
4. **Ratio ablation completion** — Continue training remaining ratio configurations
5. **Publication preparation** — Finalize figures using `--fair` leaderboard data
