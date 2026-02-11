# Evaluation Stage Status

**Last Updated:** 2026-02-11 (17:00)

---

## ⚠️ CRITICAL WARNING: Pre-Fix gen_* Results Invalid

> **MixedDataLoader Bug (Jan 28, 2026):** Generated images were **NEVER LOADED** during training.
> All `gen_*` strategy results from the original training round are **INVALID** — only pipeline augmentation was used.
>
> **Bug Status:** ✅ FIXED | **Retraining:** 🔄 In Progress (365/444 models complete in Stage 1)
>
> See [BUG_REPORT](BUG_REPORT_CROSS_DATASET_CONTAMINATION.md) for details.

---

## Overview

| Stage | Training | Testing | Status |
|-------|----------|---------|--------|
| **Stage 1** (retrained) | 🔄 365/444 models (82.2%), 53/111 configs fully complete | 374 results | 🔄 69 models pending, 13 running |
| **Stage 2** (retrained) | 🔶 128/444 (28.8%) — mostly pre-fix | 139 results (stale) | ⏳ Awaiting Stage 1 |
| **Cityscapes Replication** | ✅ Complete (5 models) | ✅ Complete | ✅ Pipeline verified |
| **Cityscapes-Gen** | ✅ 100/108 models (92.6%), 25/27 configs complete | 248 results (CS + ACDC) | 🔄 12 models pending |

**Active LSF Jobs:** 42 total (10 RUN, 32 PEND)  
**Job Breakdown:** Primarily Cityscapes-Ratio ablation jobs (step1x_v1p2, flux_kontext, TSIT)

---

## Stage 1: Clear-Day Domain Training

**Status: 🔄 RETRAINING IN PROGRESS (82.2% models complete)**

### Description
- **Training Domain Filter:** `clear_day` only
- **Weights Directory:** `/scratch/aaa_exchange/AWARE/WEIGHTS/`
- **Purpose:** Train models on clear weather conditions, evaluate cross-domain robustness
- **Strategy count:** 26 strategies × 4 datasets (gen: 21, std: 4, baseline: 1)
- **Models:** 4 per config (pspnet_r50, segformer_mit-b3, segnext_mscan-b, mask2former_swin-b)

### Coverage

| Metric | Count | Status |
|--------|-------|--------|
| Configs fully complete (4/4 models) | 53/111 | 47.7% |
| Configs partial | 37/111 | 33.3% |
| Configs running | 13/111 | 11.7% |
| Configs pending | 8/111 | 7.2% |
| Individual models complete | 365/444 | 82.2% |
| Individual models running | 13/444 | 2.9% |
| Individual models pending | 69/444 | 15.5% |
| Testing complete | 374 results | — |

#### Per-Dataset Training

*(See [TRAINING_TRACKER_STAGE1.md](TRAINING_TRACKER_STAGE1.md) for per-strategy detail)*

### Strategies (26 total: 21 gen + 4 std + 1 baseline)

| Category | Count | Strategies |
|----------|-------|------------|
| Generative | 21 | gen_Attribute_Hallucination, gen_augmenters, gen_automold, gen_CNetSeg, gen_CUT, gen_cyclediffusion, gen_cycleGAN, gen_flux_kontext, gen_Img2Img, gen_IP2P, gen_LANIT, gen_Qwen_Image_Edit, gen_stargan_v2, gen_step1x_new, gen_step1x_v1p2, gen_SUSTechGAN, gen_TSIT, gen_UniControl, gen_VisualCloze, gen_Weather_Effect_Generator, gen_albumentations_weather |
| Standard | 4 | std_autoaugment, std_cutmix, std_mixup, std_randaugment |
| Baseline | 1 | baseline |

**Note:** `std_photometric_distort` removed (redundant — applied to all strategies as default pipeline augmentation).

### Leaderboard (Default Mode — 371 results, hrnet_hr48 excluded)

Top-10 and baseline from current Stage 1 leaderboard (26 strategies, 13–16 results each):

| Rank | Strategy | mIoU | Gain vs Baseline | Num Tests |
|------|----------|------|------------------|-----------|
| 1 | gen_UniControl | 40.12% | +2.51 | 14 |
| 2 | gen_Img2Img | 39.99% | +2.37 | 14 |
| 3 | std_autoaugment | 39.99% | +2.37 | 16 |
| 4 | gen_Attribute_Hallucination | 39.94% | +2.33 | 14 |
| 5 | std_cutmix | 39.91% | +2.30 | 16 |
| 6 | gen_Qwen_Image_Edit | 39.86% | +2.24 | 14 |
| 7 | gen_stargan_v2 | 39.86% | +2.25 | 14 |
| 8 | gen_CNetSeg | 39.86% | +2.25 | 14 |
| 9 | std_mixup | 39.81% | +2.20 | 16 |
| 10 | gen_IP2P | 39.79% | +2.17 | 14 |
| — | **baseline** | **37.61%** | — | 16 |

**Key Finding:** All 25 non-baseline strategies beat baseline (gains +0.87 to +2.51 pp).

**Note:** hrnet_hr48 excluded from Stage 1/2 leaderboards (pre-fix legacy results). Results per strategy vary (13–16 tests). Use `generate_strategy_leaderboard.py --stage 1` for full detail.

See: `result_figures/leaderboard/STRATEGY_LEADERBOARD_STAGE1_MIOU.md`

### Stage 1 Key Files
- Training Tracker: [TRAINING_TRACKER_STAGE1.md](TRAINING_TRACKER_STAGE1.md)
- Training Coverage: [TRAINING_COVERAGE_STAGE1.md](TRAINING_COVERAGE_STAGE1.md)
- Testing Tracker: [TESTING_TRACKER.md](TESTING_TRACKER.md)
- Testing Coverage: [TESTING_COVERAGE.md](TESTING_COVERAGE.md)
- Leaderboard: `result_figures/leaderboard/STRATEGY_LEADERBOARD_STAGE1_MIOU.md`

---

## Stage 2: All-Domains Training

**Status: ⏳ AWAITING STAGE 1 RETRAINING COMPLETION**

### Description
- **Training Domain Filter:** None (all domains)
- **Weights Directory:** `/scratch/aaa_exchange/AWARE/WEIGHTS_STAGE_2/`
- **Purpose:** Train models on all weather conditions, evaluate domain-inclusive performance

### Coverage

| Metric | Count | Status |
|--------|-------|--------|
| Configs fully complete (4/4 models) | 3/111 | 2.7% |
| Configs partial | 75/111 | 67.6% |
| Configs running | 1/111 | 0.9% |
| Configs pending | 31/111 | 27.9% |
| Individual models complete | 128/444 | 28.8% |
| Individual models running | 1/444 | 0.2% |
| Individual models pending | 316/444 | 71.2% |
| Valid test results | 139 | — |

**Note:** Stage 2 gen_* models are mostly pre-fix and will need retraining. Priority remains on completing Stage 1 first.

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

**Status: 🔄 TRAINING + RETESTING IN PROGRESS**

### Description
- **Weights Directory:** `/scratch/aaa_exchange/AWARE/WEIGHTS_CITYSCAPES_GEN/`
- **Iterations:** 20,000
- **Models:** 4 (segformer_mit-b3, pspnet_r50, segnext_mscan-b, mask2former_swin-b)
- **Strategies:** 24 (1 baseline + 4 std + 19 gen) × 4 models = 96 configs
- **Purpose:** Test strategies on Cityscapes benchmark, evaluate cross-domain transfer to ACDC

### Coverage

| Metric | Count | Status |
|--------|-------|--------|
| Configs fully complete (4/4 models) | 25/27 | ✅ 92.6% |
| Configs pending | 3/27 | ⏳ |
| Individual models complete | 100/108 | ✅ 92.6% |
| Individual models pending | 12/108 | ⏳ |
| Cityscapes test results | 25 strategies tested | ✅ |
| ACDC cross-domain test results | 25 strategies tested | ✅ |
| Total test results | 248 | ✅ |

### Cityscapes Test Bug (Fixed Feb 9)
- **Bug:** Auto-tests in submitted training jobs used `DATA_ROOT=CITYSCAPES` + `TEST_SPLIT=val` (wrong)
- **Fix:** Should use `DATA_ROOT=FINAL_SPLITS` + `TEST_SPLIT=test`
- **Status:** ✅ Resolved — 248 valid test results collected
- **Retest script:** `scripts/retest_cityscapes_gen.py`

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
- **Models:** pspnet_r50, segformer_mit-b3
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
python scripts/auto_submit_tests.py --stage 1 --dry-run
python scripts/auto_submit_tests.py --stage 2 --dry-run
python scripts/batch_test_submission.py --stage cityscapes-gen --dry-run
```

### Update Trackers
```bash
python scripts/update_training_tracker.py --stage all
python scripts/update_testing_tracker.py --stage all
```

### Generate Leaderboards
```bash
python analysis_scripts/generate_strategy_leaderboard.py --stage all
python analysis_scripts/generate_strategy_leaderboard.py --stage 1   # Stage 1 only
python analysis_scripts/generate_strategy_leaderboard.py --stage cityscapes-gen
```

---

## Next Steps

1. **Complete Stage 1 training** — 365/444 models complete (82.2%), 69 pending + 13 running
   - Use: `python scripts/batch_training_submission.py --stage 1 --resume -y`
2. **Auto-submit Stage 1 tests** — Run after new model completions
   - `python scripts/auto_submit_tests.py --stage 1 --dry-run`
3. **Cityscapes-Gen completion** — 100/108 models complete (92.6%), 12 pending
   - 248 test results collected; remaining 3 strategy configs pending training
4. **Stage 2 retraining** — 128/444 complete but mostly pre-fix; blocked on Stage 1 + cluster availability
5. **Cityscapes-Ratio ablation** — Active (10 RUN + 32 PEND jobs for step1x_v1p2, flux_kontext, TSIT)
6. **Publication preparation** — Finalize figures once Stage 1 training reaches higher completion
