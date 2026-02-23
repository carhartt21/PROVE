# Evaluation Stage Status

**Last Updated:** 2026-02-12 (13:30)

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
| **Stage 1** (retrained) | 🔄 399/448 models (89.1%), 87/112 configs fully complete | 402 results | 🔄 13 running, 34 pending |
| **Stage 2** (retrained) | 🔶 142/400 (35.5%) | 175 results | 🔄 S2 wave active |
| **Cityscapes Replication** | ✅ Complete (5 models) | ✅ Complete | ✅ Pipeline verified |
| **Cityscapes-Gen** | ✅ **100/100 models (100%)**, 25/25 configs complete | **248 results** (CS + ACDC) | ✅ **100% complete** |

**Active LSF Jobs:** S2 wave active (~250 jobs), gen_Attr_Hall deeplabv3plus CG (Job 3076652)  
**Job Breakdown:** S2 gen strategies training, S1 resume completions, CG bonus model

---

## Stage 1: Clear-Day Domain Training

**Status: 🔄 RETRAINING IN PROGRESS (89.1% models complete)**

### Description
- **Training Domain Filter:** `clear_day` only
- **Weights Directory:** `${AWARE_DATA_ROOT}/WEIGHTS/`
- **Purpose:** Train models on clear weather conditions, evaluate cross-domain robustness
- **Strategy count:** 26 strategies × 4 datasets (gen: 21, std: 4, baseline: 1)
- **Models:** 4 per config (pspnet_r50, segformer_mit-b3, segnext_mscan-b, mask2former_swin-b)

### Coverage

| Metric | Count | Status |
|--------|-------|--------|
| Configs fully complete (4/4 models) | 87/112 | 77.7% |
| Configs partial | 12/112 | 10.7% |
| Configs running | 5/112 | 4.5% |
| Configs pending | 8/112 | 7.1% |
| Individual models complete | 399/448 | 89.1% |
| Individual models running | 13/448 | 2.9% |
| Individual models pending | 36/448 | 8.0% |
| Testing complete | 413 results | — |

#### Per-Dataset Training

*(See [TRAINING_TRACKER_STAGE1.md](TRAINING_TRACKER_STAGE1.md) for per-strategy detail)*

### Strategies (26 total: 21 gen + 4 std + 1 baseline)

| Category | Count | Strategies |
|----------|-------|------------|
| Generative | 21 | gen_Attribute_Hallucination, gen_augmenters, gen_automold, gen_CNetSeg, gen_CUT, gen_cyclediffusion, gen_cycleGAN, gen_flux_kontext, gen_Img2Img, gen_IP2P, gen_LANIT, gen_Qwen_Image_Edit, gen_stargan_v2, gen_step1x_new, gen_step1x_v1p2, gen_SUSTechGAN, gen_TSIT, gen_UniControl, gen_VisualCloze, gen_Weather_Effect_Generator, gen_albumentations_weather |
| Standard | 4 | std_autoaugment, std_cutmix, std_mixup, std_randaugment |
| Baseline | 1 | baseline |

**Note:** `std_photometric_distort` removed (redundant — applied to all strategies as default pipeline augmentation).

### Leaderboard (Default Mode — 413 results, hrnet_hr48 excluded)

Top-10 and baseline from current Stage 1 leaderboard (26 strategies, 14–17 results each):

| Rank | Strategy | mIoU | Gain vs Baseline | Num Tests |
|------|----------|------|------------------|-----------|
| 1 | gen_automold | 40.45% | +2.84 | 17 |
| 2 | gen_UniControl | 40.37% | +2.76 | 16 |
| 3 | gen_albumentations_weather | 40.35% | +2.73 | 17 |
| 4 | gen_cyclediffusion | 40.12% | +2.51 | 16 |
| 5 | gen_Qwen_Image_Edit | 40.12% | +2.51 | 16 |
| 6 | gen_stargan_v2 | 40.09% | +2.48 | 16 |
| 7 | std_randaugment | 40.04% | +2.43 | 16 |
| 8 | gen_Img2Img | 40.04% | +2.42 | 16 |
| 9 | std_autoaugment | 39.99% | +2.37 | 16 |
| 10 | gen_CNetSeg | 39.94% | +2.33 | 16 |
| — | **baseline** | **37.61%** | — | 16 |

**Key Finding:** All 25 non-baseline strategies beat baseline (gains +1.42 to +2.84 pp).

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

**Status: 🔄 RETRAINING ACTIVE (35.5% models complete)**

### Description
- **Training Domain Filter:** None (all domains)
- **Weights Directory:** `${AWARE_DATA_ROOT}/WEIGHTS_STAGE_2/`
- **Purpose:** Train models on all weather conditions, evaluate domain-inclusive performance

### Coverage

| Metric | Count | Status |
|--------|-------|--------|
| Configs fully complete (4/4 models) | 18/100 | 18.0% |
| Configs partial | 50/100 | 50.0% |
| Configs running | 8/100 | 8.0% |
| Configs pending | 24/100 | 24.0% |
| Individual models complete | 142/400 | 35.5% |
| Individual models running | 8/400 | 2.0% |
| Individual models pending | 250/400 | 62.5% |
| Valid test results | 195 | — |

**Note:** S2 wave now active with ~250 jobs submitted. Uses 25 strategies (excludes std_cutmix, std_mixup, gen_cyclediffusion).

### Stage 2 Key Files
- Training Tracker: [TRAINING_TRACKER_STAGE2.md](TRAINING_TRACKER_STAGE2.md)
- Training Coverage: [TRAINING_COVERAGE_STAGE2.md](TRAINING_COVERAGE_STAGE2.md)
- Testing Tracker: [TESTING_TRACKER_STAGE2.md](TESTING_TRACKER_STAGE2.md)
- Testing Coverage: [TESTING_COVERAGE_STAGE2.md](TESTING_COVERAGE_STAGE2.md)

---

## Cityscapes Pipeline Verification

**Status: ✅ COMPLETE**

### Description
- **Weights Directory:** `${AWARE_DATA_ROOT}/WEIGHTS_CITYSCAPES/`
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

**Status: ✅ COMPLETE (100%)**

### Description
- **Weights Directory:** `${AWARE_DATA_ROOT}/WEIGHTS_CITYSCAPES_GEN/`
- **Iterations:** 20,000
- **Models:** 4 (segformer_mit-b3, pspnet_r50, segnext_mscan-b, mask2former_swin-b) + bonus deeplabv3plus_r50
- **Strategies:** 25 (1 baseline + 4 std + 20 gen) — excludes gen_LANIT (no CS images), std_minimal, std_photometric_distort
- **Purpose:** Test strategies on Cityscapes benchmark, evaluate cross-domain transfer to ACDC

### Coverage

| Metric | Count | Status |
|--------|-------|--------|
| Configs fully complete (4/4 models) | **25/25** | ✅ **100%** |
| Individual models complete | **100/100** | ✅ **100%** |
| Cityscapes test results | 25 strategies tested | ✅ |
| ACDC cross-domain test results | 25 strategies tested | ✅ |
| Total test results | **248** | ✅ |

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

- **Location:** `${AWARE_DATA_ROOT}/WEIGHTS_RATIO_ABLATION/`
- **Ratios:** 0.00, 0.12, 0.25, 0.38, 0.50, 0.62, 0.75, 0.88
- **Models:** pspnet_r50, segformer_mit-b3
- **Datasets:** BDD10k, IDD-AW
- **Key Finding (preliminary):** Higher ratios (0.62–0.88) slightly outperform lower ratios
- **Guide:** [RATIO_ABLATION_SUBMISSION_GUIDE.md](RATIO_ABLATION_SUBMISSION_GUIDE.md)

### 2. Extended Training Study
**Status:** ✅ COMPLETE

- **Location:** `${AWARE_DATA_ROOT}/WEIGHTS_EXTENDED/`
- **Iterations:** 40k → 160k (20k increments) + 320k
- **Key Finding:** 160k iterations = 75% of gains at 50% compute cost
- **Docs:** [EXTENDED_TRAINING.md](EXTENDED_TRAINING.md)

### 3. Batch Size Ablation
**Status:** ✅ COMPLETE

- **Location:** `${AWARE_DATA_ROOT}/WEIGHTS_BATCH_SIZE_ABLATION/`
- **Batch Sizes:** 2, 4, 8, 16 with linear LR scaling

### 4. Loss Function Ablation
**Status:** ✅ COMPLETE

- **Location:** `${AWARE_DATA_ROOT}/WEIGHTS_LOSS_ABLATION/`
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

1. **Complete Stage 1 training** — 399/448 models complete (89.1%), 34 pending + 13 running
   - Use: `python scripts/batch_training_submission.py --stage 1 --resume -y`
2. **Auto-submit Stage 1 tests** — Run after new model completions
   - `python scripts/auto_submit_tests.py --stage 1 --dry-run`
3. **Cityscapes-Gen** — ✅ **100% complete** (100/100 models, 248 tests)
   - gen_Attribute_Hallucination/deeplabv3plus bonus training submitted (Job 3076652)
4. **Stage 2 retraining** — 142/400 complete; S2 wave now active (8 RUN, 243 PEND)
5. **Cityscapes-Ratio ablation** — Active (37/60 trained, 37/37 CS+ACDC tested)
6. **Publication preparation** — Finalize figures once Stage 1 training reaches higher completion
