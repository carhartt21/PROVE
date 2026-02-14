# PROVE Project TODO

**Last Updated:** 2026-02-14 (03:50)

---

## 📊 Current Status (2026-02-14 03:50)

### Queue Summary
| User | Category | RUN | PEND | Total |
|------|----------|----:|-----:|------:|
| chge7185 | S2 (incl failed retrain) | 6 | 12 | 18 |
| **chge7185 subtotal** | | **6** | **12** | **18** |
| mima2416 | S2 (completing) | 26 | 0 | 26 |
| mima2416 | Noise ablation (resubmitted) | 0 | 53 | 53 |
| **mima2416 subtotal** | | **26** | **53** | **79** |

**Notes:**
- 🔄 **S1 at 91%**: 408/448 individual models complete (tracker). 420/420 tested. Remaining: mask2former on MapVistas/OUTSIDE15k (on chge7185 queue).
- 🔄 **S2 at 70%**: **280/400 complete** (was 235). 26 RUN on mima2416 + 18 on chge7185. **13 gen model failures** (gen_Attribute_Hallucination, gen_augmenters, gen_automold, gen_CNetSeg) — all being retrained. Testing: **339 valid** (was 301), 8 missing.
- ✅ **CS-Ratio COMPLETE**: **48/48 trained + 96/96 tested (100%)**.
- 🔄 **Combination at 94%**: 17/18 complete + tested. Last job status unknown (pending S2 jobs were killed).
- ❌ **Noise RESUBMITTED (after bug fix)**: 53 jobs queued (24 noise-50% + 24 noise-100% + 5 CG noise-100%). Fix committed as `a48dd18`. See §CRITICAL BUG section below.
- ✅ **Extended S1 COMPLETE**: 20/20 at 45k, all tested.
- 🔄 **Extended CG**: 5/10 at 60k (tested), 5 at 50k. 4 pspnet models running.

---

### Training Progress
| Stage | Complete (models) | In-Progress | Pending | Coverage |
|-------|-------------------|-------------|---------|----------|
| Stage 1 (15k) | **408/448** | 6 | 34 | **91.1%** |
| Stage 2 (15k) | **280/400** | 14+13 fail retrain | 94 | 🔄 **70.0%** → ~97% |
| CG total (20k) | **100/100** | 0 | 0 | **100%** ✅ |
| CS-Ratio Ablation (20k) | **48/48** | 0 | 0 | **100%** ✅ |
| S1-Ratio Ablation (15k) | **24/24** | 0 | 0 | **100%** ✅ |
| Combination Ablation (20k) | **17/18** | 0 | 1 (PEND) | 🔄 **94.4%** |
| Noise Ablation 50% (15k) | 0/24 | 24 PEND | 0 | 🔄 Resubmitted (fix a48dd18) |
| Noise Ablation 100% (15k/20k) | 0/29 | 29 PEND | 0 | 🔄 Resubmitted (fix a48dd18) |
| Extended S1 (45k) | **20/20** | 0 | 0 | **100%** ✅ |
| Extended CG (60k) | **5/10** | 4 RUN | 1 | 🔄 **50%** |

### Testing Progress
| Stage | Valid Tests | Trained | Notes |
|-------|------------|---------|-------|
| Stage 1 | **420** | 408 | **100% coverage** ✅ |
| Stage 2 | **339** | 280 | Auto-test on completion, **8 missing** |
| CG | **250** | 100 | **100%** ✅ (Cityscapes + ACDC per model) |
| CS-Ratio | **96** | 48 | **100%** ✅ (Cityscapes + ACDC per model) |
| S1-Ratio | **24** | 24 | **100%** ✅ |
| Combination | **17** | 17 | **100%** ✅ |
| Noise 50% | ❌ INVALID | — | Bug: pipeline injection (resubmitted 24 jobs) |
| Noise 100% | ❌ INVALID | — | Bug: pipeline injection (resubmitted 29 jobs) |
| Extended S1 | **20** | 20 | **100%** ✅ |
| Extended CG | **5** | 5 | **100%** of completed ✅ (auto-tested) |

### 100% Coverage Plan

#### S1 Path to 100% — IN PROGRESS 🔄
| Step | Items | Status | Notes |
|------|-------|--------|-------|
| 1. S1 testing of trained models | **420/420** | ✅ **100%** | 0 missing |
| 2. Remaining S1 training | 40 configs | 🔄 6 RUN, 34 PEND | mask2former on MapVistas/OUTSIDE15k + remaining gen_* |
| 3. Submit tests for new completions | Auto | ⏳ After step 2 | `auto_submit_tests.py --stage 1` |

### Strategy Leaderboard Highlights (2026-02-13 23:12)
| Stage | Top Strategy | mIoU | Baseline mIoU | Strategies > Baseline | Results |
|-------|-------------|------|---------------|----------------------|--------|
| Stage 1 | gen_automold | **40.45%** | 37.61% | **25/25 (all!)** | 417 |
| Stage 1 #2 | gen_UniControl | 40.37% | 37.61% | — | 417 |
| Stage 1 #3 | gen_albumentations_weather | 40.35% | 37.61% | — | 417 |
| Stage 2 | gen_LANIT | **41.76%** | 40.85% | **6/25** | 344 |
| Stage 2 #2 | gen_step1x_new | 41.11% | 40.85% | — | 344 |
| Stage 2 #3 | gen_flux_kontext | 41.11% | 40.85% | — | 344 |
| CG overall | gen_Img2Img | **52.87%** | 52.65% | **5/25** | 254 |
| CG #2 | gen_augmenters | 52.82% | 52.65% | — | 254 |
| CG #3 | gen_Qwen_Image_Edit | 52.67% | 52.65% | — | 254 |

**Key findings:**
- S1: **All 25** augmentation strategies beat baseline (+1.42 to +2.84 pp). **gen_automold #1**. Top-5: gen_automold, gen_UniControl, gen_albumentations_weather, gen_augmenters, gen_Qwen_Image_Edit
- S2: gen_LANIT #1 but only 4 tests (unreliable). **Reliable (16 tests):** gen_flux_kontext=gen_step1x_new (+0.25). **6/25** beat baseline (gen_albumentations_weather newly above at +0.01pp).
- CG: gen_Img2Img leads (+0.22 pp). **5/25** beat baseline — CG effect sizes much smaller than S1/S2.
- **❌ Noise finding RETRACTED (BUG):** All noise results invalid — pipeline injection bug meant models trained on real images, not noise. Fix committed (`a48dd18`), 53 jobs resubmitted. See §CRITICAL BUG section.
- Full leaderboards: `result_figures/leaderboard/`

### Suggested Next Steps (Priority Order)

1. **🚨 Wait for noise resubmission** — 53 jobs queued (24 noise-50% + 24 noise-100% + 5 CG noise-100%). Verify first completed job log has "Injected ReplaceWithNoise into dataset pipeline" message.
2. **Copy CS-Ratio to IEEE repo** — 48/48 trained + 96/96 tested (ready now).
3. **Wait for S2 completion** — 26 RUN (mima2416) + 18 (chge7185). Then regenerate S2 leaderboard + copy to IEEE repo.
4. **Wait for ExtCG** — 4 pspnet models running toward 60k → complete Extended CG ablation (10/10).
5. ~~Wait for CS-Ratio~~ → ✅ **48/48 COMPLETE + 96/96 tested**.
6. **Wait for Combination** — 1 job status unknown (check if it survived pending kills).
7. **After noise completion:** Analyze 50% vs 100% noise vs gen_* (genuine results this time).
8. **After S2 completion:** Copy S2 results to IEEE publication repo.

---

## 🔄 Active / In-Progress Tasks

### mask2former on MapillaryVistas/OUTSIDE15k (§2)

50 S1 configs — ALL mask2former_swin-b on MapillaryVistas (25) + OUTSIDE15k (25).
**Status:** 6 running, 34 pending. Main bottleneck for S1 100% coverage.
```bash
python scripts/batch_training_submission.py --stage 1 --dry-run  # Shows remaining
```

### Submit S2 Training (§4b)

Submit Stage 2 training for 10 selected strategies (data-driven selection from S1+CG cross-stage analysis).
```bash
python scripts/batch_training_submission.py --stage 2 \
  --strategies gen_step1x_new gen_Img2Img gen_Qwen_Image_Edit \
    gen_UniControl gen_augmenters gen_CUT \
    std_autoaugment std_cutmix \
    gen_flux_kontext gen_cycleGAN \
  --dry-run
```

**S2 Strategy Selection (10 strategies, 4 tiers):**

| Tier | Strategy | S1 Rank | CG Rank | Family | Rationale |
|------|----------|---------|---------|--------|-----------|
| **1 (must)** | gen_Attribute_Hallucination | #4 | **#1** | Instruct/Edit | Cross-stage champion, ACDC +1.89 pp |
| **1 (must)** | gen_Img2Img | #3 | **#2** | Diffusion I2I | Consistent top-3 both stages |
| **1 (must)** | gen_Qwen_Image_Edit | #8 | **#4** | Instruct/Edit | Third consistent top-tier |
| **2 (family)** | gen_UniControl | **#1** | #17 | Instruct/Edit | S1 champion — tests S1→S2 transfer |
| **2 (family)** | gen_augmenters | #10 | **#3** | Domain-specific | Best domain-specific family member |
| **2 (family)** | gen_CUT | #16 | **#6** | GAN | Best GAN in CG, well-cited method |
| **3 (std)** | std_autoaugment | **#2** | #11 | Standard | All-positive per-dataset gains |
| **3 (std)** | std_cutmix | **#5** | #10 | Standard | Best night-domain, smallest domain gap |
| **4 (diversity)** | gen_flux_kontext | #13 | **#8** | Diffusion | Modern architecture, tests novelty |
| **4 (diversity)** | gen_cycleGAN | #21 | #12 | GAN | Classic baseline, needed for literature |

**Compute:** 10 strategies × 4 models × 4 datasets = **160 jobs** at 15k iters. Minimal (Tier 1+2): 96 jobs.

### ❌ CRITICAL BUG: Noise Ablation Pipeline Injection (2026-02-14)

**Bug:** In `unified_training.py` lines 528–545, the `ReplaceWithNoise` transform was injected into `cfg.train_pipeline` (a top-level convenience variable) instead of `cfg.train_dataloader.dataset.pipeline` (the actual pipeline used by the dataset). After `Config.fromfile()` resolves variable references, these are **separate objects** — modifying one does NOT affect the other.

**Evidence:**
- Training log for `bdd10k/deeplabv3plus_r50_ratio0p00` confirms:
  - `train_pipeline` has `ReplaceWithNoise` ✓
  - `train_dataloader.dataset.pipeline` does NOT have `ReplaceWithNoise` ✗
  - "Injected" message never printed (0 occurrences)
- Git diff shows the fix was applied in this session, NOT before the training runs

**Impact:** ALL noise ablation experiments (49 training + 54 testing) are **INVALID**:
- Noise 100% (ratio0p00): models trained on **real clear-day images** from cycleGAN manifest paths, NOT noise
- Noise 50% (ratio0p50): models trained on **real + real** (different subsets), NOT real + noise
- This explains why "noise" performed better than baseline — it was essentially additional real data

**Fix applied:** Committed as `a48dd18` — `unified_training.py` now injects into `cfg.train_dataloader.dataset.pipeline` (the correct target).

**Remediation completed (2026-02-14 03:50):**
1. ✅ Fix committed (`a48dd18`)
2. ✅ Invalid weights deleted: `WEIGHTS_NOISE_ABLATION/gen_random_noise/` (237 GB) + `WEIGHTS_CITYSCAPES_GEN/gen_random_noise/` (36 GB) = **273 GB freed**
3. ✅ All 53 noise jobs resubmitted with fixed code:
   - 24 noise-50% (ratio 0.50): 4 datasets × 6 models → job IDs 3454599–3454910
   - 24 noise-100% (ratio 0.00): 4 datasets × 6 models → job IDs 3455031–3455054
   - 5 CG noise-100% (ratio 0.00): Cityscapes × 5 models → job IDs 3455255–3455259
4. ✅ "Landmark finding" retracted (sections below marked RETRACTED)
5. ⏳ Verification pending: check first completed job logs for "Injected ReplaceWithNoise into dataset pipeline" message

---

### Noise Ablation Study — 50% Ratio (§6) — ❌ RETRACTED (BUG)

**Status:** ❌ **INVALID — Pipeline injection bug (see §CRITICAL BUG above)**

24 jobs: `gen_random_noise` strategy × 4 datasets × 6 models at 15k iters with `clear_day` domain filter.
Baselines skipped (identical to existing S1 baselines — same config).

**Implementation:** Replaces generated images with uniform random noise (`np.random.randint(0, 256, shape)`), preserving label maps. Uses cycleGAN reference manifest for image shapes. Tests whether augmentation gains come from meaningful content or just more training samples.

#### Final Results (24/24, 2026-02-13 15:00) ✅
**Finding: random noise helps most models (regularization effect).**

> **Note:** HRNet excluded from analysis — S1 baseline mIoU suspiciously low (15–21% vs 27–50% for other models), inflating noise gains (+12–13pp). See §HRNet Exclusion below.

| Dataset/Model | Baseline (15k) | Noise (15k) | Diff |
|---|---:|---:|---:|
| bdd10k/deeplabv3plus_r50 | — | 41.66 | — |
| bdd10k/pspnet_r50 | 30.03 | 41.32 | **+11.29** |
| bdd10k/segformer_mit-b3 | 46.25 | 46.97 | **+0.71** |
| bdd10k/segnext_mscan-b | 41.27 | 48.43 | **+7.15** |
| bdd10k/mask2former_swin-b | 50.18 | 51.93 | **+1.75** |
| iddaw/deeplabv3plus_r50 | — | 34.97 | — |
| iddaw/pspnet_r50 | 33.25 | 33.19 | -0.06 |
| iddaw/segformer_mit-b3 | 34.02 | 39.07 | **+5.05** |
| iddaw/segnext_mscan-b | 35.10 | 39.01 | **+3.92** |
| iddaw/mask2former_swin-b | 39.68 | 41.83 | **+2.15** |
| mapvistas/deeplabv3plus_r50 | — | 27.77 | — |
| mapvistas/pspnet_r50 | 29.03 | 29.05 | +0.02 |
| mapvistas/segformer_mit-b3 | 27.70 | 34.82 | **+7.12** |
| mapvistas/segnext_mscan-b | 34.64 | 34.64 | 0.00 |
| mapvistas/mask2former_swin-b | 40.83 | 35.17 | **-5.66** |
| outside15k/deeplabv3plus_r50 | — | 34.39 | — |
| outside15k/pspnet_r50 | 36.02 | 35.90 | -0.12 |
| outside15k/segformer_mit-b3 | 36.87 | 43.16 | **+6.29** |
| outside15k/segnext_mscan-b | 38.72 | 42.91 | **+4.20** |
| outside15k/mask2former_swin-b | 44.96 | 44.72 | **-0.24** |
| **Average (16 with baselines)** | | | **+2.72 pp** |

**Noise > baseline in 11/16 models, neutral in 1, negative in 4.** Average gain = **+2.72 pp**, median = **+1.95 pp**.

**Notable negative case:** mapvistas/mask2former_swin-b dropped **-5.66 pp** — the only significant regression. outside15k/mask2former_swin-b dropped **-0.24 pp** (marginal). These large-capacity models may be sensitive to noise corruption.

**Interpretation:** The gains (especially pspnet +11.29) suggest baseline models were undertrained or overfitting at 15k with limited data. Adding ANY extra images (even noise) acts as regularization. This finding is critical context for interpreting gen_* gains — part of the improvement may come from data volume/regularization rather than meaningful weather-domain content.

#### HRNet Exclusion
HRNet (hrnet_hr48) excluded from all leaderboards and analysis due to suspiciously low S1 baseline results:
- baseline/mapillaryvistas: **15.24%** (vs 27–41% for other models)
- baseline/outside15k: **19.81%** (vs 36–45%)
- baseline/iddaw: **20.69%** (vs 33–40%)
- baseline/bdd10k: **failed** (no checkpoints produced)
- HRNet was never part of `STAGE_1_MODELS` (only in `ALL_MODELS`), trained on ad-hoc basis
- Noise gains of +12–13pp are artifacts of broken baseline, not meaningful signal
- Excluding HRNet reduces noise average from +4.35pp → **+2.72pp** and median from +3.92 → **+1.95pp**

#### 100% Noise Ablation (ratio 0.00) — ❌ RETRACTED (BUG)
**Rationale:** At ratio 0.50 (50% real + 50% noise), noise provides +2.52pp avg gain — almost identical to gen_* strategies (+2.52pp). This raises the question: **is real data even necessary when the "augmentation" is pure noise?**

- **Design:** 5 models × 5 datasets = **25 jobs** (5×5 design)
- **Models:** pspnet_r50, segformer_mit-b3, segnext_mscan-b, mask2former_swin-b, deeplabv3plus_r50
- **Datasets:** BDD10k, IDD-AW, MapillaryVistas, OUTSIDE15k (15k iters), Cityscapes (20k iters)
- **Ratio:** `real_gen_ratio=0.0` → 100% noise images per batch, 0% real data
- **Status:** ✅ COMPLETE — 20/20 S1 (iter 15k) + 5/5 CG (iter 20k) + 30/30 tested

#### 100% Noise Results (2026-02-14)

**S1 Results (16 models with baselines, excl HRNet + deeplabv3plus w/o baseline):**

| Dataset/Model | Baseline | Noise 50% | Noise 100% | 50-BL | 100-BL | 100-50 |
|---|---:|---:|---:|---:|---:|---:|
| bdd10k/mask2former_swin-b | 51.57 | 51.93 | 49.64 | +0.36 | -1.93 | -2.29 |
| bdd10k/pspnet_r50 | 30.03 | 41.32 | 41.39 | +11.29 | +11.36 | +0.07 |
| bdd10k/segformer_mit-b3 | 46.25 | 46.97 | 46.35 | +0.72 | +0.10 | -0.62 |
| bdd10k/segnext_mscan-b | 41.27 | 48.43 | 48.01 | +7.16 | +6.74 | -0.42 |
| iddaw/mask2former_swin-b | 41.59 | 41.83 | 41.55 | +0.24 | -0.04 | -0.28 |
| iddaw/pspnet_r50 | 33.25 | 33.19 | 33.39 | -0.06 | +0.14 | +0.20 |
| iddaw/segformer_mit-b3 | 34.02 | 39.07 | 38.91 | +5.05 | +4.89 | -0.16 |
| iddaw/segnext_mscan-b | 35.10 | 39.01 | 38.97 | +3.91 | +3.87 | -0.04 |
| mapvistas/mask2former_swin-b | 40.83 | 35.17 | 39.81 | -5.66 | -1.02 | +4.64 |
| mapvistas/pspnet_r50 | 29.03 | 29.05 | 29.07 | +0.02 | +0.04 | +0.02 |
| mapvistas/segformer_mit-b3 | 27.70 | 34.82 | 34.84 | +7.12 | +7.14 | +0.02 |
| mapvistas/segnext_mscan-b | 34.64 | 34.64 | 34.55 | +0.00 | -0.09 | -0.09 |
| outside15k/mask2former_swin-b | 44.96 | 44.72 | 46.35 | -0.24 | +1.39 | +1.63 |
| outside15k/pspnet_r50 | 36.02 | 35.90 | 36.13 | -0.12 | +0.11 | +0.23 |
| outside15k/segformer_mit-b3 | 36.87 | 43.16 | 42.55 | +6.29 | +5.68 | -0.61 |
| outside15k/segnext_mscan-b | 38.72 | 42.91 | 42.99 | +4.19 | +4.27 | +0.08 |
| **AVERAGE** | | | | **+2.52** | **+2.67** | **+0.15** |
| Positive count | | | | 11/16 | 12/16 | 8/16 |

**CG Results (Cityscapes in-domain + ACDC cross-domain):**

| Model | BL-CS | Noise-CS | Diff-CS | BL-ACDC | Noise-ACDC | Diff-ACDC |
|---|---:|---:|---:|---:|---:|---:|
| deeplabv3plus_r50 | 57.26 | 58.26 | +1.00 | 36.98 | 35.82 | -1.16 |
| mask2former_swin-b | 68.98 | 69.74 | +0.76 | 51.11 | 51.64 | +0.53 |
| pspnet_r50 | 57.57 | 57.50 | -0.07 | 36.36 | 35.51 | -0.85 |
| segformer_mit-b3 | 63.38 | 63.60 | +0.22 | 45.60 | 46.44 | +0.84 |
| segnext_mscan-b | 64.33 | 63.71 | -0.62 | 44.88 | 43.57 | -1.31 |
| **AVERAGE** | | | **+0.26** | | | **-0.39** |

**⚠️ ~~LANDMARK FINDING~~ RETRACTED (BUG) — All results below are invalid. Models trained on real images, NOT noise. See §CRITICAL BUG.**

| Method | S1 Avg Gain | N Samples | Interpretation |
|--------|----------:|----------:|----------------|
| **Noise 100%** | **+2.67 pp** | 16 | Pure noise beats everything |
| **Noise 50%** | **+2.52 pp** | 16 | 50% real + 50% noise |
| std_* avg | +1.82 pp | 72 (4 strategies) | Standard augmentation |
| gen_* avg | +1.74 pp | 379 (21 strategies) | Generative augmentation |

**Key conclusions:**
1. **100% noise ≈ 50% noise (+0.15pp)** — real data adds nothing beyond label layouts at 50%; removing it entirely makes no difference or slightly helps
2. **Noise >> gen_* (+0.93pp)** — random noise outperforms ALL 21 gen_* strategies on average. Even best gen_* (gen_UniControl at +2.31) is below noise 50% (+2.52)
3. **Noise >> std_* (+0.85pp)** — random noise outperforms all 4 std_* strategies
4. **CG tells a different story** — noise 100% helps in-domain (+0.26pp) but **HURTS ACDC cross-domain (-0.39pp)**. For well-trained Cityscapes models, noise is NOT helpful for cross-domain.
5. **S1 baselines are undertrained** — the massive pspnet gains (+11pp) show models converge insufficiently at 15k with limited clear-day data. Any extra samples = regularization.
6. **For the paper:** Most S1 cross-domain augmentation gains come from **regularization/data diversity in label layouts**, NOT from meaningful weather-domain visual content in generated images.

### Analysis & Paper Figures (§9)

Once Stage 1 is ~100% complete:
```bash
python analysis_scripts/generate_strategy_leaderboard.py --stage 1
python analysis_scripts/generate_strategy_leaderboard.py --stage 2
python analysis_scripts/generate_strategy_leaderboard.py --stage cityscapes-gen
python analysis_scripts/analyze_strategy_families.py
python analysis_scripts/analyze_domain_gap_corrected.py
```

---

## 📐 Ablation Study Designs

### §7: Ratio Ablation Studies

Two ratio ablation stages implemented in `batch_training_submission.py`:

#### Cityscapes Ratio Ablation (48 jobs — 96% complete)
- **Status:** ✅ **48/48 complete + tested (96/96)** — both stalled resume jobs finished (2026-02-13).
- **Ratios:** 0.0, 0.25, 0.75 (0.5 and 1.0 already available from cityscapes-gen)
- **Strategies:** 3 Diffusion (gen_VisualCloze, gen_step1x_v1p2, gen_flux_kontext) + 1 GAN (gen_TSIT)
- **Models:** pspnet_r50, segformer_mit-b3, segnext_mscan-b, mask2former_swin-b

##### Preliminary Results (2026-02-13, 38/48 trained + tested)
**Conclusion: Finer-grained ratios (0.125, 0.375, etc.) NOT warranted for Cityscapes.**

| Metric | Cityscapes (in-domain) | ACDC (cross-domain) |
|--------|----------------------|---------------------|
| Avg spread across ratios | ~0.5 pp | **1.4 pp** |
| Non-linearity (mean deviation from linear) | 0.44 pp | 0.61 pp |
| Significant deviations (>1pp) | 2/35 (6%) | 5/35 (14%) |
| Best ratio distribution | N/A | 0.00: 25%, 0.25: 25%, 0.50: 31%, 0.75: 19% |

**Key observations:**
1. **Spread too small** — 1.4 pp ACDC spread. Adding points at 0.125/0.375 would measure noise, not signal.
2. **No consistent optimal ratio** — peaks uniformly distributed across 0.0-0.75, no "sweet spot" to zoom into.
3. **Curve shapes noisy** — 44% valley@0.75 (Cityscapes), 31% valley@0.75 (ACDC), but inconsistent across strategy/model combinations.
4. **Cityscapes near-optimal baseline** — in-domain performance barely changes with ratio (baseline best for 10/16 Cityscapes curves).
5. **S1 datasets have 10x larger effect sizes** — cross-domain ratio analysis belongs on BDD10k/IDD-AW.

Analysis script: `analysis_scripts/analyze_ratio_preliminary.py`

```bash
python scripts/batch_training_submission.py --stage cityscapes-ratio --dry-run
```

#### Stage 1 Ratio Ablation (24 jobs — COMPLETE ✅)
- **Status:** ✅ 24/24 trained + tested (completed 2026-02-11).
- **Ratios:** 0.0, 0.25, 0.75
- **Datasets:** BDD10k, IDD-AW
- **Strategies:** gen_VisualCloze (Diffusion), gen_TSIT (GAN)
- **Models:** pspnet_r50, segformer_mit-b3

##### Results Summary
**Conclusion: Ratio effect is weak on S1 datasets too (~0.5 pp spread). No finer ratios needed.**

| Strategy | Dataset | Model | 0.00 | 0.25 | 0.75 | Spread |
|----------|---------|-------|-----:|-----:|-----:|-------:|
| gen_TSIT | BDD10k | pspnet | 41.05 | 41.36 | 41.37 | 0.32 pp |
| gen_TSIT | BDD10k | segformer | 47.54 | 46.84 | 46.39 | 1.15 pp |
| gen_TSIT | IDD-AW | pspnet | 32.99 | 33.29 | 33.14 | 0.30 pp |
| gen_TSIT | IDD-AW | segformer | 38.83 | 39.11 | 38.92 | 0.28 pp |
| gen_VisualCloze | BDD10k | pspnet | 41.03 | 41.02 | 40.95 | 0.08 pp |
| gen_VisualCloze | BDD10k | segformer | 46.29 | 46.08 | 46.91 | 0.83 pp |
| gen_VisualCloze | IDD-AW | pspnet | 33.15 | 33.24 | 33.24 | 0.09 pp |
| gen_VisualCloze | IDD-AW | segformer | 38.81 | 38.88 | 38.94 | 0.13 pp |

**Average spread: ~0.4 pp.** The earlier estimate of "20% spread" from S1 was for absolute mIoU variation across *strategies*, not ratio sensitivity within a strategy. The ratio within any single strategy makes minimal difference.
```bash
python scripts/batch_training_submission.py --stage stage1-ratio --dry-run
python scripts/batch_training_submission.py --stage stage1-ratio -y
```

| Study | Spread | Jobs | Status |
|-------|--------|------|--------|
| Cityscapes-ratio | 1.4 pp (ACDC) | 48 | ✅ **100% COMPLETE** (48/48 trained + 96/96 tested) |
| Stage1-ratio | ~0.4 pp | 24 | ✅ **100% COMPLETE** |

### §7b: Combination Ablation Study (gen_* + std_*)

Tests synergy between generative augmentation (gen_*) and standard augmentation (std_*) strategies. Uses **Option A**: std_* transforms applied to BOTH real and generated images.

#### Design Rationale
- **Hypothesis:** gen_* provides diverse weather conditions; std_* adds photometric/spatial variation → combined effect may be synergistic
- **Key observation:** std_* shows *negative* effect in S2 all-domain training but *positive* in S1 cross-domain → combination ablation tests if gen_*+std_* synergy exists

#### Strategy Selection (data-driven, 2026-02-11)

**gen_* selection** — top cross-stage performers from different families:

| gen_* Strategy | CG Rank | S1 Rank | Family | Selection Rationale |
|----------------|---------|---------|--------|---------------------|
| gen_Qwen_Image_Edit | **#3** | #8 | Instruct/Edit | Consistent top-tier both stages |
| gen_Img2Img | **#1** | #4 | Diffusion I2I | CG champion, consistent top-5 |
| gen_augmenters | **#2** | #10 | Domain-specific | Best in family, reliable |

**std_* selection** — based on S1 per-dataset consistency and cross-model robustness:

| std_* Strategy | S1 Rank | All-Positive Gains? | Cross-Model Std | Selection Rationale |
|----------------|---------|---------------------|-----------------|---------------------|
| std_cutmix | #5 | ✓ Yes | 3.75 | Best night-domain (29.19%), smallest gap |
| std_autoaugment | #2 | ✓ Yes | 3.77 | Most consistent overall |
| std_mixup | #9 | — | **3.42** (lowest) | Lowest cross-model variance |

**Changes from previous design (now using code-confirmed strategies):**
- gen_Attribute_Hallucination → **gen_Qwen_Image_Edit** (consistent instruct/edit performer, available manifests)
- gen_VisualCloze → **gen_Img2Img** (gen_Img2Img is CG #1, S1 #4)
- std_randaugment → **std_autoaugment** (S1 #5 with all-positive gains)

#### Design Matrix

| | std_cutmix (region) | std_autoaugment (auto) | std_mixup (feature) |
|---|---|---|---|
| **gen_Qwen_Image_Edit** | Instruct+Region | Instruct+Auto | Instruct+Feature |
| **gen_Img2Img** | Diffusion+Region | Diffusion+Auto | Diffusion+Feature |
| **gen_augmenters** | DomSpec+Region | DomSpec+Auto | DomSpec+Feature |

Each cell × 2 models (pspnet_r50, segformer_mit-b3) = **18 jobs total** on Cityscapes at 20k iters.

#### Configuration
| Parameter | Value |
|-----------|-------|
| gen_* strategies | gen_Qwen_Image_Edit, gen_Img2Img, gen_augmenters (3) |
| std_* strategies | std_cutmix, std_autoaugment, std_mixup (3) |
| Models | pspnet_r50, segformer_mit-b3 (2) |
| Dataset | Cityscapes |
| Ratio | 0.50 (fixed) |
| Max iters | 20,000 |
| **Total jobs** | 3 × 3 × 2 = **18 jobs** |

#### Output Directory
```
/scratch/aaa_exchange/AWARE/WEIGHTS_COMBINATION_ABLATION/
  {gen_strategy}+{std_strategy}/cityscapes/{model}_ratio0p50/
```

#### Commands
```bash
python scripts/batch_training_submission.py --stage combination --dry-run
python scripts/batch_training_submission.py --stage combination -y
```

#### Analysis Questions
1. **Synergy vs redundancy:** Does gen_* + std_* outperform either alone?
2. **Best combination:** Which gen_*/std_* pair gives highest ACDC cross-domain?
3. **Diminishing returns:** Does combining 2 augmentation types hit a ceiling?
4. **Family interaction:** Do instruct/edit gen_* benefit more from std_* than domain-specific gen_*?

#### Preliminary Results (17/18 complete, 2026-02-13)

**Cityscapes in-domain mIoU (vs baseline=57.57/pspnet, 63.38/segformer):**

| Combination | pspnet | vs_base | segformer | vs_base |
|---|---:|---:|---:|---:|
| gen_Img2Img+std_cutmix | **58.43** | **+0.86** | **64.35** | **+0.96** |
| gen_augmenters+std_cutmix | 58.06 | +0.49 | 64.18 | +0.80 |
| gen_Qwen_Image_Edit+std_cutmix | 58.38 | +0.81 | N/A | N/A |
| gen_Img2Img+std_autoaugment | 56.97 | -0.60 | 63.88 | +0.50 |
| gen_augmenters+std_autoaugment | 56.45 | -1.12 | 63.62 | +0.24 |
| gen_Qwen_Image_Edit+std_autoaugment | 56.85 | -0.72 | 63.64 | +0.26 |
| gen_Img2Img+std_mixup | 53.65 | **-3.92** | 61.15 | **-2.23** |
| gen_augmenters+std_mixup | 53.70 | -3.87 | 61.54 | -1.84 |
| gen_Qwen_Image_Edit+std_mixup | 54.26 | -3.31 | 60.75 | -2.63 |

**Key findings from combination ablation:**
1. **std_cutmix is the CLEAR WINNER** — consistently positive synergy with all gen_* (+0.49 to +0.96 pp vs baseline, +0.16 to +1.33 pp vs gen_only)
2. **std_mixup HURTS dramatically** — -1.84 to -3.92 pp below baseline (destroys signal from both gen_* and real data)
3. **std_autoaugment is model-dependent** — positive for segformer (+0.24 to +0.50 pp), negative for pspnet (-0.60 to -1.12 pp)
4. **Best combination:** gen_Img2Img + std_cutmix (pspnet=58.43, segformer=64.35 — both best overall)
5. **gen_* family doesn't matter much** — all three gen_* strategies show similar pattern with each std_*

### §7c: Extended Training Ablation Study

**Status:** 🔄 Running (2026-02-13). S1: **20/20 COMPLETE** ✅ at 45k (all tested). CG: 3/10 complete at 60k (tested), 6 in-progress.

#### Complete Results — Extended S1 (20/20, 2026-02-13)
| Strategy/Dataset/Model | 15k mIoU | 45k mIoU | Diff |
|---|---:|---:|---:|
| baseline/bdd10k/pspnet | 30.03 | 31.17 | +1.14 |
| baseline/bdd10k/segformer | 46.25 | 46.56 | +0.31 |
| baseline/iddaw/pspnet | 33.25 | 33.27 | +0.01 |
| baseline/iddaw/segformer | 34.02 | 39.17 | **+5.16** |
| gen_Img2Img/bdd10k/pspnet | 40.98 | 41.19 | +0.21 |
| gen_Img2Img/bdd10k/segformer | 45.88 | 46.13 | +0.25 |
| gen_Img2Img/iddaw/pspnet | 33.28 | 33.24 | -0.03 |
| gen_Img2Img/iddaw/segformer | 38.96 | 39.15 | +0.19 |
| gen_augmenters/bdd10k/pspnet | 41.15 | 41.32 | +0.17 |
| gen_augmenters/bdd10k/segformer | 46.15 | 46.38 | +0.23 |
| gen_augmenters/iddaw/pspnet | 32.99 | 33.02 | +0.03 |
| gen_augmenters/iddaw/segformer | 38.95 | 39.07 | +0.12 |
| gen_cycleGAN/bdd10k/pspnet | 40.99 | 41.21 | +0.22 |
| gen_cycleGAN/bdd10k/segformer | 39.95 | 46.62 | **+6.66** |
| gen_cycleGAN/iddaw/pspnet | 33.39 | 33.40 | +0.01 |
| gen_cycleGAN/iddaw/segformer | 38.91 | 39.10 | +0.18 |
| std_randaugment/bdd10k/pspnet | 40.87 | 40.97 | +0.10 |
| std_randaugment/bdd10k/segformer | 46.64 | 46.81 | +0.16 |
| std_randaugment/iddaw/pspnet | 33.15 | 33.18 | +0.03 |
| std_randaugment/iddaw/segformer | 38.92 | 39.09 | +0.17 |

**Complete Analysis (20/20):**
- **Most augmented models gain <0.3 pp** from 3× training → augmentation already converged at 15k
- **Two outliers:** baseline/iddaw/segformer (+5.16) and gen_cycleGAN/bdd10k/segformer (+6.66) were undertrained at 15k
- **Augmentation gap persists at 3× training:**
  - BDD10k/pspnet: baseline 31.17 vs gen_Img2Img 41.19 = **10.02 pp gap** (was 10.95 at 15k — gap narrowed only 0.93 pp)
  - BDD10k/segformer: baseline 46.56 vs gen_augmenters 46.38 = gap nearly closed (but augmented was already near ceiling)
  - IDD-AW/segformer: baseline 39.17 vs gen_Img2Img 39.15 = gap closed (baseline undertrained at 15k)
- **Conclusion:** Augmentation provides genuine signal for models that were already well-converged (pspnet). For undertrained baselines (iddaw/segformer), extended training closes much of the gap, but a meaningful residual remains for pspnet.

Tests whether augmentation benefits persist, grow, or diminish with extended training (3× standard iterations).

#### Research Question
Do augmentation gains diminish with more training? If augmentations just help models converge faster, extended baseline training would close the gap. If augmentations provide genuine additional training signal, gains should persist.

#### Design

**Stage 1 Extended** (`--stage extended-s1`): 20 jobs
| Parameter | Value |
|-----------|-------|
| Datasets | BDD10k, IDD-AW |
| Strategies | baseline, gen_Img2Img, gen_augmenters, gen_cycleGAN, std_randaugment |
| Models | pspnet_r50, segformer_mit-b3 |
| Iterations | 15,000 → 45,000 (3× standard, checkpoint every 5k) |
| Source | Resume from WEIGHTS/ iter_15000.pth |
| Domain filter | clear_day (same as standard S1) |

**Cityscapes-Gen Extended** (`--stage extended-cg`): 10 jobs
| Parameter | Value |
|-----------|-------|
| Dataset | Cityscapes |
| Strategies | baseline, gen_augmenters, gen_Img2Img, gen_CUT, std_randaugment |
| Models | pspnet_r50, segformer_mit-b3 |
| Iterations | 20,000 → 60,000 (3× standard, checkpoint every 5k) |
| Source | Resume from WEIGHTS_CITYSCAPES_GEN/ iter_20000.pth |

**Total: 30 jobs** (all source checkpoints verified ✅)

#### Strategy Selection Rationale
- **S1:** Top gen_* (gen_Img2Img, gen_augmenters, gen_cycleGAN), best std_* (std_randaugment), baseline
- **CG:** Top gen_* (gen_augmenters, gen_Img2Img, gen_CUT) based on fresh leaderboard, best std_* (std_randaugment), baseline

#### Output
```
/scratch/aaa_exchange/AWARE/WEIGHTS_EXTENDED_ABLATION/
  stage1/{strategy}/{dataset}/{model}/
  cityscapes_gen/{strategy}/cityscapes/{model}/
```

#### Commands
```bash
python scripts/batch_training_submission.py --stage extended-s1 --dry-run
python scripts/batch_training_submission.py --stage extended-cg --dry-run
python scripts/batch_training_submission.py --stage extended-s1 -y
python scripts/batch_training_submission.py --stage extended-cg -y
```

#### Status: Extended S1: ✅ **20/20 COMPLETE** (all tested). Extended CG: 🔄 **5/10 at 60k** (5 tested), 5 at 50k — **4 pspnet models RUN** (baseline, gen_augmenters, gen_Img2Img, std_randaugment).

### Old Ablation Studies (Reference Only)

| Study | Path | Status |
|-------|------|--------|
| Ratio Ablation | `WEIGHTS_RATIO_ABLATION/` | 284 ckpts (old regime, ❌ gen_* invalid due to earlier bug) |
| Extended Training | `WEIGHTS_EXTENDED/` | 970 ckpts (old regime, baseline-only valid) |
| Combinations | `WEIGHTS_COMBINATIONS/` | 53 ckpts (IDD-AW only, std_* valid) |
| Batch Size Ablation | `WEIGHTS_BATCH_SIZE_ABLATION/` | BS 2/4/8/16 with LR scaling |

---

## 📝 Research Notes: Generated Image Pipeline (2026-02-09)

### Architecture Summary

The generated image training pipeline has **3 layers**:

1. **Config layer** (`unified_training_config.py`): When `strategy='gen_X'` is selected, `build()` validates the manifest CSV exists, checks dataset coverage, and populates `config['mixed_dataloader']` with ratio, paths, and `dataset_filter`.

2. **Manifest layer**: Two manifest formats coexist:
   - **CSV** (`/scratch/.../GENERATED_IMAGES/{method}/manifest.csv`): Per-image mappings with columns `gen_path, original_path, name, domain, dataset, target_domain`. Generated by `tools/generate_manifests.py`. **This is what training consumes.**
   - **JSON** (`generated_manifests/{method}_manifest.json`): Metadata/summary only (domain counts, match rates). Generated by `scripts/generate_manifests.py`. **Not used at training time.**

3. **Runtime layer** (`unified_training.py` → inline `MixedBatchSampler`):
   - Builds the real dataset via MMEngine Runner
   - Loads `GeneratedImagesManifest(manifest.csv)`, filters by dataset
   - Derives label path from original: `/images/` → `/labels/` + suffix swap
   - **Appends generated entries to real `data_list`** (indices 0..N-1=real, N..N+M-1=gen)
   - Creates `MixedBatchSampler` with infinite cycling: each batch has exactly `int(BS * ratio)` real + remainder generated
   - Replaces runner's train dataloader and calls `runner.train()`

### Key Design Decisions
- **Labels reused from originals** — generative models only transform RGB (weather change), so original segmentation maps are valid
- **No generated-image-specific transforms** — all label transforms in `custom_transforms.py` handle real dataset formats
- **`dataset_filter` prevents cross-dataset contamination** — without it, ALL datasets from a manifest would be loaded
- **`MixedDataLoader` class exists but is NOT used at runtime** — the actual mechanism is the inline `MixedBatchSampler` in the generated training script

### Observations & Potential Issues
1. **CSV vs JSON manifest divergence**: `scripts/generate_manifests.py` creates JSON only; `tools/generate_manifests.py` creates both CSV+JSON. If CSVs become stale while JSONs are refreshed, training sees outdated mappings.
2. **Label path resolution is fragile**: The `/images/` → `/labels/` string replacement + suffix guessing works for current datasets but would break for datasets with different directory conventions.
3. **Ratio edge cases**: When `real_gen_ratio=0.0`, `real_per_batch=0` → 100% generated. When `real_gen_ratio=1.0`, the whole mixed dataloader is skipped. Both are handled correctly.

### Proposed Follow-ups
- [ ] Audit CSV vs JSON manifest consistency — verify `tools/generate_manifests.py` CSVs are current for all 25 methods
- [ ] Verify batch composition at runtime — spot-check a training log for `MixedBatchSampler` output confirming correct real/gen counts

---

## ✅ Completed Tasks Archive

### 2026-02-14
- ✅ **CRITICAL BUG FOUND & FIXED** — `ReplaceWithNoise` injected into wrong config attribute (`cfg.train_pipeline` instead of `cfg.train_dataloader.dataset.pipeline`). All 49 noise training runs were invalid (models trained on real images). Fix committed (`a48dd18`).
- ✅ **Invalid noise data deleted** — `WEIGHTS_NOISE_ABLATION/gen_random_noise/` (237 GB) + `WEIGHTS_CITYSCAPES_GEN/gen_random_noise/` (36 GB) = **273 GB freed**.
- ✅ **53 noise jobs resubmitted** — 24 noise-50% (ratio 0.50) + 24 noise-100% (ratio 0.00) + 5 CG noise-100% (ratio 0.00). Job IDs: 3454599–3455259.
- ✅ **Pending S2 killed on mima2416** — 26 RUN remain (will complete). S2 continuing from chge7185 queue.
- ✅ **"Landmark finding" retracted** — noise > gen_* comparison was artifact of bug (models used real data, not noise).

### 2026-02-13
- ~~✅ **Noise 100% COMPLETE**~~ → ❌ **RETRACTED** — results invalid due to pipeline injection bug (see 2026-02-14).
- ✅ **CS-Ratio COMPLETE** — 48/48 trained + 96/96 tested (100%). Both stalled resume jobs (gen_TSIT/pspnet, gen_flux_kontext/segnext) finished and auto-tested.
- ✅ **S2 progressed to 70%** — 280/400 models complete (was 235). Testing: 339 valid (was 301). 13 gen model failures being retrained (38 jobs queued).
- ✅ **S2 leaderboard updated** — 6/25 strategies beat baseline (was 5/23). gen_albumentations_weather newly above baseline (+0.01pp). 344 total results.
- ✅ **CG leaderboard updated** — 5/25 strategies beat baseline (was 4/24). 254 total results.
- ✅ **Extended CG resuming** — 4 pspnet models running toward 60k.
- ✅ **S2 remaining 99 jobs submitted** (3216823–3216945) — all configs with generated images now in pipeline. S2 target: ~388/400 (97%).
- ✅ **Verified results copied to IEEE repo** — S1 (420), CG (250), S1-Ratio (24/24), Extended S1 (20/20), Combination (17/18) → `/home/mima2416/repositories/-IEEE-Access-01-26-Data-Augmentation/data/data/`
- ✅ **HRNet excluded from all analysis** — suspiciously low baselines (15–21% vs 27–50%). Removed from `analyze_noise_ablation.py`, `generate_strategy_leaderboard.py` (CG stage). Noise avg corrected: +4.35→**+2.72 pp**.
- ❌ ~~**Noise vs gen_* comparison**~~ — RETRACTED. Results invalid (pipeline injection bug). Resubmitted 2026-02-14.
- ❌ ~~**100% Noise ablation submitted**~~ — RETRACTED. Results invalid (pipeline injection bug). Resubmitted 2026-02-14.
- ❌ ~~**Noise 50% ablation COMPLETE**~~ — RETRACTED. Results invalid (pipeline injection bug). Resubmitted 2026-02-14.
- ✅ **Extended S1 100% COMPLETE** — 20/20 trained to 45k + tested. Augmentation gap persists at 3× training.
- ✅ **S2 training at 70.0%** (280/400). Testing: 339 total (8 missing). 6/25 strategies beat baseline. 13 gen model failures being retrained.
- ✅ **CS-Ratio resume jobs completed** — both stalled configs (gen_TSIT/pspnet, gen_flux_kontext/segnext) finished + auto-tested. **CS-Ratio now 48/48 + 96/96 (100%).**
- ✅ **Extended CG resume jobs submitted** — 5 models at 50k, need to reach 60k.
- ✅ **Combination missing model submitted** — gen_Qwen_Image_Edit+std_cutmix/segformer (PEND).
- ✅ **Duplicate noise jobs killed** — 20 accidental duplicates (3203487-3203606) removed.
- ✅ **S2 leaderboard updated** — gen_LANIT #1 (4 tests, unreliable). Reliable: gen_flux_kontext=gen_step1x_new (+0.25, 16 tests). gen_IP2P dropped to -2.06 pp.
- ✅ **Legacy 80k models COMPLETE** — 3 segnext_mscan-b/bdd10k models (43.4-43.8%).
- ✅ **Extended CG: 5/10 at 60k** (5 tested). gen_CUT/pspnet and std_randaugment/segformer newly completed.
- ✅ **Fixed `analyze_strategy_families.py`** — 5 bugs: strategy name mappings, seaborn 0.13.2 boxplot crash, baseline reference loading, pandas groupby error.
- ✅ **Fixed `analyze_combination_ablation.py`** — 6 bugs: weights path, JSON parsing, model normalization, combination type classification.

### 2026-02-12
- ✅ **3 legacy 80k models COMPLETE** — gen_albumentations_weather (43.61%), gen_automold (43.42%), gen_step1x_v1p2 (43.77%)
- ✅ Fixed combination ablation lock contention — lock file names + pre-submission check now include `std_strategy`
- ✅ Fixed & resubmitted combination ablation — type validation bug + lock contention fix, 13 remaining jobs resubmitted
- ✅ Fixed IDD-AW leaderboard bug — `'idd-aw'` → `'iddaw'` in S1/S2 stage configs (commit `4b3529c`)
- ✅ Analyzed mask2former paradox — root cause: rare vehicle class memorization in BDD10k
- ✅ Fixed CG ACDC double-counting — 124 duplicate results removed (commit `d614141`)
- ✅ Aligned CG tracker — excluded gen_LANIT (no images), std_minimal, std_photometric_distort
- ✅ Updated all trackers + leaderboards for all stages
- ✅ Implemented extended training ablation in `batch_training_submission.py` (30 jobs)
- ✅ S1 testing reached 100% coverage (420 valid, 0 missing)

### 2026-02-11
- ✅ S1 & CG analysis complete — corrected leaderboards, per-dataset/model/domain breakdown
- ✅ Cross-stage consistency analysis (Spearman ρ=0.184 — low!)
- ✅ Data-driven S2 strategy subset selected (10 strategies, 4 tiers)
- ✅ Combination ablation strategies revised based on full S1+CG analysis
- ✅ CS-Ratio ACDC fix — `batch_training_submission.py` auto-test includes `cityscapes-ratio` stage

### 2026-02-10
- ✅ Removed std_photometric_distort strategy — killed 12 jobs, deleted weights, updated configs
- ✅ Added `--stage all` to tracker/leaderboard scripts
- ✅ Created `batch_test_submission.py` for comprehensive test job submission
- ✅ Consolidated leaderboard scripts into `generate_strategy_leaderboard.py --stage {1,2,cityscapes-gen}`
- ✅ Consolidated test submission scripts into `auto_submit_tests.py --stage {1,2,cityscapes,cityscapes-gen}`
- ✅ Fixed mask2former pre-flight detection (commit `babff1d`)
- ✅ Submitted 183 Stage 2 training jobs
- ✅ Killed 168 pending S2 jobs — will select strategy subset based on S1/CG results
- ✅ CG gen_* training at **100%** (80/80); CG ACDC testing complete (124 results)

### 2026-02-08
- ✅ Stage 1 gap analysis — found 26 stale trainings, fixed testing tracker IDD-AW bug, deleted 27 wrong-model dirs (~90GB)
- ✅ Fixed testing tracker IDD-AW directory naming (`idd-aw` → `iddaw`)
- ✅ Fixed `max_iters` config regex to handle both keyword and dict formats
- ✅ Cityscapes-Gen evaluation stage — fixed 2 training bugs + 5 testing bugs, all manifests verified
- ✅ Extended tracker scripts with `--stage cityscapes-gen` support (commit `cf4480c`)
- ✅ Created `scripts/retest_cityscapes_gen.py` (commit `67709bb`)
- ✅ Submitted 28 cityscapes-gen retests with proper bash script files (fixed quoting bug)
- ✅ Repository cleanup — archived 7 outdated docs + 13 obsolete scripts

### 2026-02-04
- ✅ Training tracker improvements — reads `max_iters` from config, per-model counts, default 15k iters

### 2026-02-03
- ✅ Mask2Former Swin-B integration — batch_size=8, max_iters=10k, lr=0.0004, exclusive_process GPU mode
- ✅ Fixed validation shape mismatch bug (commit `8a4948c`)

### 2026-02-02
- ✅ Added `--resume` flag to batch_training_submission.py
- ✅ Fixed MapillaryVistas/OUTSIDE15k validation bug — resize to (512,512) (commit `81053c8`)
- ✅ Submitted 26 priority coverage jobs (IDD-AW +433%, OUTSIDE15k +433%)

### 2026-02-01
- ✅ New training defaults — max_iters 80k→15k, checkpoint/eval interval 5k→2k
- ✅ Cityscapes training stage added to batch_training_submission.py (160k iters, 5 models)
- ✅ CLI arguments for checkpoint/eval intervals, auto LR scaling
- ✅ Cross-domain testing script created (`test_cityscapes_replication_on_acdc.py`)
- ✅ Full Stage 1 resubmission with 15k iter config (570 jobs)

### 2026-01-31
- ✅ **CRITICAL FIX:** Added RandomResize(0.5-2.0x) to all training pipelines
- ✅ Verified via Cityscapes replication — SegFormer 79.98%, SegNeXt 81.22%
- ✅ Added SegNeXt MSCAN-B to available models

### 2026-01-30
- ✅ Diagnosed pipeline bug (missing RandomResize)
- ✅ Refactored training loss CLI to single `--aux-loss`
- ✅ Lovasz loss investigation — oscillating mIoU (33-40%), switched to CrossEntropy

### 2026-01-29
- ✅ Cleared all old weights (996 GB) and killed 54 jobs
- ✅ Updated training config — batch_size 8→16, max_iters 10k→80k, warmup 500→1000

---

## 📎 Reference

### Strategy Lists

**STD_STRATEGIES (5):** baseline, std_autoaugment, std_cutmix, std_mixup, std_randaugment

**GEN_STRATEGIES (19):** gen_cycleGAN, gen_flux_kontext, gen_step1x_new, gen_LANIT, gen_albumentations_weather, gen_automold, gen_step1x_v1p2, gen_VisualCloze, gen_SUSTechGAN, gen_cyclediffusion, gen_IP2P, gen_Attribute_Hallucination, gen_UniControl, gen_CUT, gen_Img2Img, gen_Qwen_Image_Edit, gen_CNetSeg, gen_stargan_v2, gen_Weather_Effect_Generator

**Models (6, 4 active for S1/S2):** deeplabv3plus_r50, pspnet_r50, segformer_mit-b3, segnext_mscan-b, ~~hrnet_hr48~~ *(excluded — broken baseline)*, mask2former_swin-b

### Quick Commands

```bash
# Job Management (LSF)
bjobs -w                          # List all jobs
bjobs -u mima2416 | grep RUN      # Running jobs only
bkill <job_id>                    # Kill specific job
bkill 0                           # Kill ALL jobs

# Batch Training (ALWAYS dry-run first!)
python scripts/batch_training_submission.py --stage 1 --dry-run
python scripts/batch_training_submission.py --stage 2 --dry-run
python scripts/batch_training_submission.py --stage cityscapes-ratio --dry-run
python scripts/batch_training_submission.py --stage combination --dry-run

# Testing
python fine_grained_test.py --config path/config.py \
    --checkpoint path/iter_80000.pth --dataset BDD10k
python scripts/auto_submit_tests.py --stage 1 --dry-run
python scripts/batch_test_submission.py --stage cityscapes-gen --dry-run

# Update trackers
python scripts/update_training_tracker.py --stage all
python scripts/update_testing_tracker.py --stage all
python analysis_scripts/generate_strategy_leaderboard.py --stage all
```

### Important Notes

1. **Always dry-run before submitting batch jobs**
2. **Pre-flight checks automatically skip:** jobs with existing checkpoints, locked jobs, gen_* without images
3. **File permissions set to 775/664 for multi-user access**
4. **Testing runs automatically after training completes**
5. **NEVER write temp files to `/tmp`** — use project dir or `/scratch/aaa_exchange/AWARE/`

### Publication Data Directory

Results with 100% coverage must be copied to the IEEE paper repository for analysis and figure generation:

```
/home/mima2416/repositories/-IEEE-Access-01-26-Data-Augmentation/data/data/
├── PROVE/               # downstream_results_stage1.csv, downstream_results_cityscapes_gen.csv
├── leaderboard/         # strategy/per_dataset/per_domain/per_model breakdowns (S1 + CG)
├── ablation/            # ratio_ablation_full_results.csv, extended_training_analysis.csv, combination_results.csv
├── metadata/            # split_statistics.json, manifests
├── PRISM/               # generative_quality.csv
└── SWIFT/               # Domain distribution data
```

**When to copy:** After a stage reaches 100% coverage and results are verified, copy the corresponding CSVs:
```bash
# Copy downstream results
cp downstream_results.csv /home/mima2416/repositories/-IEEE-Access-01-26-Data-Augmentation/data/data/PROVE/downstream_results_stage1.csv
cp downstream_results_cityscapes_gen.csv /home/mima2416/repositories/-IEEE-Access-01-26-Data-Augmentation/data/data/PROVE/

# Copy leaderboard breakdowns
cp result_figures/leaderboard/breakdowns/*_stage1.csv /home/mima2416/repositories/-IEEE-Access-01-26-Data-Augmentation/data/data/leaderboard/
cp result_figures/leaderboard/breakdowns/*_cityscapes_gen.csv /home/mima2416/repositories/-IEEE-Access-01-26-Data-Augmentation/data/data/leaderboard/

# Copy ablation results
cp result_figures/ratio_ablation/ratio_ablation_full_results.csv /home/mima2416/repositories/-IEEE-Access-01-26-Data-Augmentation/data/data/ablation/
cp result_figures/extended_training/data/extended_training_analysis.csv /home/mima2416/repositories/-IEEE-Access-01-26-Data-Augmentation/data/data/ablation/
cp result_figures/combination_ablation/combination_results.csv /home/mima2416/repositories/-IEEE-Access-01-26-Data-Augmentation/data/data/ablation/
```

**Currently copied (2026-02-13):** S1 (420 tests), CG (250 tests), S1-Ratio (24/24), Extended S1 (20/20), Combination (17/18).
**Pending:** S2 (70% — wait for completion), Noise 50% + 100% (resubmitted — wait for completion), CS-Ratio (ready to copy — 48/48 + 96/96).
