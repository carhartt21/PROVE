# PROVE Project TODO

**Last Updated:** 2026-02-13 (07:00)

---

## 📊 Current Status (2026-02-13 07:00)

### Queue Summary
| User | Category | RUN | PEND | Total |
|------|----------|----:|-----:|------:|
| chge7185 | Noise ablation | 6 | 5 | 11 |
| chge7185 | Extended CG (**DUPLICATE** — kill these) | 0 | 10 | 10 |
| **chge7185 subtotal** | | **6** | **15** | **21** |
| mima2416 | S2 | 8 | 80 | 88 |
| mima2416 | Extended S1 | 4 | 0 | 4 |
| mima2416 | Extended CG | 10 | 0 | 10 |
| mima2416 | CS-Ratio | 2 | 0 | 2 |
| mima2416 | Combination | 1 | 0 | 1 |
| **mima2416 subtotal** | | **29** | **80** | **109** |

**Notes:**
- ⚠️ **DUPLICATE extended-cg jobs**: 10 identical extended-cg on chge7185 still pending. Kill with `bkill 3118898-3118907`.
- ✅ **S1 at ~93%**: 408/448 individual models complete (tracker), 420/420 tested. Remaining are mask2former on MapVistas/OUTSIDE15k.
- 🔄 **S2 at 62%**: 248 complete, 10 in-progress, 80 pending in queue. **10/11 selected strategies at 100%**, only baseline still at 33/40.
- ✅ **CS-Ratio at 96%**: 46/48 complete, 2 in-progress. Near completion.
- ✅ **Combination at 94%**: 17/18 complete + tested! Only `gen_Qwen_Image_Edit+std_cutmix/segformer` missing (1 RUN).
- 🔄 **Noise at 54%**: 13/24 complete + tested, 6 in-progress. **Early results: +5.02 pp avg over baseline!**
- 🔄 **Extended S1**: 13/20 complete at 45k (tested), 4 still running. 7 remaining pspnet jobs in progress.
- 🔄 **Extended CG**: 5/10 reached 25k (no final 60k yet), 10 RUN on mima2416.

---

### Training Progress
| Stage | Complete (models) | In-Progress | Pending | Coverage |
|-------|-------------------|-------------|---------|----------|
| Stage 1 (15k) | **408/448** | 6 | 34 | **91.1%** |
| Stage 2 (15k) | **248/400** | 10 | 142 | 🔄 **62.0%** |
| CG total (20k) | **125/100+** | 0 | 0 | **100%** ✅ |
| CS-Ratio Ablation (20k) | **46/48** | 2 | 0 | 🔄 **95.8%** |
| S1-Ratio Ablation (15k) | **24/24** | 0 | 0 | **100%** ✅ |
| Combination Ablation (20k) | **17/18** | 1 | 0 | 🔄 **94.4%** |
| Noise Ablation (15k) | **13/24** | 6 | 5 | 🔄 **54.2%** |
| Extended S1 (45k) | **13/20** | 7 | 0 | 🔄 **65.0%** |
| Extended CG (60k) | **0/10** | 10 | 0 | 🔄 **running** |

### Testing Progress
| Stage | Valid Tests | Trained | Notes |
|-------|------------|---------|-------|
| Stage 1 | **420** | 408 | **100% coverage** ✅ |
| Stage 2 | **247** | 248 | Auto-test on completion, 2 missing |
| CG | **250** | 125 | **100%** ✅ (Cityscapes + ACDC per model) |
| CS-Ratio | **92** | 46 | **100%** ✅ (Cityscapes + ACDC per model) |
| S1-Ratio | **24** | 24 | **100%** ✅ |
| Combination | **17** | 17 | **100%** ✅ |
| Noise | **13** | 13 | **100%** of completed |
| Extended S1 | **13** | 13 | **100%** of completed |

### 100% Coverage Plan

#### S1 Path to 100% — IN PROGRESS 🔄
| Step | Items | Status | Notes |
|------|-------|--------|-------|
| 1. S1 testing of trained models | **420/420** | ✅ **100%** | 0 missing |
| 2. Remaining S1 training | ~31 configs | 🔄 mask2former (MapVistas+OUTSIDE15k) | S1 at 417/448 |
| 3. Submit tests for new completions | Auto | ⏳ After step 2 | `auto_submit_tests.py --stage 1` |
| 4. Legacy 80k models | 3 configs | ✅ **COMPLETE** | All reached iter_80000 + tested (43.4-43.8%) |

### Strategy Leaderboard Highlights (2026-02-13 07:00)
| Stage | Top Strategy | mIoU | Baseline mIoU | Strategies > Baseline | Results |
|-------|-------------|------|---------------|----------------------|--------|
| Stage 1 | gen_automold | **40.45%** | 37.61% | **25/25 (all!)** | 417 |
| Stage 1 #2 | gen_UniControl | 40.37% | 37.61% | — | 417 |
| Stage 1 #3 | gen_albumentations_weather | 40.35% | 37.61% | — | 417 |
| Stage 2 | gen_IP2P | **41.98%** | 40.80% | 14/21 | 252 |
| Stage 2 #2 | gen_VisualCloze | 41.90% | 40.80% | — | 252 |
| Stage 2 #3 | gen_albumentations_weather | 41.87% | 40.80% | — | 252 |
| CG overall | gen_Img2Img | **52.87%** | 52.65% | 4/24 | 250 |
| CG #2 | gen_augmenters | 52.82% | 52.65% | — | 250 |
| CG #3 | gen_Qwen_Image_Edit | 52.67% | 52.65% | — | 250 |

**Key findings:**
- S1: **All 25** augmentation strategies beat baseline (+1.42 to +2.84 pp). **gen_automold now #1** (was gen_UniControl). Top-5: gen_automold, gen_UniControl, gen_albumentations_weather, gen_augmenters, gen_Qwen_Image_Edit
- S2: **Leaderboard shift!** gen_IP2P now #1 (+1.18 pp, 4 models), gen_VisualCloze #2, gen_albumentations_weather #3. ⚠️ Top-9 strategies all have only 4 test results (1 model) — rankings will change as more S2 completes. 14/21 beat baseline.
- CG: gen_Img2Img leads (+0.22 pp overall). Only 4/24 beat baseline — CG effect sizes much smaller than S1/S2
- **S2 caveat:** Many gen_* with only 4 results (single model tested) risk being biased. The 8 strategies with 16 results are more reliable: gen_step1x_new (+0.31), gen_flux_kontext (+0.31), gen_UniControl (+0.24), gen_Qwen_Image_Edit (+0.17) beat baseline; gen_Img2Img (-0.03), gen_CUT (-0.12), gen_augmenters (-0.36), gen_cycleGAN (-0.46) below.
- Full leaderboards: `result_figures/leaderboard/`

---

## 🔄 Active / In-Progress Tasks

### ✅ COMPLETE: 3 Legacy 80k Models (2026-02-12)

All 3 S1 segnext_mscan-b/bdd10k models completed training to 80k iterations and auto-tested:
- `gen_albumentations_weather`: mIoU=**43.61%** (baseline 80k: 41.27%)
- `gen_automold`: mIoU=**43.42%**
- `gen_step1x_v1p2`: mIoU=**43.77%**

**Config comparison (legacy 80k vs current 15k):**
### mask2former on MapillaryVistas/OUTSIDE15k (§2)

50 S1 configs — ALL mask2former_swin-b on MapillaryVistas (25) + OUTSIDE15k (25).
**Status:** 6 still running on chge7185 (80GB GPUs). Remaining pending in queue.
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

### Noise Ablation Study (§6)

**Status:** 🔄 13/24 complete + tested, 6 in-progress, 5 pending.

24 jobs: `gen_random_noise` strategy × 4 datasets × 6 models at 15k iters with `clear_day` domain filter.
Baselines skipped (identical to existing S1 baselines — same config).

**Implementation:** Replaces generated images with uniform random noise (`np.random.randint(0, 256, shape)`), preserving label maps. Uses cycleGAN reference manifest for image shapes. Tests whether augmentation gains come from meaningful content or just more training samples.

#### ⚠️ Preliminary Results (13/24, 2026-02-13)
**Surprising finding: random noise SIGNIFICANTLY helps!**

| Dataset/Model | Baseline (15k) | Noise (15k) | Diff |
|---|---:|---:|---:|
| bdd10k/mask2former_swin-b | 50.18 | 51.93 | **+1.75** |
| bdd10k/pspnet_r50 | 30.03 | 41.32 | **+11.29** |
| bdd10k/segformer_mit-b3 | 46.25 | 46.97 | **+0.71** |
| bdd10k/segnext_mscan-b | 41.27 | 48.43 | **+7.15** |
| iddaw/hrnet_hr48 | 20.69 | 33.99 | **+13.30** |
| iddaw/pspnet_r50 | 33.25 | 33.19 | -0.06 |
| iddaw/segformer_mit-b3 | 34.02 | 39.07 | **+5.05** |
| iddaw/segnext_mscan-b | 35.10 | 39.01 | **+3.92** |
| mapvistas/segformer_mit-b3 | 27.70 | 34.82 | **+7.12** |
| mapvistas/segnext_mscan-b | 34.64 | 34.64 | 0.00 |
| **Average** | | | **+5.02 pp** |

**Noise > baseline in 9/10 completed models.** Average gain = **+5.02 pp**.

**Interpretation:** The large gains (especially pspnet +11.29, hrnet +13.30) suggest the baselines for some models were undertrained or overfitting at 15k with limited data. Adding ANY extra images (even noise) acts as regularization. This finding is critical context for interpreting gen_* gains — part of the improvement may be from data volume/regularization rather than meaningful weather-domain content. Remaining 11 results (MapVistas + OUTSIDE15k) needed to confirm.

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

#### Cityscapes Ratio Ablation (48 jobs — 79% complete)
- **Status:** 🔄 38/48 complete, 10 RUN. All submitted.
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
| Cityscapes-ratio | 1.4 pp (ACDC) | 48 | 🔄 96% complete (46/48), 2 running |
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

**Status:** 🔄 Running (2026-02-13). S1: 13/20 complete at 45k, 7 in-progress. CG: 0/10 complete (all at 25k so far).
⚠️ **10 DUPLICATE extended-cg jobs on chge7185** — kill with `bkill 3118898-3118907`.

#### Preliminary Results — Extended S1 (13/20, 2026-02-13)
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
| gen_augmenters/iddaw/segformer | 38.95 | 39.07 | +0.12 |
| gen_cycleGAN/bdd10k/segformer | 39.95 | 46.62 | **+6.66** |
| gen_cycleGAN/iddaw/segformer | 38.91 | 39.10 | +0.18 |

**Interpretation (tentative):**
- **Most augmented models gain <0.3 pp** from 3× training → augmentation already converged at 15k
- **Two outliers:** baseline/iddaw/segformer (+5.16) and gen_cycleGAN/bdd10k/segformer (+6.66) were likely undertrained at 15k
- **Key question:** Does the baseline gap vs augmented models close with extended training? Baseline/bdd10k/pspnet went from 30.03→31.17 (+1.14) while gen_Img2Img went 40.98→41.19 (+0.21) — **augmentation gap persists** (10.02 pp at 45k vs 10.95 pp at 15k)

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

#### Status: 🔄 Running (2026-02-13). S1: 13/20 complete (45k), 7 in-progress. CG: 0/10 at final 60k (5/10 at 25k checkpoint), 10 running.

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

### 2026-02-13
- ✅ **S1-Ratio ablation 100% COMPLETE** — 24/24 trained + tested. Ratio effect weak (~0.4 pp spread). No finer ratios needed.
- ✅ **CS-Ratio preliminary analysis** — 38/48 tested. 1.4 pp ACDC spread, uniform peak distribution. Finer ratios not warranted.
- ✅ **CS-Ratio near completion** — 46/48 trained + tested (96%). Only 2 remaining.
- ✅ **Combination ablation near completion** — 17/18 trained + tested (94%). Only 1 remaining (gen_Qwen_Image_Edit+std_cutmix/segformer).
- ✅ **Noise ablation 54% complete** — 13/24 trained + tested. **Surprising: noise > baseline by +5.02 pp avg (9/10 positive!).**
- ✅ **Extended S1 65% complete** — 13/20 at 45k (tested). Most augmented models gain <0.3pp from 3× training. Augmentation gap persists.
- ✅ **Extended CG running** — 5/10 at 25k checkpoint, all 10 running.
- ✅ **Noise ablation submitted** — 24 gen_random_noise jobs on chge7185.
- ✅ **Extended training submitted** — 20 extended-s1 + 10 extended-cg on mima2416.
- ✅ **S2 major batch submitted** — 89 S2 jobs queued on mima2416.
- ✅ **Leaderboard updated** — S1: gen_automold now #1 (was gen_UniControl). S2: gen_IP2P now #1 (was gen_Attribute_Hallucination, but only 4-result strategies).
- ✅ **Extended training submitted** — 20 extended-s1 + 10 extended-cg on mima2416.
- ✅ **CS-Ratio remaining submitted** — 11 jobs (3118689-3118699), now 38/48 complete.
- ⚠️ **Duplicate extended-cg detected** — 10 identical jobs on chge7185 (3118898-3118907). Kill these.

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

**Models (6):** deeplabv3plus_r50, pspnet_r50, segformer_mit-b3, segnext_mscan-b, hrnet_hr48, mask2former_swin-b

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
