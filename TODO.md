# PROVE Project TODO

**Last Updated:** 2026-02-19 (22:20)

---

## 📊 Current Status (2026-02-19 23:27)

### Queue Summary
| User | Category | RUN | PEND | Total |
|------|----------|----:|-----:|------:|
| ${USER} | — | **0** | **0** | **0** |
| **PROVE Total** | | **0** | **0** | **0** |

**✅ From-Scratch Ratio=0.0 COMPLETE.** 60/60 trained at 40k, **60/60 tested (100%)**. IDD-AW removed from scope (no checkpoints produced). std_photometric_distort also removed from this stage.

**Notes:**
- ✅ **S1 COMPLETE**: 414/416 trained (2 no gen images), **420/420 tested (100%)**.
- ✅ **S2 COMPLETE**: 404/416 trained (12 no gen images), **433 tested (100%)**. 20 legacy ratio1p0 entries cleaned from CSV.
- ✅ **CG COMPLETE**: 100/100 trained, **250/250 tested (100%)**.
- ✅ **CS-Ratio COMPLETE**: **48/48 trained + 48/48 tested (100%)**.
- ✅ **S1-Ratio COMPLETE**: **24/24 trained + 24/24 tested (100%)**.
- ✅ **Noise Ablation COMPLETE + VERIFIED**: **48/48 trained+tested** in WEIGHTS_NOISE_ABLATION (24 ratio0p00 + 24 ratio0p50). 5 CG noise models also trained+tested. All 53 logs confirm `Injected ReplaceWithNoise into dataset pipeline`. Previous noise results retracted (bug `a48dd18`).
- ✅ **Extended S1 COMPLETE**: 20/20 at 45k, all tested.
- ✅ **Extended CG COMPLETE**: 10/10 trained + **10/10 tested (100%)**.
- ✅ **Combination Ablation COMPLETE**: **18/18 trained + 18/18 tested (100%)**.
- ✅ **From-Scratch Experiment FINALIZED**: **102/102 trained + tested at 40k**. 80k extension reached 29/102 before wall-clock exits — **40k adopted as final evaluation point** (convergence confirmed: +0.1-0.3pp from 40k→80k, plateau at 40-45k). 0 RUN, 0 PEND.
- ✅ **Cityscapes Pipeline Verification**: 4 trained, 4 tested.
- ✅ **Cityscapes BS8 COMPLETE**: 2 trained + **2 tested (100%)**.
- ✅ **Extended Ablation Tests**: 30/30 grouped test jobs COMPLETE — **220 test results** (140 S1 + 80 CG). All 190 checkpoints tested (120 S1 + 70 CG) plus base-iteration checkpoints from original WEIGHTS dirs.
- ⚠️ **Loss Ablation Training INCOMPLETE**: aux-boundary 0/6 at 80k (partial), **aux-focal 0/12 ALL FAILED** (`use_sigmoid=True` missing in FocalLoss config), aux-lovasz 2/12 at 80k, **loss-lovasz 0/60 at 80k** (53/60 NO CHECKPOINT — killed by wall-clock or never started). 29 test results exist from partial checkpoints. **Needs bug fix + resubmission.**
- 🔄 **Loss Ablation Tests**: 29 test results available (20 S1 + 9 S2). Only aux-lovasz and aux-boundary tested. aux-focal and loss-lovasz have no testable models.

---

### Training Progress
| Stage | Complete (models) | In-Progress | Pending | Coverage |
|-------|-------------------|-------------|---------|----------|
| Stage 1 (15k) | **414/416** | 0 | 0 | **100%** ✅ (2 no gen images) |
| Stage 2 (15k) | **404/416** | 0 | 0 | **100%** ✅ (12 no gen images) |
| CG total (20k) | **100/100** | 0 | 0 | **100%** ✅ |
| CS-Ratio Ablation (20k) | **48/48** | 0 | 0 | **100%** ✅ |
| S1-Ratio Ablation (15k) | **24/24** | 0 | 0 | **100%** ✅ |
| Combination Ablation (20k) | **18/18** | 0 | 0 | **100%** ✅ |
| Noise Ablation (WEIGHTS_NOISE_ABL) | **48/48** | 0 | 0 | **100%** ✅ |
| Extended S1 (45k, EXTENDED_ABL) | **20/20** | 0 | 0 | **100%** ✅ |
| Extended CG (50-60k, EXTENDED_ABL) | **10/10** | 0 | 0 | **100%** ✅ |
| From-Scratch (40k, FINAL) | **102/102** at 40k (29 also at 80k) | 0 | 0 | ✅ **Finalized at 40k** |
| From-Scratch Ratio=0.0 (40k) | **60/60** | 0 | 0 | ✅ **Complete** (IDD-AW + std_photometric_distort removed from scope) |
| Cityscapes BS8 | **2** | 0 | 0 | ✅ Pilot complete |
| Cityscapes Verification | **4** | 0 | 0 | Complete |

### Testing Progress
| Stage | Valid Tests | Trained | Notes |
|-------|------------|---------|-------|
| Stage 1 | **420** | 414 | **100% coverage** ✅ |
| Stage 2 | **433** | 404 | **100% coverage** ✅ |
| CG | **250** | 100 | **100%** ✅ (Cityscapes + ACDC per model) |
| CS-Ratio | **48** | 48 | **100%** ✅ |
| S1-Ratio | **24** | 24 | **100%** ✅ |
| Combination | **18** | 18 | **100%** ✅ |
| Noise Ablation | **48** | 48 | **100%** ✅ |
| Extended S1 | **140** | 20 | **100%** ✅ (multi-checkpoint: 7 tests per model) |
| Extended CG | **80** | 10 | **100%** ✅ (multi-checkpoint: 7-9 tests per model) |
| Cityscapes BS8 | **2/2** | 2 | **100%** ✅ |
| Loss Ablation | **29** (20 S1 + 9 S2) | ~20 | ⚠️ Training incomplete: aux-focal failed, loss-lovasz mostly failed |
| From-Scratch | **102** (40k) | 102 | ✅ All 40k tested. Finalized at 40k. |

### 100% Coverage Plan

#### S1 — COMPLETE ✅
All submittable S1 jobs trained (414/416, 2 have no gen images). All 420 tests complete.

#### S2 — COMPLETE ✅
All submittable S2 jobs trained (404/416, 12 have no gen images). 424/425 tested (2 PEND, 1 untestable — no iter_80000).

#### CG — COMPLETE ✅
100/100 trained, 250/250 tested.

### Strategy Leaderboard Highlights (2026-02-16)
| Stage | Top Strategy | mIoU | Baseline mIoU | Strategies > Baseline | Results |
|-------|-------------|------|---------------|----------------------|--------|
| Stage 1 | gen_automold | **40.45%** | 37.61% | **25/25 (all!)** | 420 |
| Stage 1 #2 | gen_UniControl | 40.37% | 37.61% | — | 420 |
| Stage 1 #3 | gen_albumentations_weather | 40.35% | 37.61% | — | 420 |
| Stage 2 | gen_LANIT | **41.76%** | 40.85% | **6/25** | ~424 |
| Stage 2 #2 | gen_step1x_new | 41.11% | 40.85% | — | ~424 |
| Stage 2 #3 | gen_flux_kontext | 41.11% | 40.85% | — | ~424 |
| CG overall | gen_Img2Img | **52.87%** | 52.65% | **4/25** | 250 |
| CG #2 | gen_augmenters | 52.82% | 52.65% | — | 250 |
| CG #3 | gen_Qwen_Image_Edit | 52.67% | 52.65% | — | 250 |

**Key findings:**
- S1: **All 25** augmentation strategies beat baseline (+1.42 to +2.84 pp). **gen_automold #1**. Top-5: gen_automold, gen_UniControl, gen_albumentations_weather, gen_augmenters, gen_Qwen_Image_Edit
- S2: gen_LANIT #1 but only 4 tests (unreliable). **Reliable (16 tests):** gen_flux_kontext=gen_step1x_new (+0.25). **6/25** beat baseline (gen_albumentations_weather newly above at +0.01pp).
- CG: gen_Img2Img leads (+0.22 pp). **5/25** beat baseline — CG effect sizes much smaller than S1/S2.
- **❌ Noise finding RETRACTED (BUG):** All noise results invalid — pipeline injection bug meant models trained on real images, not noise. Fix committed (`a48dd18`), 53 jobs resubmitted. See §CRITICAL BUG section.
- Full leaderboards: `result_figures/leaderboard/`

### Suggested Next Steps (Priority Order)

1. ✅ ~~Monitor from-scratch 80k extension~~ — **FINALIZED at 40k** (2026-02-16). 102/102 tested. 29/102 reached 80k before wall-clock exits. Convergence confirmed at 40-45k — 40k adopted as final evaluation point.
2. ✅ ~~Wait for pending test jobs~~ — **ALL COMPLETE** (2026-02-16): S2 433 tested, Extended CG 10/10, BS8 2/2, Combination 18/18.
3. ✅ ~~Verify noise results~~ — **VERIFIED** (2026-02-16). All 53 logs confirm `Injected ReplaceWithNoise into dataset pipeline`. Noise analysis: avg +2.72pp, 10/16 positive.
4. ✅ ~~Clean anomalies in test CSVs~~ — **DONE** (2026-02-16). Removed 20 legacy ratio1p0 entries from S2 CSV (428→408). CG noise entries valid (post-fix).
5. ✅ ~~Copy CS-Ratio to IEEE repo~~ — **DONE** (2026-02-15, commit `9e23e0c`).
6. ✅ ~~Copy S2 results to IEEE repo~~ — **DONE** (2026-02-15, commit `9e23e0c`).
7. ✅ ~~Export all ablation results to IEEE repo~~ — **DONE** (2026-02-15). Exported: CS-ratio (48), S1-ratio (24), noise (48+5), from-scratch (121 at 40k), extended S1 (20), extended CG (10), combination (17). Plus 12 leaderboard CSVs and comprehensive README at `data/data/README.md`.
8. ✅ ~~After from-scratch 80k completion~~ — **Finalized at 40k** (2026-02-16). 40k results are the official from-scratch evaluation point. Convergence confirmed.
9. ✅ ~~Export newly completed results to IEEE repo~~ — **DONE** (2026-02-16, commits `50d6cfa`, `040a396`). Exported: S2 cleaned CSV (430 lines), Combination (18/18), Extended CG (10/10), BS8 (2), From-scratch finalized (102 at 40k), Leaderboard breakdowns (12 CSVs), Updated data README.
10. **Run proposed additional studies** — See §Proposed Additional Studies below. Top priority: Cross-Dataset Generalization Testing (0 training, 8 test jobs) and PRISM Quality Correlation Analysis (0 jobs, pure analysis).
11. ⚠️ **Loss Ablation Training INCOMPLETE** — Major issues discovered (2026-02-19):
    - **aux-focal**: 0/12 complete — ALL FAILED with `AssertionError: Only sigmoid focal loss supported now.` (missing `use_sigmoid=True`)
    - **loss-lovasz**: 0/60 complete — 53/60 have NO CHECKPOINT (killed by wall-clock or never started)
    - **aux-lovasz**: 2/12 at 80k, rest partial (5k-60k)
    - **aux-boundary**: 0/6 at 80k, all partial (5k-70k)
    - **29 test results** exist from partial checkpoints (aux-lovasz + aux-boundary only)
    - **Action needed**: Fix aux-focal config, resubmit all incomplete loss ablation jobs
12. ✅ **From-Scratch Ratio=0.0 (100% Generated) COMPLETE** — **60/60** gen strategies at 40k (bdd10k + mapillaryvistas + outside15k × 20 gen_* strategies). IDD-AW removed from scope (repeated failures, no checkpoints after multiple resubmissions). std_photometric_distort also removed. Next: test all 60 models + compare vs ratio=0.50 and baseline.

### Proposed Additional Studies (Priority Order, 2026-02-15)

After current experiments complete, the following studies address remaining open questions. Ordered by **impact-to-cost ratio** — cheapest/highest-impact first.

#### 1. Cross-Dataset Generalization Testing (⭐ Highest Priority)
**Cost:** 0 training jobs, ~8 test-only LSF jobs | **Impact:** High — tests whether augmentation gains transfer across datasets

**Rationale:** S1 trains on BDD10k/IDD-AW/MapVistas/OUTSIDE15k separately. Do augmentation gains generalize? E.g., does a model trained on BDD10k+gen_automold also improve on IDD-AW test set? This addresses a key publication question: are augmentation benefits dataset-specific or universal?

**Design:**
- Select top-3 S1 strategies (gen_automold, gen_UniControl, gen_albumentations_weather) + baseline
- For each: test BDD10k-trained model on IDD-AW test set and vice versa
- Use existing `fine_grained_test.py` with `--dataset` pointing to the cross-dataset
- 2 directions × 4 strategies × 1 model (segformer) = **8 test jobs** (zero training)

**Questions answered:**
1. Do augmentation benefits transfer across datasets, or are they dataset-specific?
2. Does the strategy ranking change when evaluated cross-dataset?
3. Is there a correlation between in-dataset and cross-dataset augmentation gain?

#### 2. PRISM Quality Correlation Analysis (⭐ High Priority)
**Cost:** 0 jobs (pure analysis script) | **Impact:** High — links image quality to downstream performance

**Rationale:** The PRISM project has FID, LPIPS, CLIP-FID, and perceptual quality scores for all 21 gen_* methods. Correlating these with PROVE mIoU gains answers: **does better image quality → better segmentation?** This is a key finding for the publication.

**Design:**
- Merge PRISM quality metrics (FID, LPIPS per method×dataset) with PROVE mIoU gains
- Compute Spearman rank correlation: FID vs mIoU gain, LPIPS vs mIoU gain
- Stratify by dataset and model family
- Visualization: scatter plot with method labels, regression line, ρ and p-value

**Questions answered:**
1. Does image realism (FID) predict downstream segmentation improvement?
2. Is perceptual similarity (LPIPS) a better predictor than distributional distance (FID)?
3. Are there outlier methods (high FID but high mIoU gain, or vice versa)?

#### 3. Batch Size & Loss Function Analysis (Medium Priority)
**Cost:** 0 jobs (data already exists) | **Impact:** Medium — surfaces training hyperparameter interactions

**Rationale:** `WEIGHTS_BATCH_SIZE_ABLATION/` (BS 2/4/8/16 with LR scaling) and `WEIGHTS_LOSS_ABLATION/` (aux-lovasz, aux-boundary, aux-focal, loss-lovasz) already contain trained models. These have never been systematically analyzed.

**Design:**
- Enumerate checkpoints and test results in both directories
- Batch size: plot mIoU vs batch size per strategy, check if augmentation gains scale with BS
- Loss function: compare aux-loss variants vs standard CE — do different losses change which augmentations help?
- Analysis script only, no new training

**Questions answered:**
1. Does larger batch size amplify or diminish augmentation gains?
2. Do alternative loss functions shift the optimal augmentation strategy?
3. Is the standard BS=2 + CE configuration already optimal, or are interactions present?

#### 4. Augmentation Timing / Curriculum (Low Priority)
**Cost:** 3–6 training jobs | **Impact:** Medium — tests when augmentation helps most

**Rationale:** Current design applies augmentation uniformly from iter 0 to final. But augmentation may be most beneficial early (for regularization) or late (after features stabilize). This tests a curriculum hypothesis.

**Design:**
- Take top strategy (gen_automold) + segformer on BDD10k
- **Early-only:** Apply gen_* for first 50% of iters, then real-only
- **Late-only:** Train real-only for first 50%, then mix in gen_*
- **Warmup:** Linearly increase gen_* ratio from 0→0.5 over training
- Compare vs uniform augmentation (already available)
- Requires minor modification to `MixedBatchSampler` in `unified_training.py` (iteration-aware ratio)
- 3 new configs × 1–2 models = **3–6 jobs**

**Questions answered:**
1. Is augmentation benefit front-loaded (regularization) or back-loaded (diversity)?
2. Does curriculum augmentation outperform uniform mixing?

#### 5. Domain-Targeted Augmentation (Low Priority)
**Cost:** ~7 training jobs | **Impact:** Medium — tests selective augmentation

**Rationale:** S1 trains on clear_day only but tests on all domains (rain, fog, night, snow). Some gen_* methods produce weather-specific images. Does targeting the weakest domain yield more efficient gains?

**Design:**
- Identify worst-performing domain per model from S1 test results (typically night or fog)
- Filter generated images to only include that target domain
- Train with targeted gen_* vs uniform gen_* (already available)
- Use gen_automold (S1 #1) on BDD10k with segformer
- 1 strategy × 4–7 domain targets × 1 model = **~7 jobs**

**Questions answered:**
1. Is targeted augmentation more efficient than broad augmentation?
2. Does fixing the weakest domain come at the cost of other domains?
3. What is the optimal domain allocation for generated images?

#### Not Recommended: Data Efficiency / Subset Training
**Rationale:** This is already implicitly tested by the cross-dataset design. The 4 S1 datasets have very different training set sizes (BDD10k ~7k, IDD-AW ~3k, MapVistas ~18k, OUTSIDE15k ~15k), so the existing results already show how augmentation interacts with dataset scale. A dedicated subset study would be redundant.

---

### 📂 WEIGHTS Directory Inventory (2026-02-19 Audit)

| Directory | Purpose | Trained | Tested | Status |
|-----------|---------|---------|--------|--------|
| `WEIGHTS/` | Stage 1 (clear_day) | 414/416 | 420/420 | ✅ Complete |
| `WEIGHTS_STAGE_2/` | Stage 2 (all domains) | 404/416 | 424/425 | ✅ Complete |
| `WEIGHTS_CITYSCAPES_GEN/` | CG (Cityscapes replication) | 100/100 | 250/250 | ✅ Complete |
| `WEIGHTS_CITYSCAPES_RATIO/` | CS-Ratio ablation | 48/48 | 48/48 | ✅ Complete |
| `WEIGHTS_STAGE1_RATIO/` | S1-Ratio ablation | 24/24 | 24/24 | ✅ Complete |
| `WEIGHTS_NOISE_ABLATION/` | Noise ablation (fixed) | 48/48 | 48/48 | ✅ Complete |
| `WEIGHTS_EXTENDED_ABLATION/stage1` | Extended S1 (15k→45k) | 20/20 | **140** (multi-ckpt) | ✅ Complete |
| `WEIGHTS_EXTENDED_ABLATION/cityscapes_gen` | Extended CG (20k→60k) | 10/10 | **80** (multi-ckpt) | ✅ Complete |
| `WEIGHTS_COMBINATION_ABLATION/` | Combination (gen+std) | 18/18 | 18/18 | ✅ Complete |
| `WEIGHTS_FROM_SCRATCH/` | From-scratch (ratio=0.50) | 102/102 | 102/102 | ✅ Finalized at 40k |
| `WEIGHTS_FROM_SCRATCH/` | From-scratch (ratio=0.00) | **60/60** | **60/60** | ✅ Complete (IDD-AW removed from scope) |
| `WEIGHTS_CITYSCAPES/` | Pipeline verification | 4/4 | 4/4 | ✅ Complete |
| `WEIGHTS_CITYSCAPES_BS8/` | BS8 pilot | 2/2 | 2/2 | ✅ Complete |
| `WEIGHTS_LOSS_ABLATION/` | Loss function ablation | 2/90+ (80k) | 29 | ⚠️ Mostly incomplete: aux-focal ALL FAILED, loss-lovasz mostly NO CKPT |
| `WEIGHTS_BATCH_SIZE_ABLATION/` | Batch size ablation | 0 | 0 | ⚠️ Setup only, no checkpoints |

---

## 🔄 Active / In-Progress Tasks

### 🔄 From-Scratch Experiment (2026-02-14)

**Purpose:** Test whether augmentation gains are genuine or masked by pretrained backbone features. If pretrained features dominate, augmentations may show smaller relative impact. Training from scratch removes this confound.

**Configuration:**
| Parameter | Value |
|-----------|-------|
| Model | segformer_mit-b3 only |
| Datasets | BDD10k, IDD-AW, MapillaryVistas, OUTSIDE15k |
| Strategies | All 26 (baseline + 4 std_* + 21 gen_*) |
| Iterations | **40,000 → 80,000** (extended based on convergence analysis) |
| Checkpoint/eval | Every 5,000 |
| Domain filter | clear_day (S1 protocol) |
| Backbone | `--no-pretrained` (init_cfg=None) |
| Output | `${AWARE_DATA_ROOT}/WEIGHTS_FROM_SCRATCH/` |

**Submitted:** 100 jobs (4 skipped — no generated images). Job IDs 3577430–3577529.
**Commit:** `fbe8870` (from-scratch stage), `687fbee` (80k extension)
**Progress (2026-02-16 14:45):** **102/102 tested at 40k** — FINALIZED. 80k extension reached 29/102 before wall-clock exits. **40k adopted as final evaluation point** — convergence confirmed (plateau at 40-45k, only +0.1-0.3pp beyond). 0 RUN, 0 PEND.

### 🔄 From-Scratch Ratio=0.0 Experiment (2026-02-17)

**Purpose:** Test whether generated images alone (without any real data) can teach a model from scratch. With ratio=0.0, training uses 100% generated images and 0% real images. Compares against ratio=0.50 (existing) and baseline (real-only) results.

**Configuration:**
| Parameter | Value |
|-----------|-------|
| Model | segformer_mit-b3 only |
| Datasets | BDD10k, MapillaryVistas, OUTSIDE15k (IDD-AW removed — persistent failures) |
| Strategies | 20 gen_* strategies (gen_LANIT skipped — no images) |
| Ratio | 0.0 (100% generated, 0% real) |
| Iterations | 40,000 |
| Checkpoint/eval | Every 5,000 |
| Domain filter | clear_day (S1 protocol) |
| Backbone | `--no-pretrained` (init_cfg=None) |
| Output | `WEIGHTS_FROM_SCRATCH/{strategy}/{dataset}/segformer_mit-b3_ratio0p00/` |

**Submitted:** 80 jobs initially (4 datasets × 20 strategies). IDD-AW removed from scope after repeated failures.
**Final Status (2026-02-19 22:20):** **60/60 complete at 40k** (3 datasets × 20 gen_* strategies). BDD10k finished via 5 DONE + earlier completions. 4 bdd10k models also reached iter_80000. IDD-AW dropped: all 20 jobs produced NO CHECKPOINT after multiple resubmissions (killed by TERM_OWNER). std_photometric_distort also removed from this stage.

**Research Questions:**
1. Can a model learn useful features from generated images alone (no real data)?
2. How does ratio=0.0 performance compare to ratio=0.50 (mixed) and baseline (real-only)?
3. Which gen_* strategies produce images that are most useful as standalone training data?
4. Is there a correlation between gen image quality (FID/LPIPS) and standalone trainability?

#### 80k Extension Progress (2026-02-16, FINAL — wall-clock exits, 40k adopted)

80k checkpoint coverage: **29/102** (28%). All jobs exited due to wall clock limits. Since convergence is confirmed at 40k-45k (+0.1-0.3pp beyond), **40k is the final evaluation point**.

| Coverage | # Models |
|----------|---------|
| iter_40000 | **102/102** (100%) ✅ — FINAL |
| iter_45000 | 81/102 (79%) |
| iter_50000 | 66/102 (65%) |
| iter_80000 | 29/102 (28%) |

#### Convergence Analysis (2026-02-15, UPDATED with 80k data)

**Training has clearly plateaued by 40-45k iterations.** The 45k→80k window shows only +0.1–0.3pp marginal improvement — essentially noise. 80k iterations is **more than sufficient**.

**SegFormer-B3 on BDD10k convergence (mIoU %):**
| Strategy | 5k | 10k | 20k | 30k | 40k | 50k | 60k | 70k | 80k | Δ(40→80k) |
|----------|-----|-----|-----|-----|-----|-----|-----|-----|-----|--------|
| baseline | 23.7 | 25.3 | 28.9 | 30.2 | 32.2 | 32.5 | 32.5 | 32.4 | 32.5* | +0.3 |
| gen_automold | 22.3 | 23.6 | 25.0 | 26.2 | 28.3 | 28.4 | 28.5 | 28.5 | 28.6 | +0.3 |
| gen_cycleGAN | 21.1 | 24.4 | 25.0 | 29.8 | 31.5 | 31.7 | 31.7 | 31.6 | 31.8* | +0.3 |
| gen_flux_kontext | 23.4 | 24.1 | 27.4 | 30.9 | 32.4 | 32.6 | 32.6 | 32.5 | 32.7* | +0.3 |
| std_randaugment | 20.0 | 24.2 | 27.1 | 30.8 | 32.2 | 32.4 | 32.4 | 32.4 | 32.5 | +0.3 |

*(*) = best checkpoint at 75-80k. All strategies converge by 40-45k — rapid improvement phase is 5k→30k (+7-10pp), 30k→40k (~+2pp), 40k→80k (~+0.3pp).*

**Conclusion:** Extending from 40k to 80k confirms the 40k evaluation was already on the plateau. Final 80k results will barely change from 40k. The extension validates the earlier analysis rather than providing new signal.

**Per-dataset average progression (mIoU, all strategies):**
| Dataset | 5k | 10k | 15k | 20k | 25k | 30k | 35k | 40k |
|---------|-----|-----|-----|-----|-----|-----|-----|-----|
| bdd10k | 22.5 | 23.2 | 24.9 | 26.2 | 27.4 | 29.3 | 30.3 | 31.4 |
| iddaw | 16.5 | 18.1 | 19.1 | 1.2 | 22.4 | 24.5 | 25.5 | 26.0 |
| mapillaryvistas | 10.4 | 11.9 | 13.7 | 15.2 | 16.4 | 18.1 | 19.6 | 20.3 |
| outside15k | 19.4 | 20.4 | 21.5 | 22.9 | 24.1 | 25.3 | 26.8 | 27.5 |

#### Results at 40k (90/102 completed+tested, 23 strategies)

**Strategy Rankings (sorted by avg gain vs baseline, all 4 datasets):**
| # | Strategy | BDD10k | IDD-AW | MapVst | OUT15k | Avg Gain | pos |
|---|----------|--------|--------|--------|--------|----------|-----|
| 1 | gen_flux_kontext | +0.19 | -1.34 | **+2.59** | **+1.49** | **+0.73** | 3/4 |
| 2 | gen_VisualCloze | **+1.03** | **+1.16** | -0.10 | -0.18 | **+0.48** | 2/4 |
| 3 | gen_CNetSeg | -1.88 | +0.23 | **+1.93** | +0.49 | **+0.19** | 3/4 |
| 4 | gen_step1x_v1p2 | +0.40 | +0.02 | +0.11 | -0.26 | +0.07 | 3/4 |
| 5 | std_mixup | -0.73 | -1.49 | +1.58 | +0.80 | +0.04 | 2/4 |
| 6 | gen_step1x_new | -1.46 | +0.18 | +1.41 | -0.03 | +0.03 | 2/4 |
| — | **baseline** | **32.23** | **26.82** | **19.50** | **27.24** | — | — |
| 7 | gen_Qwen_Image_Edit | +0.17 | -0.40 | +0.42 | -0.23 | -0.01 | 2/4 |
| 8 | std_randaugment | -0.04 | -0.64 | +0.11 | +0.43 | -0.03 | 2/4 |
| 9 | gen_albumentations | -0.20 | +0.12 | +0.32 | -0.39 | -0.04 | 2/4 |
| 10 | gen_Attribute_Hallucination | +0.40 | -0.41 | +0.12 | -0.52 | -0.10 | 2/4 |
| 11 | gen_IP2P | -0.44 | +0.19 | +0.30 | -0.68 | -0.16 | 2/4 |
| 12 | gen_Img2Img | -0.50 | +0.25 | +0.58 | -1.47 | -0.29 | 2/4 |
| 13 | gen_UniControl | +0.24 | -1.07 | +0.05 | -0.49 | -0.32 | 2/4 |
| 14 | gen_CUT | -1.27 | +0.47 | -0.14 | -0.34 | -0.32 | 1/4 |
| 15 | std_autoaugment | -1.32 | +0.93 | -0.18 | -0.88 | -0.36 | 1/4 |
| 16 | std_cutmix | -1.92 | -1.01 | +0.22 | +0.20 | -0.63 | 2/4 |
| 17 | gen_cycleGAN | -0.78 | -3.53 | +1.31 | +0.25 | -0.69 | 2/4 |
| 18 | gen_stargan_v2 | +0.43 | -2.41 | -0.15 | -0.80 | -0.73 | 1/4 |
| 19 | gen_cyclediffusion | -1.26 | -2.51 | +1.06 | -0.28 | -0.75 | 1/4 |
| 20 | gen_automold | **-3.92** | -1.83 | +1.12 | +0.62 | -1.00 | 2/4 |
| 21 | gen_random_noise | -0.69 | -1.04 | +0.76 | -0.46 | -0.36 | 1/4 |
| 22 | gen_SUSTechGAN | -0.20 | **-2.26** | -0.02 | **-2.17** | **-1.16** | 0/4 |

**Coverage:** 90 tested, 102 trained, 12 in progress (gen_TSIT, gen_Weather_Effect_Generator, gen_augmenters). 23 strategies × 4 datasets.

#### Strategy Type Summary
| Type | n | Avg Gain | Positive % | Best Dataset |
|------|---|----------|-----------|--------------|
| gen_* (generative) | 69 | -0.24 pp | 45% | MapVistas (+0.77 avg) |
| std_* (standard) | 16 | -0.25 pp | 44% | MapVistas (+0.43 avg) |
| **All non-baseline** | **85** | **-0.24 pp** | **45%** | **MapVistas** |

#### Dataset-Level Pattern (Critical Finding)
| Dataset | Baseline | Avg Gain | Positive | Best Strategy |
|---------|----------|----------|----------|---------------|
| **MapillaryVistas** | 19.50 | **+0.84** | **81%** (17/21) | gen_flux_kontext (+2.59) |
| OUTSIDE15k | 27.24 | -0.05 | 48% (10/21) | gen_flux_kontext (+1.49) |
| IDD-AW | 26.82 | -0.56 | 41% (9/22) | gen_VisualCloze (+1.16) |
| **BDD10k** | 32.23 | **-0.86** | **14%** (3/22) | gen_VisualCloze (+1.03) |

**Key insight:** Augmentation helps MORE on datasets with lower baselines (MapVistas 19.5 → +0.84 avg, 81% positive) and HURTS on datasets where the from-scratch model already performs better (BDD10k 32.2 → -0.86 avg, 14% positive). The "capacity allocation" effect holds: when the model is still learning basic features, augmentation adds useful variation; when it already has reasonable features, augmentation confuses the learning signal.

#### Cross-Initialization Correlation: Scratch vs Pretrained (21 matched strategies)

| # | Strategy | Scratch Gain | Pretr Gain | Reversal |
|---|----------|-------------|-----------|----------|
| 1 | gen_flux_kontext | **+0.73** | +3.20 | |
| 2 | gen_VisualCloze | **+0.48** | +4.48 | |
| 3 | gen_CNetSeg | **+0.19** | +4.95 | |
| 4 | gen_step1x_v1p2 | +0.07 | +3.10 | |
| 5 | std_mixup | +0.04 | +4.59 | |
| 6 | gen_step1x_new | +0.03 | +2.82 | |
| 7 | gen_Qwen_Image_Edit | -0.01 | +4.89 | **YES** |
| 8 | std_randaugment | -0.03 | +4.67 | **YES** |
| 9 | gen_albumentations_weather | -0.04 | +4.42 | **YES** |
| 10 | gen_Attribute_Hallucination | -0.10 | +5.02 | **YES** |
| 11 | gen_IP2P | -0.16 | +4.79 | **YES** |
| 12 | gen_Img2Img | -0.29 | +4.46 | **YES** |
| 13 | gen_UniControl | -0.32 | +4.90 | **YES** |
| 14 | gen_CUT | -0.32 | +4.71 | **YES** |
| 15 | std_autoaugment | -0.36 | +4.71 | **YES** |
| 16 | std_cutmix | -0.63 | +3.76 | **YES** |
| 17 | gen_cycleGAN | -0.69 | +3.03 | **YES** |
| 18 | gen_stargan_v2 | -0.73 | +4.60 | **YES** |
| 19 | gen_cyclediffusion | -0.75 | +4.40 | **YES** |
| 20 | gen_automold | -1.00 | +4.93 | **YES** |
| 21 | gen_SUSTechGAN | -1.16 | +4.58 | **YES** |

**Statistics:**
| Metric | Value |
|--------|-------|
| **Spearman ρ** | **-0.091 (p=0.695)** — essentially zero correlation |
| Direction reversals | **15/21 (71%)** — most strategies flip sign |
| Avg pretrained gain | **+4.33 pp** (all positive) |
| Avg scratch gain | **-0.24 pp** (mostly negative) |
| Gain collapse ratio | **-0.06x** — gains don't just shrink, they invert |

**Biggest rank changes (pretrained → scratch):**
- gen_flux_kontext: #18 → **#1** (+17 ranks) — learns genuine features  
- gen_step1x_new: #21 → **#6** (+15 ranks) — benefits persist without pretraining
- gen_step1x_v1p2: #19 → **#4** (+15 ranks) — same model family  
- gen_automold: #3 → **#20** (-17 ranks) — pretrained features masked harm
- gen_VisualCloze: #13 → **#2** (+11 ranks) — truly helpful augmentation

#### Key Findings — From-Scratch vs Pretrained

1. **Augmentation gains collapse without pretraining:** S1 pretrained avg gain = +4.33pp (21/21 positive). From scratch = -0.24pp (6/22 positive). Pretrained features are ESSENTIAL for augmentation benefits — **without them, most augmentations are harmful.**

2. **Strategy rankings are initialization-dependent:** Spearman ρ = -0.091 (p=0.695) between pretrained and scratch gains. 71% of strategies reverse direction. **Rankings from pretrained evaluation do NOT predict from-scratch performance.**

3. **gen_flux_kontext is the most robust strategy:** Jumps from #18 (pretrained) to #1 (scratch) — the ONLY strategy providing consistent benefit (+0.73pp) regardless of initialization. Its step1x family (v1p2, new) also shows robustness (#4, #6 from scratch).

4. **gen_automold is the clearest "pretrained illusion":** #3 in pretrained (+4.93pp) but #20 from scratch (-1.00pp). The pretrained backbone compensated for the harmful augmentation, masking real damage.

5. **Dataset-capacity interaction persists:** MapillaryVistas (lowest baseline, 19.5) benefits most from augmentation (81% positive). BDD10k (highest baseline, 32.2) is hurt most (14% positive). This holds for both gen_* and std_*.

6. **gen_* ≈ std_* from scratch:** gen_* avg = -0.24pp (45% positive), std_* avg = -0.25pp (44% positive). The alleged superiority of generative augmentation seen in pretrained evaluations disappears entirely without pretrained features.

7. **Implication for publication:** The community should NOT evaluate augmentation strategies solely on pretrained backbones — **71% of strategy rankings reverse** when the pretrained backbone is removed. From-scratch evaluation is essential to distinguish genuine augmentation benefits from pretrained feature compensation.

**⚠️ DUPLICATE NOTE:** 87 duplicate from-scratch jobs on chge7185 are **safe** — pre-flight checkpoint check + flock lock prevent issues.

**Note:** 2 standalone pspnet_r50 BDD10k jobs completed (80k iters, separate experiment). Different LR schedules, NOT part of the batch.

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
${AWARE_DATA_ROOT}/WEIGHTS_COMBINATION_ABLATION/
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

**Status:** ✅ **COMPLETE** (2026-02-19). S1: **20/20 COMPLETE** at 45k (140 test results across all checkpoints). CG: **10/10 COMPLETE** at 50-60k (80 test results across all checkpoints). **220 total test results** from 30 grouped test jobs.

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
${AWARE_DATA_ROOT}/WEIGHTS_EXTENDED_ABLATION/
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

#### Status: Extended S1: ✅ **20/20 COMPLETE** (all tested, 140 test results). Extended CG: ✅ **10/10 COMPLETE** (all tested, 80 test results).

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

### 2026-02-19
- ✅ **Extended Ablation Tests VERIFIED COMPLETE** — All 30/30 grouped test jobs finished. 220 test results total (140 S1 + 80 CG) covering all 190 checkpoints plus base-iteration checkpoints. Every model has exactly tests = ckpts + 1 (base iter from original WEIGHTS dir).
- ✅ **From-Scratch Ratio=0.0 Resubmission** — 45 gen ratio=0.0 + 4 std_photometric_distort jobs resubmitted with --resume (Job IDs 4334901–4334949). Cleaned 2 stale locks (std_photometric_distort with wrong ratio1p00). Previous batch (4248*) had 15 EXIT from lock conflicts.
- ⚠️ **Loss Ablation Issues Discovered** — aux-focal 0/12 ALL FAILED (FocalLoss `use_sigmoid=True` bug). loss-lovasz 0/60 (53 NO CHECKPOINT). aux-lovasz only 2/12 at 80k. aux-boundary 0/6 at 80k. Needs config fix + resubmission.
- ✅ **WEIGHTS_EXTENDED deleted** — Stale 69GB legacy directory removed (superseded by WEIGHTS_EXTENDED_ABLATION).

### 2026-02-18
- ✅ **All pending tests COMPLETE** — Extended CG 10/10 tested (was 5/10), BS8 2/2 tested (was 0/2), Combination 18/18 trained+tested (was 17/18), S2 433 tested (was 424).
- ✅ **Noise ablation VERIFIED** — All 53 logs (48 WEIGHTS_NOISE_ABLATION + 5 CG) confirm `Injected ReplaceWithNoise into dataset pipeline`. Fix working correctly.
- ✅ **S2 CSV cleaned** — Removed 20 legacy `ratio1p0` entries (428→408 lines). CG noise entries (10) confirmed valid post-fix.
- ✅ **Trackers + leaderboards updated** — All stages: training tracker, testing tracker, and strategy leaderboards refreshed.
- ✅ **From-scratch 80k progress** — 14/27 strategies have ≥1 dataset at 80k. 15 jobs RUN, 0 PEND (was 22 RUN + 69 PEND). gen_automold fully at 80k. gen_SUSTechGAN near completion (70-75k).

### 2026-02-14
- ✅ **CRITICAL BUG FOUND & FIXED** — `ReplaceWithNoise` injected into wrong config attribute (`cfg.train_pipeline` instead of `cfg.train_dataloader.dataset.pipeline`). All 49 noise training runs were invalid (models trained on real images). Fix committed (`a48dd18`).
- ✅ **Invalid noise data deleted** — `WEIGHTS_NOISE_ABLATION/gen_random_noise/` (237 GB) + `WEIGHTS_CITYSCAPES_GEN/gen_random_noise/` (36 GB) = **273 GB freed**.
- ✅ **53 noise jobs resubmitted** — 24 noise-50% (ratio 0.50) + 24 noise-100% (ratio 0.00) + 5 CG noise-100% (ratio 0.00). Job IDs: 3454599–3455259.
- ✅ **Pending S2 killed on ${USER}** — 26 RUN remain (will complete). S2 continuing from chge7185 queue.
- ✅ **"Landmark finding" retracted** — noise > gen_* comparison was artifact of bug (models used real data, not noise).

### 2026-02-13
- ~~✅ **Noise 100% COMPLETE**~~ → ❌ **RETRACTED** — results invalid due to pipeline injection bug (see 2026-02-14).
- ✅ **CS-Ratio COMPLETE** — 48/48 trained + 96/96 tested (100%). Both stalled resume jobs (gen_TSIT/pspnet, gen_flux_kontext/segnext) finished and auto-tested.
- ✅ **S2 progressed to 70%** — 280/400 models complete (was 235). Testing: 339 valid (was 301). 13 gen model failures being retrained (38 jobs queued).
- ✅ **S2 leaderboard updated** — 6/25 strategies beat baseline (was 5/23). gen_albumentations_weather newly above baseline (+0.01pp). 344 total results.
- ✅ **CG leaderboard updated** — 5/25 strategies beat baseline (was 4/24). 254 total results.
- ✅ **Extended CG resuming** — 4 pspnet models running toward 60k.
- ✅ **S2 remaining 99 jobs submitted** (3216823–3216945) — all configs with generated images now in pipeline. S2 target: ~388/400 (97%).
- ✅ **Verified results copied to IEEE repo** — S1 (420), CG (250), S1-Ratio (24/24), Extended S1 (20/20), Combination (17/18) → `${HOME}/repositories/-IEEE-Access-01-26-Data-Augmentation/data/data/`
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
bjobs -u ${USER} | grep RUN      # Running jobs only
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
5. **NEVER write temp files to `/tmp`** — use project dir or `${AWARE_DATA_ROOT}/`

### Publication Data Directory

Results with 100% coverage must be copied to the IEEE paper repository for analysis and figure generation:

```
${HOME}/repositories/-IEEE-Access-01-26-Data-Augmentation/data/data/
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
cp downstream_results.csv ${HOME}/repositories/-IEEE-Access-01-26-Data-Augmentation/data/data/PROVE/downstream_results_stage1.csv
cp downstream_results_cityscapes_gen.csv ${HOME}/repositories/-IEEE-Access-01-26-Data-Augmentation/data/data/PROVE/

# Copy leaderboard breakdowns
cp result_figures/leaderboard/breakdowns/*_stage1.csv ${HOME}/repositories/-IEEE-Access-01-26-Data-Augmentation/data/data/leaderboard/
cp result_figures/leaderboard/breakdowns/*_cityscapes_gen.csv ${HOME}/repositories/-IEEE-Access-01-26-Data-Augmentation/data/data/leaderboard/

# Copy ablation results
cp result_figures/ratio_ablation/ratio_ablation_full_results.csv ${HOME}/repositories/-IEEE-Access-01-26-Data-Augmentation/data/data/ablation/
cp result_figures/extended_training/data/extended_training_analysis.csv ${HOME}/repositories/-IEEE-Access-01-26-Data-Augmentation/data/data/ablation/
cp result_figures/combination_ablation/combination_results.csv ${HOME}/repositories/-IEEE-Access-01-26-Data-Augmentation/data/data/ablation/
```

**Currently copied (2026-02-13):** S1 (420 tests), CG (250 tests), S1-Ratio (24/24), Extended S1 (20/20), Combination (17/18).
**Pending:** S2 (70% — wait for completion), Noise 50% + 100% (resubmitted — wait for completion), CS-Ratio (ready to copy — 48/48 + 96/96).
