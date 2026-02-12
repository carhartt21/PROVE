# PROVE Project TODO

**Last Updated:** 2026-02-12 (21:20)

---

## 📊 Current Status (2026-02-12 21:20)

### Queue Summary
| User | Category | RUN | PEND | Total |
|------|----------|----:|-----:|------:|
| chge7185 | S1 mask2former (MapVistas+OUTSIDE15k, 80GB) | 6 | 6 | 12 |
| chge7185 | S2 curated strategies | 0 | 102 | 102 |
| **chge7185 subtotal** | | **6** | **108** | **114** |
| mima2416 | S2 training (gen_flux_kontext, std_* on mask2former) | 5 | ~64 | ~69 |
| mima2416 | CS-Ratio ablation | 2 | 0 | 2 |
| mima2416 | Combination ablation | 7 | 7 | 14 |
| **mima2416 subtotal** | | **14** | **~71** | **~85** |

**Notes:**
- ✅ **3 legacy 80k models COMPLETE** (2026-02-12): gen_albumentations_weather, gen_automold, gen_step1x_v1p2 (segnext/bdd10k) — all reached iter_80000 + tested (mIoU: 43.4-43.8%, baseline 80k: 41.27%).
- 🔄 **Combination ablation progressing** (2026-02-12 21:15): 4/18 complete (iter_20000), 7 RUN (iter_2000-8000), 7 PEND. Lock contention fix working — new autoaugment/mixup jobs successfully started.
- 🔄 **S2 training wave active**: gen_flux_kontext + std_autoaugment/cutmix on mask2former running. S2 coverage: 188/400 = 47.0%.
- 🔄 **S1 at 91.1%**: 408/448 complete. Remaining 40 are mask2former on MapVistas/OUTSIDE15k (chge7185, 80GB GPUs).
- ⚠️ **CS-Ratio: 9 jobs need submission** — 37/48 complete, 2 running, 9 not yet in queue. Submit with `--stage cityscapes-ratio -y`.
- ✅ **S1 testing: 100% coverage** (420 valid, 0 missing)
- ✅ **CG testing: 100% coverage** (250 valid, 0 missing)

---

### Training Progress
| Stage | Complete (models) | Running | Pending | Failed | Coverage |
|-------|-------------------|---------|---------|--------|----------|
| Stage 1 (15k) | **408/448** | 6 | 34 | 0 | **91.1%** |
| Stage 2 (15k) | **188/400** | 7 | 202 | 3 | **47.0%** |
| CG baseline+std (20k) | **20/20** | 0 | 0 | 0 | **100%** ✅ |
| CG gen_* (20k) | **80/80** | 0 | 0 | 0 | **100%** ✅ |
| CG total (20k) | **100/100** | 0 | 0 | 0 | **100%** ✅ |
| CS-Ratio Ablation (20k) | **37/48** | 2 | 9 (not submitted) | 0 | **77.1%** |
| Combination Ablation (20k) | **4/18** | 7 | 7 | 0 | 🔄 **22.2%** |

### Testing Progress
| Stage | Valid Tests | Missing | Notes |
|-------|------------|---------|-------|
| Stage 1 | **420** | **0** | **100% coverage** ✅ |
| Stage 2 | **233** | 7 | Auto-test on training completion; 7 still training |
| CG Cityscapes | **125** | 0 | **100%** ✅ |
| CG ACDC | **125** | 0 | **100%** ✅ |
| CS-Ratio CS | **37/37** | 0 | Auto-tested at training completion |
| CS-Ratio ACDC | **37/37** | 0 | Complete |

### 100% Coverage Plan

#### S1 Path to 100% — IN PROGRESS (80GB GPUs) 🔄
| Step | Items | Status | Notes |
|------|-------|--------|-------|
| 1. S1 testing of trained models | **420/420** | ✅ **100%** | 0 missing |
| 2. Remaining S1 training | ~34 configs | 🔄 chge7185 (80GB GPUs) | 6 RUN (mask2former on MapVistas+OUTSIDE15k) |
| 3. Submit tests for new completions | Auto | ⏳ After step 2 | `auto_submit_tests.py --stage 1` |
| 4. Legacy 80k models | 3 configs | ✅ **COMPLETE** | All reached iter_80000 + tested (43.4-43.8%) |

### Strategy Leaderboard Highlights (2026-02-12 21:20)
| Stage | Top Strategy | mIoU | Baseline mIoU | Strategies > Baseline | Results |
|-------|-------------|------|---------------|----------------------|--------|
| Stage 1 | gen_UniControl | **40.37%** | 37.61% | **25/25 (all!)** | 420 |
| Stage 1 #2 | gen_cyclediffusion | 40.12% | 37.61% | — | 420 |
| Stage 1 #3 | std_randaugment | 40.04% | 37.61% | — | 420 |
| Stage 2 | gen_Attribute_Hallucination | **41.77%** | 40.80% | 12/21 | 240 |
| Stage 2 #2 | gen_UniControl | 41.04% | 40.80% | — | 240 |
| Stage 2 #3 | gen_Qwen_Image_Edit | 40.97% | 40.80% | — | 240 |
| CG overall | gen_Img2Img | **52.87%** | 52.65% | 4/24 | 250 |
| CG #2 | gen_augmenters | 52.82% | 52.65% | — | 250 |
| CG #3 | gen_Qwen_Image_Edit | 52.67% | 52.65% | — | 250 |

**Key findings:**
- S1: **All 25** augmentation strategies beat baseline (+1.16 to +2.76 pp). Top-5: gen_UniControl, gen_cyclediffusion, std_randaugment, gen_Img2Img, std_autoaugment
- S2: gen_Attribute_Hallucination leads (+0.98 pp, only 4 models); gen_UniControl #2 (+0.24 pp, 16 models). 12/21 beat baseline. std_* strategies **hurt** (-0.04 to -1.97 pp)
- CG: gen_Img2Img leads (+0.22 pp overall). Only 4/24 beat baseline — CG effect sizes much smaller than S1/S2
- **Cross-stage consistency:** gen_Attribute_Hallucination is cross-stage champion (#4 S1, #1 S2, #4 CG). gen_Img2Img consistently top-5 (#4 S1, #5 S2, #1 CG)
- **mask2former paradox resolved:** S1 degradation driven by rare vehicle class memorization in BDD10k
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

32 jobs designed (commit `4262e8b`). Tests whether models learn from image content or just label layouts.
```bash
python scripts/noise_ablation_submission.py --dry-run
```

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

#### Cityscapes Ratio Ablation (48 jobs — 77% complete)
- **Status:** 🔄 37/48 complete, 2 RUN, **9 NOT SUBMITTED** — need `--stage cityscapes-ratio -y`
- **Ratios:** 0.0, 0.25, 0.75 (0.5 and 1.0 already available from cityscapes-gen)
- **Strategies:** 3 Diffusion (gen_VisualCloze, gen_step1x_v1p2, gen_flux_kontext) + 1 GAN (gen_TSIT)
- **Models:** pspnet_r50, segformer_mit-b3, segnext_mscan-b, mask2former_swin-b
- **Note:** Cityscapes in-domain shows only ~2% spread (near-optimal), so focus analysis on ACDC cross-domain results (3.5% spread)
```bash
python scripts/batch_training_submission.py --stage cityscapes-ratio --dry-run
python scripts/batch_training_submission.py --stage cityscapes-ratio -y  # SUBMIT 9 remaining
```

#### Stage 1 Ratio Ablation (24 jobs READY TO SUBMIT)
- **Status:** ⏳ Stage prepared, ready to submit after cityscapes-ratio analysis
- **Ratios:** 0.0, 0.25, 0.75
- **Datasets:** BDD10k (21.5% spread), IDD-AW (20.9% spread) — **10x larger effect sizes than Cityscapes!**
- **Strategies:** gen_VisualCloze (Diffusion), gen_TSIT (GAN)
- **Models:** pspnet_r50, segformer_mit-b3
```bash
python scripts/batch_training_submission.py --stage stage1-ratio --dry-run
python scripts/batch_training_submission.py --stage stage1-ratio -y
```

| Study | Spread | Jobs | Status |
|-------|--------|------|--------|
| Cityscapes-ratio | 2-3.5% | 48 | 🔄 ~81% complete, 9 running |
| Stage1-ratio | 20-24% | 24 | ⏳ Ready |

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

### §7c: Extended Training Ablation Study

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

#### Status: ⏳ Ready to submit after S1 mask2former + CG deeplabv3plus complete

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
