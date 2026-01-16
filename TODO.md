# PROVE Project TODO List

**Last Updated:** 2026-01-23

## In Progress

### Extended Training Testing
- [x] Modified `submit_test_extended_training.sh` to use `fine_grained_test.py`
- [x] Added `find_dataset_dir()` function to handle directory naming variations
- [x] Submitted 504 test jobs (9 strategies × 4 datasets × 2 models × 7 iterations)
- [ ] Wait for all extended training test jobs to complete
- [ ] Analyze complete results with `analyze_extended_training.py`
- [ ] Generate final IEEE figures with full dataset

### Domain Adaptation Ablation (READY)
- [x] Updated `submit_domain_adaptation_ablation.sh` for correct weights paths
- [x] Fixed `conda activate` → `mamba activate`
- [x] Script now uses WEIGHTS_STAGE_2 for full dataset, WEIGHTS for clear_day
- [x] Added support for Top 15 augmentation strategies
- [x] Updated documentation with strategy tables
- [ ] Submit baseline jobs (8 available): `./scripts/submit_domain_adaptation_ablation.sh --all-full`
- [ ] Submit strategy jobs (76 available): `./scripts/submit_domain_adaptation_ablation.sh --all-strategies`
- [ ] Analyze results with `analyze_domain_adaptation_ablation.py`

### Analysis Scripts
- [x] Updated `analyze_extended_training.py` with Pattern 5 for new output format
- [x] Created `generate_ieee_figures_extended_training.py` for publication figures
- [x] Updated documentation in `docs/EXTENDED_TRAINING.md`
- [x] Updated `docs/TESTING_TRACKER.md` with extended training section
- [x] Updated `docs/DOMAIN_ADAPTATION_ABLATION.md` with correct paths

## Pending

### Publication
- [ ] Finalize extended training analysis figures
- [ ] Run statistical significance tests on extended training results
- [ ] Prepare tables for publication

### Job Monitoring
- [ ] Monitor extended training test jobs: `bjobs | grep test_ext`
- [ ] Check for failed jobs and resubmit if needed
- [ ] Document any strategies that fail consistently

## Completed

### Domain Adaptation Infrastructure (2026-01-23)
- [x] Script updated to use WEIGHTS_STAGE_2 for full dataset models
- [x] Script updated to use WEIGHTS for clear_day models
- [x] Documentation updated with checkpoint availability table
- [x] 8 jobs available: 5 full dataset + 3 clear_day

### Extended Training Infrastructure (2026-01-23)
- [x] All 9 strategies have complete checkpoints (40k-160k iterations)
- [x] Test command uses `fine_grained_test.py` for per-domain metrics
- [x] Directory naming variations handled (`bdd10k_ad`, `iddaw_ad`, etc.)
- [x] Test output stored in `test_results_iter_{N}/{timestamp}/results.json`
- [x] Analysis scripts support new output format
- [x] IEEE figure generation working with partial results (66 results)

### Documentation Updates (2026-01-23)
- [x] EXTENDED_TRAINING.md - Added testing and analysis sections
- [x] TESTING_TRACKER.md - Added extended training section
- [x] DOMAIN_ADAPTATION_ABLATION.md - Updated checkpoint paths

## Notes

### Domain Adaptation Checkpoint Availability

**Baseline Models:** 8 / 12 available
**Strategy Models:** 76 / 90 available  
**Total:** 84 configurations ready for evaluation

| Dataset | Model | Full (WEIGHTS_STAGE_2) | Clear_day (WEIGHTS) |
|---------|-------|:----------------------:|:-------------------:|
| BDD10k | pspnet_r50 | ✅ | ✅ |
| BDD10k | segformer_mit-b5 | ❌ | ✅ |
| IDD-AW | pspnet_r50 | ✅ | ❌ |
| IDD-AW | segformer_mit-b5 | ✅ | ❌ |
| MapillaryVistas | pspnet_r50 | ✅ | ✅ |
| MapillaryVistas | segformer_mit-b5 | ❌ | ✅ |

**Top 15 Strategies:** gen_cyclediffusion, gen_flux_kontext, gen_step1x_new, gen_step1x_v1p2, gen_stargan_v2, gen_cycleGAN, gen_automold, gen_albumentations_weather, gen_TSIT, gen_UniControl, std_randaugment, std_autoaugment, std_cutmix, std_mixup, photometric_distort

### Extended Training Test Configuration
- **Strategies:** gen_albumentations_weather, gen_automold, gen_cyclediffusion, gen_cycleGAN, gen_flux_kontext, gen_step1x_new, gen_TSIT, gen_UniControl, std_randaugment
- **Datasets:** BDD10k, IDD-AW, MapillaryVistas, OUTSIDE15k
- **Models:** segformer_mit-b5, pspnet_r50
- **Iterations:** 40000, 60000, 80000, 100000, 120000, 140000, 160000
- **Total Jobs:** 504

### Key Scripts
- `scripts/submit_test_extended_training.sh` - Submit extended training test jobs
- `scripts/submit_domain_adaptation_ablation.sh` - Submit domain adaptation jobs
- `analysis_scripts/analyze_extended_training.py` - Analyze extended training results
- `analysis_scripts/generate_ieee_figures_extended_training.py` - Generate IEEE figures
- `fine_grained_test.py` - Per-domain, per-class testing

### Output Directories
- Extended training tests: `{WEIGHTS_EXTENDED}/{strategy}/{dataset}/{model}_ratio0p50/test_results_iter_{N}/`
- Extended training figures: `result_figures/extended_training/{ieee/,preview/,data/}`
- Domain adaptation results: `{WEIGHTS}/domain_adaptation_ablation/`
