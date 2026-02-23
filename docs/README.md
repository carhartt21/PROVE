# PROVE Documentation

**Last Updated:** 2026-02-08

---

## Progress Tracking (auto-generated)

These files are maintained by `update_training_tracker.py` and `update_testing_tracker.py`.

### Training Trackers

| Stage | Tracker | Coverage |
|-------|---------|----------|
| **Stage 1** (clear_day) | [TRAINING_TRACKER_STAGE1.md](TRAINING_TRACKER_STAGE1.md) | [TRAINING_COVERAGE_STAGE1.md](TRAINING_COVERAGE_STAGE1.md) |
| **Stage 2** (all domains) | [TRAINING_TRACKER_STAGE2.md](TRAINING_TRACKER_STAGE2.md) | [TRAINING_COVERAGE_STAGE2.md](TRAINING_COVERAGE_STAGE2.md) |
| **Cityscapes-Gen** | [TRAINING_TRACKER_CITYSCAPES_GEN.md](TRAINING_TRACKER_CITYSCAPES_GEN.md) | — |

### Testing Trackers

| Stage | Tracker | Coverage |
|-------|---------|----------|
| **Stage 1** | [TESTING_TRACKER.md](TESTING_TRACKER.md) | [TESTING_COVERAGE.md](TESTING_COVERAGE.md) |
| **Stage 2** | [TESTING_TRACKER_STAGE2.md](TESTING_TRACKER_STAGE2.md) | [TESTING_COVERAGE_STAGE2.md](TESTING_COVERAGE_STAGE2.md) |
| **Cityscapes-Gen** | [TESTING_TRACKER_CITYSCAPES_GEN.md](TESTING_TRACKER_CITYSCAPES_GEN.md) | [TESTING_COVERAGE_CITYSCAPES_GEN.md](TESTING_COVERAGE_CITYSCAPES_GEN.md) |

### Status Overviews

| Document | Purpose |
|----------|---------|
| [EVALUATION_STAGE_STATUS.md](EVALUATION_STAGE_STATUS.md) | **Master overview** of all stages, leaderboards, ablation studies, and next steps |
| [BASELINE_OVERVIEW.md](BASELINE_OVERVIEW.md) | Auto-generated baseline training & testing status |

---

## Technical Reference

| Document | Purpose |
|----------|---------|
| [UNIFIED_TRAINING.md](UNIFIED_TRAINING.md) | Training pipeline (`unified_training.py`, `batch_training_submission.py`) |
| [UNIFIED_TESTING.md](UNIFIED_TESTING.md) | Testing pipeline (`fine_grained_test.py`, `batch_test_submission.py`) |
| [EVALUATION_CONCEPT.md](EVALUATION_CONCEPT.md) | Experimental design: two-variant training, domain gap analysis |
| [LABEL_HANDLING.md](LABEL_HANDLING.md) | Dataset label formats (RGB-encoded, trainIds, channel order) |
| [LOSS_CONFIGURATION.md](LOSS_CONFIGURATION.md) | Multi-loss training setup (CE + auxiliary losses) |
| [RESULT_VISUALIZATION.md](RESULT_VISUALIZATION.md) | Visualization scripts reference |

## Dataset & Model Analysis

| Document | Purpose |
|----------|---------|
| [MAPILLARYVISTAS_CLASS_ANALYSIS.md](MAPILLARYVISTAS_CLASS_ANALYSIS.md) | Only 23/66 classes have GT in test set — dataset characteristic |
| [CROP_SIZE_IMPACT_ANALYSIS.md](CROP_SIZE_IMPACT_ANALYSIS.md) | CNN models lose 8-15% mIoU at small crops; Transformers are robust |
| [CITYSCAPES_COMPARISON_RESULTS.md](CITYSCAPES_COMPARISON_RESULTS.md) | Pipeline verification: PROVE vs reference mmseg configs |
| [CITYSCAPES_ACDC_CROSS_DOMAIN_RESULTS.md](CITYSCAPES_ACDC_CROSS_DOMAIN_RESULTS.md) | Cross-domain evaluation (Cityscapes → ACDC per-domain) |

## Ablation Study Guides

| Document | Purpose |
|----------|---------|
| [EXTENDED_TRAINING.md](EXTENDED_TRAINING.md) | Extended training ablation (80k→320k iterations) |
| [NOISE_ABLATION_STUDY.md](NOISE_ABLATION_STUDY.md) | Noise ablation: do models learn from image content or label layouts? |
| [RATIO_ABLATION_SUBMISSION_GUIDE.md](RATIO_ABLATION_SUBMISSION_GUIDE.md) | How to submit ratio ablation jobs |

## Historical Reference

| Document | Purpose |
|----------|---------|
| [BUG_REPORT_CROSS_DATASET_CONTAMINATION.md](BUG_REPORT_CROSS_DATASET_CONTAMINATION.md) | MixedDataLoader bug (Jan 28, 2026) — gen images never loaded. **FIXED.** |

---

## Archived Documentation

| Archive | Contents |
|---------|----------|
| [archived_20260208/](archived_20260208/) | Outdated status overviews, stale trackers, pre-bug-fix analysis |
| [archived_20260128/](archived_20260128/) | Historical bug fix records and planning docs |
| [archived_invalid_results_20260128/](archived_invalid_results_20260128/) | Analysis files containing invalid gen_* results |
