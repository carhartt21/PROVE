# PROVE Documentation

**Last Updated:** 2026-01-28 (13:30)

---

## ⚠️ CRITICAL: gen_* Results Invalid

> **MixedDataLoader Bug (Jan 28, 2026):** Generated images were **NEVER LOADED** during training.
> All `gen_*` strategy results are **INVALID**. Bug is **FIXED** - retraining required.
> 
> See [BUG_REPORT](BUG_REPORT_CROSS_DATASET_CONTAMINATION.md) for details.

---

## Primary Documentation

| Document | Purpose | Status |
|----------|---------|--------|
| **[STUDY_COVERAGE_ANALYSIS.md](STUDY_COVERAGE_ANALYSIS.md)** | Main reference - all study findings | ⚠️ gen_* findings invalid |
| **[BUG_REPORT_CROSS_DATASET_CONTAMINATION.md](BUG_REPORT_CROSS_DATASET_CONTAMINATION.md)** | MixedDataLoader bug details | ✅ **READ THIS FIRST** |
| [EVALUATION_STAGE_STATUS.md](EVALUATION_STAGE_STATUS.md) | Stage 1 & Stage 2 status | ⚠️ gen_* invalid |
| [ABLATION_STUDIES_ANALYSIS.md](ABLATION_STUDIES_ANALYSIS.md) | Ablation study findings | ⚠️ Ratio ablation invalid |

## Study-Specific Documentation

| Document | Study | Status |
|----------|-------|--------|
| [EXTENDED_TRAINING.md](EXTENDED_TRAINING.md) | Extended training (80K→320K) | ⚠️ Baseline only valid |

## Technical Reference (Still Valid)

| Document | Purpose |
|----------|---------|
| [UNIFIED_TRAINING.md](UNIFIED_TRAINING.md) | Training pipeline documentation |
| [UNIFIED_TESTING.md](UNIFIED_TESTING.md) | Testing pipeline documentation |
| [LABEL_HANDLING.md](LABEL_HANDLING.md) | Dataset label format reference |
| [MAPILLARYVISTAS_CLASS_ANALYSIS.md](MAPILLARYVISTAS_CLASS_ANALYSIS.md) | MapillaryVistas 66-class mapping |
| [RESULT_VISUALIZATION.md](RESULT_VISUALIZATION.md) | Visualization scripts reference |
| [EVALUATION_CONCEPT.md](EVALUATION_CONCEPT.md) | Original evaluation design |

## Progress Tracking (⚠️ gen_* Invalid)

| Stage | Training | Testing | Status |
|-------|----------|---------|--------|
| **Stage 1** | [TRAINING_TRACKER_STAGE1.md](TRAINING_TRACKER_STAGE1.md) | — | ⚠️ gen_* invalid |
| **Stage 2** | [TRAINING_TRACKER_STAGE2.md](TRAINING_TRACKER_STAGE2.md) | [TESTING_TRACKER_STAGE2.md](TESTING_TRACKER_STAGE2.md) | ⚠️ gen_* invalid |

## Archived Documentation

| Archive | Contents |
|---------|----------|
| [archived_invalid_results_20260128/](archived_invalid_results_20260128/) | Analysis files containing invalid gen_* results |
| [archived_20260128/](archived_20260128/) | Historical bug fix records and planning docs |

## Key Findings Quick Reference

| Study | Best Strategy | Key Insight |
|-------|--------------|-------------|
| **Stage 1** | gen_Attribute_Hallucination (39.83%) | +1.4 mIoU over baseline |
| **Stage 2** | gen_stargan_v2 (41.73%) | Domain gap shrinks 5-6× |
| **Ratio Ablation** | 0.75 ratio (25% gen) | +1.56% mIoU optimal |
| **Extended Training** | 77% configs improve | Baseline overfits after 90K |
| **Combinations** | std_mixup+photometric (45.22%) | std_std_photometric_distort dominates |
| **Domain Adaptation** | All 15/15 beat baseline | +1.03% to +1.96% |

See [STUDY_COVERAGE_ANALYSIS.md](STUDY_COVERAGE_ANALYSIS.md) for detailed findings and recommended figures.
