# PROVE Documentation

**Last Updated:** 2026-01-28

## Primary Documentation

| Document | Purpose |
|----------|---------|
| **[STUDY_COVERAGE_ANALYSIS.md](STUDY_COVERAGE_ANALYSIS.md)** | **Main reference** - Comprehensive analysis of all studies with key findings and figure recommendations |
| [EVALUATION_STAGE_STATUS.md](EVALUATION_STAGE_STATUS.md) | Stage 1 & Stage 2 training/testing status overview |
| [ABLATION_STUDIES_ANALYSIS.md](ABLATION_STUDIES_ANALYSIS.md) | Consolidated ablation study findings |
| [STAGE_COMPARISON_ANALYSIS.md](STAGE_COMPARISON_ANALYSIS.md) | Stage 1 vs Stage 2 comparison |

## Study-Specific Documentation

| Document | Study |
|----------|-------|
| [EXTENDED_TRAINING.md](EXTENDED_TRAINING.md) | Extended training (80K→320K iterations) details |
| [DOMAIN_ADAPTATION_ABLATION.md](DOMAIN_ADAPTATION_ABLATION.md) | Cross-dataset transfer evaluation (ACDC) |

## Technical Reference

| Document | Purpose |
|----------|---------|
| [UNIFIED_TRAINING.md](UNIFIED_TRAINING.md) | Training pipeline documentation |
| [UNIFIED_TESTING.md](UNIFIED_TESTING.md) | Testing pipeline documentation |
| [LABEL_HANDLING.md](LABEL_HANDLING.md) | Dataset label format reference |
| [MAPILLARYVISTAS_CLASS_ANALYSIS.md](MAPILLARYVISTAS_CLASS_ANALYSIS.md) | MapillaryVistas 66-class mapping |
| [RESULT_VISUALIZATION.md](RESULT_VISUALIZATION.md) | Visualization scripts reference |
| [EVALUATION_CONCEPT.md](EVALUATION_CONCEPT.md) | Original evaluation design |

## Progress Tracking

| Stage | Training | Testing |
|-------|----------|---------|
| **Stage 1** | [TRAINING_TRACKER_STAGE1.md](TRAINING_TRACKER_STAGE1.md) | — |
| **Stage 2** | [TRAINING_TRACKER_STAGE2.md](TRAINING_TRACKER_STAGE2.md) | [TESTING_TRACKER_STAGE2.md](TESTING_TRACKER_STAGE2.md) |

**Coverage Matrices:**
- [TRAINING_COVERAGE_STAGE1.md](TRAINING_COVERAGE_STAGE1.md)
- [TRAINING_COVERAGE_STAGE2.md](TRAINING_COVERAGE_STAGE2.md)
- [TESTING_COVERAGE_STAGE2.md](TESTING_COVERAGE_STAGE2.md)

## Archived Documentation

Historical and superseded documents are in [archived_20260128/](archived_20260128/):
- Bug fix records (completed)
- Earlier analysis versions
- Planning documents

## Key Findings Quick Reference

| Study | Best Strategy | Key Insight |
|-------|--------------|-------------|
| **Stage 1** | gen_Attribute_Hallucination (39.83%) | +1.4 mIoU over baseline |
| **Stage 2** | gen_stargan_v2 (41.73%) | Domain gap shrinks 5-6× |
| **Ratio Ablation** | 0.75 ratio (25% gen) | +1.56% mIoU optimal |
| **Extended Training** | 77% configs improve | Baseline overfits after 90K |
| **Combinations** | std_mixup+photometric (45.22%) | photometric_distort dominates |
| **Domain Adaptation** | All 15/15 beat baseline | +1.03% to +1.96% |

See [STUDY_COVERAGE_ANALYSIS.md](STUDY_COVERAGE_ANALYSIS.md) for detailed findings and recommended figures.
