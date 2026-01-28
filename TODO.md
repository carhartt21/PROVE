# PROVE Project TODO List

**Last Updated:** 2026-01-28 (11:30)

> **Note:** For detailed study results, analysis, and findings, see [docs/STUDY_COVERAGE_ANALYSIS.md](docs/STUDY_COVERAGE_ANALYSIS.md)

---

## 🎯 Active Jobs

### Baseline Extended Training (Jan 28 07:08)
| Job ID | Configuration | Status | Checkpoints |
|--------|---------------|--------|-------------|
| 443834 | ext_baseline_bdd10k_pspnet | **RUN** | 90k✓, 100k✓, 110k✓, 120k✓ |
| 443835 | ext_baseline_bdd10k_segformer | **RUN** | 90k✓, 100k🔄 |
| 443836 | ext_baseline_iddaw_pspnet | **RUN** | 90k✓, 100k✓, 110k✓, 120k✓ |
| 443837 | ext_baseline_iddaw_segformer | **RUN** | 90k✓, 100k🔄 |

**Baseline Tests:** 12 checkpoints, 10 tested, 2 testing (468661, 468662)  

### Blocked (Permission Issues - Need chge7185)
5 checkpoints with `-rw-------` permissions:
- `gen_flux_kontext/iddaw/segformer_mit-b5_ratio0p{00,25,62,75,88}`

---

## 📋 Pending Actions

### Priority 1: Extended Training Baseline
- [x] First batch of tests submitted (8 jobs for 90k-110k checkpoints)
- [x] All 8 tests completed successfully
- [x] Fixed submission script to skip already-tested checkpoints
- [x] 120k tests completed (465658, 465659)
- [x] Submitted 100k segformer tests (468661, 468662)
- [ ] Continue monitoring for more checkpoints (130k-320k)

### Priority 2: Ratio Ablation Completion
- [x] Completed training jobs (291472, 291559, 291560)
- [x] 9 test jobs (3 succeeded, 1 timeout, 5 permission)
- [x] Timeout retest completed (467660) ✅
- [ ] **BLOCKED**: Contact chge7185 to fix permissions on 5 checkpoints

### Priority 3: Documentation
- [x] Analyzed STUDY_COVERAGE for mismatches
- [x] Updated STUDY_COVERAGE_ANALYSIS.md with mismatch findings
- [ ] Final paper figures after baseline extended training results
- [x] All visualizations regenerated (11 plots including baseline comparison)

---

## 📊 Quick Status Overview

| Study | Status | Checkpoints | Tests | Delta |
|-------|--------|-------------|-------|-------|
| **Stage 1** | ✅ COMPLETE | 324 | 324 | 0 |
| **Stage 2** | ✅ COMPLETE | 325 | 325 | 0 |
| **Ratio Ablation** | 🔄 98% | 284 | 279 | 5 (perm) |
| **Extended Training** | 🔄 79% | 970 | 766 | 204 |
| **Combinations** | ✅ COMPLETE | 53 | 53 | 0 |
| **Domain Adaptation** | ✅ COMPLETE | N/A | 64 | N/A |

### Mismatch Details
- **Ratio Ablation**: 5 permission issues (`chge7185` checkpoints), 1 timeout ✅ fixed
- **Extended Training**: 204 = early checkpoints (10k-70k) + ongoing baseline training

---

## ✅ Recently Completed

### Jan 28, 2026 (08:50)
- ✅ Timeout retest (467660) **completed successfully**
- ✅ 120k baseline tests completed (465658, 465659)
- ✅ Submitted 2 new 100k segformer tests (468661, 468662)
- ✅ Ratio Ablation: 279/284 complete (5 permission-blocked)
- ✅ Extended Training: 766/970 complete (baseline progress)

### Jan 28, 2026 (08:45)
- ✅ **Mismatch Analysis Complete** - identified 5+1 ratio ablation issues, 206 extended training gaps
- ✅ Cleaned up 6 empty test directories
- ✅ Resubmitted timeout job (467660) with 2h wall time
- ✅ Identified 5 permission-blocked checkpoints (need chge7185)

### Jan 28, 2026 (08:35)
- ✅ All 8 baseline extended tests completed (90k-110k checkpoints)
- ✅ Fixed submission script to skip already-tested checkpoints  
- ✅ 2 new test jobs submitted for 120k checkpoints (465658, 465659)

### Jan 28, 2026 (08:30)
- ✅ Fixed extended training analysis script to detect `test_results_detailed/iter_*/` pattern
- ✅ Extended training analysis updated: 722 results total, baseline now included
- ✅ **Key Finding**: Baseline degrades after 90k (46.11→43.47), generative strategies don't!
- ✅ All 11 visualizations regenerated including baseline_comparison.png
- ✅ Ratio ablation analysis: Best ratio = **0.75** (mIoU 41.46)
- ✅ 9 ratio ablation test jobs submitted

### Jan 28, 2026 (08:15)
- ✅ Domain adaptation testing complete (64/64) - all 15/15 strategies beat baseline
- ✅ Baseline extended training jobs submitted (4 jobs: 443834-443837)

### Jan 26, 2026
- ✅ Domain adaptation analysis - gen_stargan_v2 best (+1.96% vs baseline)
- ✅ Archived logs investigation - 117 dirs NOT reusable (no checkpoints)

### Jan 25, 2026
- ✅ Stage 1 ratio ablation training submitted - 140 jobs
- ✅ Fixed ratio ablation submission script
- ✅ Moved buggy gen_TSIT ratio ablation to backup

### Jan 24, 2026
- ✅ Stage 1 & Stage 2 fully complete - 648 tests
- ✅ Stage comparison analysis - 6 figures generated
- ✅ Initial checkpoint tests (10k-80k) - 269/392 complete
- ✅ Domain adaptation MV re-tests submitted

---

## 🔑 Key Findings Summary

| Study | Key Result |
|-------|------------|
| **Stage 1** | gen_Attribute_Hallucination best (+1.36% vs baseline at 39.83%), generative consistently outperforms standard aug |
| **Stage 2** | gen_stargan_v2 best (+0.38% vs baseline at 41.73%), gains compress when training includes all domains |
| **Ratio Ablation** | Best ratio = **0.75** (75% real + 25% gen) at 41.46% mIoU; optimal range is 12-38% synthetic |
| **Extended Training** | +12.09% mIoU improvement (10k→320k), 77% configs improve, 75% gains by 160k iters |
| **Extended (Baseline)** | ⚠️ Baseline degrades after 90k (46.11→43.47) - overfitting! |
| **Combinations** | std_mixup+photometric_distort best at 45.22%; +photometric_distort combos dominate |
| **Domain Adaptation** | ALL 15/15 strategies beat baseline (+1.03% to +1.96%), gen_stargan_v2 best |

### Recommended Paper Figures (3 per Study)

| Study | Figure 1 | Figure 2 | Figure 3 |
|-------|----------|----------|----------|
| **Stage 1** | Strategy ranking bar chart | Domain gap scatter plot | Per-dataset heatmap |
| **Stage 2** | Stage 1 vs 2 comparison | Domain gap reduction | Rank change chart |
| **Ratio Ablation** | Ratio vs mIoU line plot | Optimal ratio heatmap | Performance variance boxplot |
| **Extended Training** | Learning curves multi-panel | Convergence heatmap | Diminishing returns plot |
| **Combinations** | Combination matrix heatmap | Component contribution | Combination type boxplot |

---

## 📁 Important Paths

```
# Main weights directories
/scratch/aaa_exchange/AWARE/WEIGHTS/           # Stage 1
/scratch/aaa_exchange/AWARE/WEIGHTS_STAGE_2/   # Stage 2

# Ablation study directories
/scratch/aaa_exchange/AWARE/WEIGHTS_RATIO_ABLATION/
/scratch/aaa_exchange/AWARE/WEIGHTS_EXTENDED/
/scratch/aaa_exchange/AWARE/WEIGHTS_COMBINATIONS/

# Results figures
result_figures/extended_training/
result_figures/ratio_ablation/
result_figures/domain_adaptation/
result_figures/combination_ablation/
```

---

## 🛠️ Useful Commands

```bash
# Check job status
bjobs -w -u mima2416

# Run extended training analysis
python analysis_scripts/analyze_extended_training.py
python analysis_scripts/visualize_extended_training.py

# Submit baseline extended tests
python scripts/submit_baseline_extended_tests.py --dry-run
python scripts/submit_baseline_extended_tests.py --submit

# Run ratio ablation analysis
python analysis_scripts/analyze_ratio_ablation.py
python analysis_scripts/visualize_ratio_ablation.py

# Update trackers
python scripts/update_training_tracker.py --stage 1
python scripts/update_training_tracker.py --stage 2
```
