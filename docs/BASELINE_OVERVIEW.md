# Baseline Training & Testing Overview

**Generated:** 2026-02-03 17:41

## Summary

| Status | Count | Percentage |
|--------|------:|----------:|
| ✅ Complete (trained + tested) | 29 | 60.4% |
| 🔶 Trained (no test results) | 0 | 0.0% |
| 🔄 Running | 0 | 0.0% |
| ❌ Failed | 11 | 22.9% |
| ⏳ Pending | 8 | 16.7% |
| **Total** | **48** | **100%** |

## Stage 1 Baseline Status

| Dataset | DeepLabV3+ | PSPNet | SegFormer | SegNeXt | HRNet | Mask2Former |
|---------|----------|----------|----------|----------|----------|----------|
| BDD10k | ⏳ | ✅ 30.0% | ❌ | ✅ 41.3% | ❌ | ❌ |
| IDD-AW | ⏳ | ❌ | ✅ 34.0% | ✅ 35.1% | ✅ 20.7% | ❌ |
| MapillaryVistas | ⏳ | ✅ 29.0% | ✅ 27.7% | ✅ 34.6% | ✅ 15.2% | ❌ |
| OUTSIDE15k | ⏳ | ✅ 36.0% | ✅ 36.9% | ✅ 38.7% | ✅ 19.8% | ❌ |

### Best mIoU per Model

- **DeepLabV3+**: No results yet
- **PSPNet**: 36.02% (OUTSIDE15k)
- **SegFormer**: 36.87% (OUTSIDE15k)
- **SegNeXt**: 41.27% (BDD10k)
- **HRNet**: 20.69% (IDD-AW)
- **Mask2Former**: No results yet

## Stage 2 Baseline Status

| Dataset | DeepLabV3+ | PSPNet | SegFormer | SegNeXt | HRNet | Mask2Former |
|---------|----------|----------|----------|----------|----------|----------|
| BDD10k | ✅ 30.8% | ✅ 37.0% | ✅ 47.2% | ✅ 47.3% | ⏳ | ✅ 47.0% |
| IDD-AW | ✅ 38.4% | ✅ 33.5% | ✅ 40.6% | ✅ 41.0% | ⏳ | ✅ 40.7% |
| MapillaryVistas | ❌ | ✅ 29.5% | ✅ 34.9% | ✅ 34.9% | ⏳ | ❌ |
| OUTSIDE15k | ❌ | ✅ 36.0% | ✅ 44.8% | ✅ 44.1% | ⏳ | ❌ |

### Best mIoU per Model

- **DeepLabV3+**: 38.37% (IDD-AW)
- **PSPNet**: 37.05% (BDD10k)
- **SegFormer**: 47.16% (BDD10k)
- **SegNeXt**: 47.29% (BDD10k)
- **HRNet**: No results yet
- **Mask2Former**: 47.04% (BDD10k)

## Missing Baseline Configurations

| Stage | Dataset | Model | Status |
|-------|---------|-------|--------|
| Stage 1 | BDD10k | DeepLabV3+ | ⏳ pending |
| Stage 1 | BDD10k | SegFormer | ❌ failed |
| Stage 1 | BDD10k | HRNet | ❌ failed |
| Stage 1 | BDD10k | Mask2Former | ❌ failed |
| Stage 1 | IDD-AW | DeepLabV3+ | ⏳ pending |
| Stage 1 | IDD-AW | PSPNet | ❌ failed |
| Stage 1 | IDD-AW | Mask2Former | ❌ failed |
| Stage 1 | MapillaryVistas | DeepLabV3+ | ⏳ pending |
| Stage 1 | MapillaryVistas | Mask2Former | ❌ failed |
| Stage 1 | OUTSIDE15k | DeepLabV3+ | ⏳ pending |
| Stage 1 | OUTSIDE15k | Mask2Former | ❌ failed |
| Stage 2 | BDD10k | HRNet | ⏳ pending |
| Stage 2 | IDD-AW | HRNet | ⏳ pending |
| Stage 2 | MapillaryVistas | DeepLabV3+ | ❌ failed |
| Stage 2 | MapillaryVistas | HRNet | ⏳ pending |
| Stage 2 | MapillaryVistas | Mask2Former | ❌ failed |
| Stage 2 | OUTSIDE15k | DeepLabV3+ | ❌ failed |
| Stage 2 | OUTSIDE15k | HRNet | ⏳ pending |
| Stage 2 | OUTSIDE15k | Mask2Former | ❌ failed |

## Recommendations

### Priority Training Jobs

**DeepLabV3+** - Missing from 4 configurations:
  - Stage 1/BDD10k
  - Stage 1/IDD-AW
  - Stage 1/MapillaryVistas
  - Stage 1/OUTSIDE15k

**HRNet** - Missing from 4 configurations:
  - Stage 2/BDD10k
  - Stage 2/IDD-AW
  - Stage 2/MapillaryVistas
  - Stage 2/OUTSIDE15k
