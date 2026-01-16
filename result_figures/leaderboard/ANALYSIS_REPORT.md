# PROVE Strategy Analysis Report

> **Datasets:** BDD10k, IDD-AW, MapillaryVistas, Outside15k  
> **Models:** DeepLabV3+ (ResNet-50), PSPNet (ResNet-50), SegFormer (MiT-B5)  
> **Note:** ACDC excluded from main analysis (reserved for domain adaptation ablation)

## 1. Baseline Performance Summary

### Overall Baseline
| Metric | Value |
|:---|---:|
| **mIoU** | 60.28% |
| **fwIoU** | 84.12% |

### Per-Dataset Baseline Performance
| Dataset | mIoU |
|:---|---:|
| BDD10k | 54.37% |
| IDD-AW | 77.87% |
| MapillaryVistas | 44.68% |
| Outside15k | 64.19% |

### Per-Model Baseline Performance
| Model | mIoU | fwIoU |
|:---|---:|---:|
| deeplabv3plus_r50 | 53.13% | 80.79% |
| pspnet_r50 | 55.72% | 79.39% |
| segformer_mit-b5 | 71.98% | 92.19% |

## 2. Strategy Leaderboard

**Baseline Reference: 60.28% mIoU**

| Rank | Strategy | Type | Overall mIoU | Δ vs Baseline | BDD10k | IDD-AW | MapVist | Out15k |
|---:|:---|:---|---:|---:|---:|---:|---:|---:|
| 1 | `gen_automold` | Generative | 62.10% | +1.82% | 57.89% | 77.75% | 47.54% | 65.22% |
| 2 | `std_cutmix+std_autoaugment` | Standard Aug | 61.72% | +1.44% | 58.46% | 77.56% | 45.14% | 65.71% |
| 3 | `gen_NST` | Generative | 61.61% | +1.33% | 58.95% | 78.09% | 44.20% | 65.21% |
| 4 | `photometric_distort` | Augmentation | 61.49% | +1.21% | 58.05% | 77.92% | 43.61% | 66.36% |
| 5 | `std_mixup+std_autoaugment` | Standard Aug | 61.39% | +1.11% | 58.48% | 78.37% | 42.62% | 66.08% |
| 6 | `gen_SUSTechGAN` | Generative | 61.37% | +1.09% | 58.65% | 78.11% | 43.63% | 65.09% |
| 7 | `gen_UniControl` | Generative | 61.34% | +1.06% | 58.25% | 77.86% | 43.08% | 66.16% |
| 8 | `gen_Attribute_Hallucination` | Generative | 61.12% | +0.84% | 57.59% | 78.27% | 44.03% | 64.58% |
| 9 | `std_randaugment` | Standard Aug | 61.09% | +0.81% | 59.29% | 77.81% | 41.17% | 66.09% |
| 10 | `std_mixup` | Standard Aug | 61.08% | +0.80% | 60.03% | 77.34% | 41.41% | 65.54% |
| 11 | `gen_LANIT` | Generative | 61.05% | +0.77% | 56.94% | 78.73% | 43.54% | 64.98% |
| 12 | `gen_step1x_new` | Generative | 60.97% | +0.69% | 58.14% | 78.01% | 41.75% | 65.96% |
| 13 | `std_cutmix` | Standard Aug | 60.94% | +0.66% | 57.36% | 78.85% | 42.12% | 65.42% |
| 14 | `gen_CUT` | Generative | 60.93% | +0.65% | 58.17% | 78.66% | 42.86% | 64.04% |
| 15 | `gen_flux1_kontext` | Generative | 60.86% | +0.58% | 57.67% | 78.63% | 41.32% | 65.83% |
| 16 | `gen_TSIT` | Generative | 60.65% | +0.37% | 58.74% | 77.69% | 41.00% | 65.15% |
| 17 | `gen_Weather_Effect_Generator` | Generative | 60.60% | +0.32% | 57.50% | 77.86% | 42.08% | 64.96% |
| 18 | `gen_IP2P` | Generative | 60.57% | +0.29% | 56.76% | 77.72% | 42.31% | 65.47% |
| 19 | `gen_EDICT` | Generative | 60.55% | +0.27% | 57.92% | 78.21% | 40.89% | 65.17% |
| 20 | `gen_StyleID` | Generative | 60.41% | +0.13% | 56.44% | 78.24% | 41.03% | 65.92% |
| 21 | `gen_stargan_v2` | Generative | 60.40% | +0.12% | 57.76% | 76.01% | 41.99% | 65.84% |
| 22 | `gen_Img2Img` | Generative | 60.39% | +0.11% | 58.52% | 77.74% | 40.84% | 64.45% |
| 23 | `gen_CUT+std_mixup` | Generative | 60.28% | +0.00% | 59.18% | 77.71% | 38.63% | 65.61% |
| 24 | `baseline` | Baseline | 60.28% | 0.00% | 54.37% | 77.87% | 44.68% | 64.19% |
| 25 | `gen_cycleGAN` | Generative | 60.12% | -0.15% | 57.88% | 77.81% | 41.05% | 63.76% |
| 26 | `std_randaugment+std_mixup` | Standard Aug | 60.08% | -0.20% | 59.37% | 77.85% | 38.56% | 64.54% |
| 27 | `gen_CUT+std_randaugment` | Generative | 59.65% | -0.63% | 58.16% | 77.74% | 37.66% | 65.03% |
| 28 | `std_autoaugment` | Standard Aug | 59.64% | -0.63% | 57.97% | 77.28% | 39.64% | 63.69% |
| 29 | `std_randaugment+std_autoaugment` | Standard Aug | 59.60% | -0.68% | 59.62% | 77.73% | 37.33% | 63.71% |
| 30 | `std_randaugment+std_cutmix` | Standard Aug | 59.59% | -0.69% | 58.14% | 77.85% | 36.73% | 65.63% |
| 31 | `gen_cycleGAN+std_randaugment` | Generative | 59.23% | -1.05% | 59.23% | 77.51% | 38.28% | 61.91% |
| 32 | `gen_cycleGAN+std_mixup` | Generative | 59.16% | -1.12% | 57.29% | 77.53% | 36.29% | 65.53% |
| 33 | `std_mixup+std_cutmix` | Standard Aug | 58.99% | -1.29% | 58.01% | 77.79% | 34.50% | 65.65% |

## 3. Summary Statistics

### Strategy Type Performance
| Type | Count | Avg mIoU | Best mIoU | Above Baseline |
|:---|---:|---:|---:|---:|
| Generative | 21 | 60.64% | 62.10% | 17 |
| Standard Aug | 10 | 60.41% | 61.72% | 5 |
| Augmentation | 1 | 61.49% | 61.49% | 1 |
| Baseline | 1 | 60.28% | - | - |

### Key Findings

1. **Best Overall Strategy**: `gen_automold` (62.10% mIoU, +1.82%)
2. **Best Generative**: `gen_automold` (62.10% mIoU)
3. **Best Standard Aug**: `std_cutmix+std_autoaugment` (61.72% mIoU)
4. **Total Strategies Above Baseline**: 23 / 33

### Dataset-Specific Recommendations

| Dataset | Best Strategy | Improvement |
|:---|:---|---:|
| **BDD10k** | `std_mixup` | +5.66% |
| **IDD-AW** | `std_cutmix` | +0.98% |
| **MapillaryVistas** | `gen_automold` | +2.86% |
| **Outside15k** | `photometric_distort` | +2.17% |

## 4. Notes

- ACDC dataset is excluded from main evaluation and reserved for domain adaptation ablation study
- Models are trained separately on each dataset's training set
- Results averaged across 3 model architectures (DeepLabV3+, PSPNet, SegFormer)
