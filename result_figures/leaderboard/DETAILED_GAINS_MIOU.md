# Detailed Per-Dataset and Per-Domain Analysis (mIoU)

## Per-Dataset mIoU by Strategy

| Strategy | Type | bdd10k | Δbdd10k | idd-aw | Δidd-aw | mapillaryvistas | Δmapillaryvistas | outside15k | Δoutside15k | Avg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gen_step1x_v1p2 | Generative | 45.7 | +0.1 | - | - | - | - | - | - | +0.12 |
| std_photometric_distort | Augmentation | 45.6 | +0.0 | 40.5 | +0.5 | - | - | - | - | +0.24 |
| gen_cycleGAN | Generative | 45.8 | +0.2 | 40.2 | +0.1 | - | - | - | - | +0.15 |
| std_cutmix | Standard Aug | 45.4 | -0.2 | 40.3 | +0.3 | - | - | - | - | +0.05 |
| gen_flux_kontext | Generative | 45.5 | -0.1 | 40.2 | +0.2 | - | - | - | - | +0.03 |
| gen_automold | Generative | 45.4 | -0.2 | 40.2 | +0.2 | - | - | - | - | +0.01 |
| std_mixup | Standard Aug | 45.3 | -0.3 | 40.3 | +0.3 | - | - | - | - | -0.00 |
| gen_step1x_new | Generative | 45.2 | -0.4 | 40.3 | +0.3 | - | - | - | - | -0.05 |
| std_autoaugment | Standard Aug | 45.5 | -0.1 | 40.0 | -0.0 | - | - | - | - | -0.08 |
| gen_albumentations_weather | Generative | 45.3 | -0.3 | 40.1 | +0.1 | - | - | - | - | -0.14 |
| gen_LANIT | Generative | 45.2 | -0.4 | 40.1 | +0.1 | - | - | - | - | -0.17 |
| baseline | Baseline | 45.6 | +0.0 | 40.1 | +0.0 | 37.6 | +0.0 | - | - | +0.00 |
| std_minimal | Standard Aug | 46.1 | +0.5 | 40.9 | +0.8 | 35.7 | -1.9 | - | - | -0.19 |

## Per-Domain mIoU by Strategy

| Strategy                   | Type         |   clear_day |   cloudy |   dawn_dusk |   foggy |   night |   rainy |   snowy |   Normal Avg |   Adverse Avg |   Gap |
|:---------------------------|:-------------|------------:|---------:|------------:|--------:|--------:|--------:|--------:|-------------:|--------------:|------:|
| baseline                   | Baseline     |       44.93 |    42.04 |       36.26 |   41.3  |   27.62 |   35.84 |   37.91 |        43.48 |         35.67 |  7.82 |
| gen_LANIT                  | Generative   |       45.93 |    42.74 |       36.02 |   42.76 |   26.59 |   35.49 |   38.47 |        44.34 |         35.83 |  8.51 |
| gen_albumentations_weather | Generative   |       46.16 |    43.09 |       36.05 |   41.51 |   26.15 |   34.63 |   38.33 |        44.63 |         35.16 |  9.47 |
| gen_automold               | Generative   |       46.11 |    43.17 |       36.44 |   41.33 |   26.3  |   35.49 |   38.24 |        44.64 |         35.34 |  9.3  |
| gen_cycleGAN               | Generative   |       46.24 |    43    |       36.16 |   41.62 |   26.38 |   35.38 |   39.16 |        44.62 |         35.64 |  8.98 |
| gen_flux_kontext           | Generative   |       45.9  |    43.31 |       36.36 |   43.02 |   26.66 |   35.78 |   39.01 |        44.6  |         36.12 |  8.48 |
| gen_step1x_new             | Generative   |       45.89 |    43.37 |       36.23 |   42.19 |   26.13 |   35.79 |   38.59 |        44.63 |         35.68 |  8.95 |
| gen_step1x_v1p2            | Generative   |       47.52 |    51.52 |       39.09 |   54.86 |   24.69 |   44.26 |   52.53 |        49.52 |         44.08 |  5.44 |
| std_autoaugment            | Standard Aug |       46.02 |    42.85 |       36.17 |   40.38 |   26.65 |   35.2  |   38.8  |        44.44 |         35.26 |  9.18 |
| std_cutmix                 | Standard Aug |       46.13 |    43.21 |       35.63 |   41.06 |   26.77 |   35.65 |   38.84 |        44.67 |         35.58 |  9.09 |
| std_minimal                | Standard Aug |       43.45 |    41.08 |       37.29 |   40.19 |   28.15 |   35.53 |   37    |        42.27 |         35.22 |  7.05 |
| std_mixup                  | Standard Aug |       45.92 |    42.95 |       36.55 |   42.62 |   26.68 |   35.72 |   38.62 |        44.43 |         35.91 |  8.52 |
| std_photometric_distort    | Augmentation |       45.98 |    43.48 |       37.46 |   43.48 |   27.52 |   36.7  |   38.96 |        44.73 |         36.66 |  8.07 |