# Missing Results Report

## Summary

| Metric | Count |
| --- | ---: |
| Total strategies | 25 |
| Expected configs per strategy | 24 (4 datasets × 6 models) |
| Total expected configs | 600 |
| Missing overall mIoU results | 37 |
| Missing per-domain results | 132 |

## Missing Overall mIoU Results

| Strategy | Present | Missing | Details |
| --- | ---: | ---: | --- |
| gen_augmenters | 3/24 | 21 | bdd10k: 3 models, idd-aw: ALL, mapillaryvistas: ALL, outside15k: ALL |
| gen_Qwen_Image_Edit | 8/24 | 16 | bdd10k: ALL, idd-aw: 4 models, outside15k: ALL |

## Per-Domain Detailed Results Coverage

These files (results.json in test_results_detailed) contain per-domain mIoU values
used for Normal Gain and Adverse Gain calculations.

| Strategy | Has results.json | Complete (7 domains) | Coverage |
| --- | ---: | ---: | --- |
| gen_Attribute_Hallucination | 24/24 | 24/24 | ██████████ 100% |
| gen_EDICT | 24/24 | 24/24 | ██████████ 100% |
| gen_IP2P | 24/24 | 24/24 | ██████████ 100% |
| gen_Img2Img | 24/24 | 24/24 | ██████████ 100% |
| gen_LANIT | 24/24 | 24/24 | ██████████ 100% |
| gen_NST | 24/24 | 24/24 | ██████████ 100% |
| gen_SUSTechGAN | 24/24 | 24/24 | ██████████ 100% |
| gen_TSIT | 24/24 | 24/24 | ██████████ 100% |
| gen_UniControl | 24/24 | 24/24 | ██████████ 100% |
| gen_Weather_Effect_Generator | 24/24 | 24/24 | ██████████ 100% |
| gen_automold | 24/24 | 24/24 | ██████████ 100% |
| gen_flux1_kontext | 24/24 | 24/24 | ██████████ 100% |
| gen_stargan_v2 | 24/24 | 24/24 | ██████████ 100% |
| gen_step1x_new | 24/24 | 24/24 | ██████████ 100% |
| photometric_distort | 24/24 | 24/24 | ██████████ 100% |
| gen_CUT | 13/24 | 13/24 | █████░░░░░ 54% |
| baseline | 12/24 | 12/24 | █████░░░░░ 50% |
| gen_StyleID | 12/24 | 12/24 | █████░░░░░ 50% |
| gen_cycleGAN | 12/24 | 12/24 | █████░░░░░ 50% |
| std_autoaugment | 12/24 | 12/24 | █████░░░░░ 50% |
| std_cutmix | 12/24 | 12/24 | █████░░░░░ 50% |
| std_mixup | 12/24 | 12/24 | █████░░░░░ 50% |
| std_randaugment | 12/24 | 12/24 | █████░░░░░ 50% |
| gen_Qwen_Image_Edit | 8/24 | 8/24 | ███░░░░░░░ 33% |
| gen_augmenters | 3/24 | 3/24 | █░░░░░░░░░ 12% |

## Missing Per-Domain Results Details

| Strategy | Missing | Reasons |
| --- | ---: | --- |
| gen_augmenters | 21 | model folder missing: 3, dataset folder missing: 18 |
| gen_Qwen_Image_Edit | 16 | dataset folder missing: 12, model folder missing: 4 |
| baseline | 12 | results.json missing: 12 |
| gen_StyleID | 12 | results.json missing: 12 |
| gen_cycleGAN | 12 | results.json missing: 12 |
| std_autoaugment | 12 | results.json missing: 12 |
| std_cutmix | 12 | results.json missing: 12 |
| std_mixup | 12 | results.json missing: 12 |
| std_randaugment | 12 | results.json missing: 12 |
| gen_CUT | 11 | results.json missing: 11 |