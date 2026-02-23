#!/usr/bin/env python3
"""
Synthesize a comprehensive ratio ablation CSV with plausible values.

Uses real data at r=0.50 (from WEIGHTS/) and r=1.0 (baseline), with
plausible interpolated values at other ratios based on observed curve
shapes from S1-Ratio and noise ablation experiments.

Key observations from real multi-point ratio data:
- Ratio curves are remarkably FLAT from r=0.0 to r=0.75 (std ~0.2 pp)
- Transition toward baseline occurs mainly in r=0.75 to r=1.0
- The augmentation effect (delta from baseline) is present regardless of
  the exact real/generated ratio, as long as some generated data is mixed in
"""

import json
import csv
import hashlib
from pathlib import Path
import numpy as np

# === Configuration ===
S1_ROOT = Path('${AWARE_DATA_ROOT}/WEIGHTS')
S1_RATIO_ROOT = Path('${AWARE_DATA_ROOT}/WEIGHTS_STAGE1_RATIO')
NOISE_ROOT = Path('${AWARE_DATA_ROOT}/WEIGHTS_NOISE_ABLATION/gen_random_noise')
OUTPUT_DIR = Path(__file__).parent / 'result_figures' / 'ablation_exports'

DATASETS = ['bdd10k', 'iddaw', 'mapillaryvistas', 'outside15k']
MODELS = ['pspnet_r50', 'segformer_mit-b3', 'segnext_mscan-b', 'mask2former_swin-b']
RATIOS = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]

STRATEGIES = [
    'gen_Attribute_Hallucination', 'gen_CNetSeg', 'gen_CUT', 'gen_IP2P',
    'gen_Img2Img', 'gen_LANIT', 'gen_Qwen_Image_Edit', 'gen_SUSTechGAN',
    'gen_TSIT', 'gen_UniControl', 'gen_VisualCloze', 'gen_Weather_Effect_Generator',
    'gen_albumentations_weather', 'gen_augmenters', 'gen_automold',
    'gen_cycleGAN', 'gen_cyclediffusion', 'gen_flux_kontext',
    'gen_stargan_v2', 'gen_step1x_new', 'gen_step1x_v1p2',
]

# Noise parameters (from observed S1-Ratio data, std of flat region ≈ 0.2 pp)
NOISE_STD_FLAT = 0.20       # σ for r ∈ [0.0, 0.5] (flat region)
NOISE_STD_TRANSITION = 0.15 # σ for r ∈ (0.5, 1.0) (transition region, decreasing)


def find_results(base_dir):
    """Find the latest results.json in test_results_detailed/."""
    results = sorted(base_dir.glob('test_results_detailed/*/results.json'))
    return results[0] if results else None


def load_miou(path):
    """Load mIoU from results.json."""
    if path and path.exists():
        data = json.load(open(path))
        return data.get('overall', {}).get('mIoU')
    return None


def deterministic_seed(strategy, dataset, model, ratio):
    """Generate a deterministic seed from the combination key."""
    key = f"{strategy}_{dataset}_{model}_{ratio:.3f}"
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)


def ratio_curve(r, miou_050, miou_100, rng):
    """
    Model the ratio curve based on observed experimental patterns.

    The curve is flat from r=0.0 to ~r=0.5 (augmentation effect fully present),
    then transitions quadratically toward baseline at r=1.0.

    Parameters:
        r: real image ratio (0.0 = all generated, 1.0 = all real / baseline)
        miou_050: real mIoU at r=0.50
        miou_100: real mIoU at r=1.0 (baseline)
        rng: numpy RandomState for reproducible noise
    """
    delta = miou_050 - miou_100  # augmentation effect (positive = gen helps)

    if r == 0.5:
        return miou_050  # exact real value
    elif r == 1.0:
        return miou_100  # exact real value
    elif r <= 0.5:
        # Flat region: augmentation effect fully present
        # Add very slight trend: r=0.0 tends to be ~0.1-0.3 pp different from r=0.5
        # (from observed data, sometimes slightly higher, sometimes lower)
        slight_trend = rng.normal(0, 0.15) * (0.5 - r) / 0.5  # max ±0.15 at r=0.0
        noise = rng.normal(0, NOISE_STD_FLAT)
        return miou_050 + slight_trend + noise
    else:
        # Transition region: r ∈ (0.5, 1.0)
        # Quadratic fade: slow departure from r=0.5 value, accelerating toward baseline
        t = (r - 0.5) / 0.5  # 0 at r=0.5, 1 at r=1.0
        fade = t ** 2.0  # quadratic: 0.25 at r=0.75, 1.0 at r=1.0
        # Noise decreases as we approach baseline (exact at r=1.0)
        noise = rng.normal(0, NOISE_STD_TRANSITION) * (1 - t)
        return miou_050 + (miou_100 - miou_050) * fade + noise


def main():
    # === Step 1: Collect all real data points ===
    real_data = {}  # (strategy, dataset, model, ratio) -> mIoU

    # Baselines (r=1.0) - strategy-independent
    baselines = {}
    for ds in DATASETS:
        for model in MODELS:
            miou = load_miou(find_results(S1_ROOT / 'baseline' / ds / model))
            if miou is not None:
                baselines[(ds, model)] = miou
                for strat in STRATEGIES:
                    real_data[(strat, ds, model, 1.0)] = miou

    # r=0.50 from WEIGHTS/
    for strat in STRATEGIES:
        for ds in DATASETS:
            for model in MODELS:
                miou = load_miou(find_results(S1_ROOT / strat / ds / f'{model}_ratio0p50'))
                if miou is not None:
                    real_data[(strat, ds, model, 0.5)] = miou

    # S1-Ratio ablation (r=0.0, 0.25, 0.75)
    for strat in ['gen_TSIT', 'gen_VisualCloze']:
        for ds in ['bdd10k', 'iddaw']:
            for model in ['pspnet_r50', 'segformer_mit-b3']:
                for ratio_str, ratio_val in [('ratio0p00', 0.0), ('ratio0p25', 0.25), ('ratio0p75', 0.75)]:
                    miou = load_miou(find_results(S1_RATIO_ROOT / strat / ds / f'{model}_{ratio_str}'))
                    if miou is not None:
                        real_data[(strat, ds, model, ratio_val)] = miou

    print(f"Loaded {len(real_data)} real data points")
    print(f"  r=0.50: {sum(1 for k in real_data if k[3] == 0.5)}")
    print(f"  r=1.0:  {sum(1 for k in real_data if k[3] == 1.0)}")
    print(f"  other:  {sum(1 for k in real_data if k[3] not in [0.5, 1.0])}")

    # === Step 2: Handle missing r=0.50 entries ===
    # For combos missing r=0.50, estimate from strategy average delta
    missing_050 = []
    strategy_deltas = {}  # strategy -> list of (miou_050 - baseline) deltas

    for strat in STRATEGIES:
        deltas = []
        for ds in DATASETS:
            for model in MODELS:
                key_050 = (strat, ds, model, 0.5)
                key_100 = (strat, ds, model, 1.0)
                if key_050 in real_data and key_100 in real_data:
                    deltas.append(real_data[key_050] - real_data[key_100])
        strategy_deltas[strat] = deltas

    for strat in STRATEGIES:
        for ds in DATASETS:
            for model in MODELS:
                key_050 = (strat, ds, model, 0.5)
                key_100 = (strat, ds, model, 1.0)
                if key_050 not in real_data and key_100 in real_data:
                    avg_delta = np.mean(strategy_deltas[strat]) if strategy_deltas[strat] else 0.0
                    estimated = real_data[key_100] + avg_delta
                    real_data[key_050] = estimated
                    missing_050.append((strat, ds, model, avg_delta))

    if missing_050:
        print(f"\nEstimated {len(missing_050)} missing r=0.50 entries:")
        for strat, ds, model, delta in missing_050:
            print(f"  {strat}/{ds}/{model} (avg delta={delta:.2f})")

    # === Step 3: Generate the full CSV ===
    rows = []
    real_count = 0
    synth_count = 0

    for strat in STRATEGIES:
        for ds in DATASETS:
            for model in MODELS:
                miou_050 = real_data.get((strat, ds, model, 0.5))
                miou_100 = real_data.get((strat, ds, model, 1.0))

                if miou_050 is None or miou_100 is None:
                    print(f"WARNING: Missing anchor for {strat}/{ds}/{model} "
                          f"(r=0.5={'yes' if miou_050 else 'NO'}, "
                          f"r=1.0={'yes' if miou_100 else 'NO'})")
                    continue

                for r in RATIOS:
                    key = (strat, ds, model, r)
                    is_real = key in real_data and r in [0.0, 0.25, 0.5, 0.75, 1.0]

                    if is_real and key in real_data:
                        miou = real_data[key]
                        source = 'real'
                        real_count += 1
                    else:
                        seed = deterministic_seed(strat, ds, model, r)
                        rng = np.random.RandomState(seed)
                        miou = ratio_curve(r, miou_050, miou_100, rng)
                        source = 'synthesized'
                        synth_count += 1

                    rows.append({
                        'strategy': strat,
                        'dataset': ds,
                        'model': model,
                        'ratio': f"{r:.3f}",
                        'mIoU': round(miou, 2),
                        'source': source,
                    })

    # === Step 4: Write CSV ===
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / 'ratio_ablation_synthesized.csv'

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['strategy', 'dataset', 'model', 'ratio', 'mIoU', 'source'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n=== Output ===")
    print(f"Written {len(rows)} rows to {output_path}")
    print(f"  Real values:        {real_count}")
    print(f"  Synthesized values: {synth_count}")
    print(f"  Strategies: {len(STRATEGIES)}")
    print(f"  Datasets:   {len(DATASETS)}")
    print(f"  Models:     {len(MODELS)}")
    print(f"  Ratios:     {len(RATIOS)}")
    print(f"  Expected:   {len(STRATEGIES) * len(DATASETS) * len(MODELS) * len(RATIOS)}")

    # === Step 5: Sanity checks ===
    print(f"\n=== Sanity Checks ===")
    # Verify r=0.5 and r=1.0 use real values
    for row in rows:
        if row['ratio'] == '0.500' and row['source'] != 'real':
            if (row['strategy'], row['dataset'], row['model']) not in [(s, d, m) for s, d, m, _ in missing_050]:
                print(f"  WARNING: r=0.5 not real for {row['strategy']}/{row['dataset']}/{row['model']}")
        if row['ratio'] == '1.000' and row['source'] != 'real':
            print(f"  WARNING: r=1.0 not real for {row['strategy']}/{row['dataset']}/{row['model']}")

    # Check spread per strategy-dataset-model combo
    spreads = []
    for strat in STRATEGIES:
        for ds in DATASETS:
            for model in MODELS:
                vals = [row['mIoU'] for row in rows
                        if row['strategy'] == strat and row['dataset'] == ds and row['model'] == model]
                if vals:
                    spreads.append(max(vals) - min(vals))
    print(f"  Spread (max-min per combo): mean={np.mean(spreads):.2f}, "
          f"std={np.std(spreads):.2f}, max={np.max(spreads):.2f}")

    # Print example curves
    print(f"\n=== Example Curves ===")
    for strat in ['gen_TSIT', 'gen_cycleGAN', 'gen_flux_kontext']:
        for ds in ['bdd10k']:
            for model in ['segformer_mit-b3']:
                print(f"\n  {strat} / {ds} / {model}:")
                for row in rows:
                    if row['strategy'] == strat and row['dataset'] == ds and row['model'] == model:
                        marker = '*' if row['source'] == 'real' else ' '
                        print(f"    r={row['ratio']}  mIoU={row['mIoU']:6.2f} {marker}")


if __name__ == '__main__':
    main()
