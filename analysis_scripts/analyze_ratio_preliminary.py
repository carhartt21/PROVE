#!/usr/bin/env python3
"""Preliminary analysis of CS-Ratio ablation results.
Determines if finer-grained ratios (0.125, 0.375, etc.) are warranted."""

import json
import re
from pathlib import Path
from collections import defaultdict

# Directories
CS_RATIO_DIR = Path('${AWARE_DATA_ROOT}/WEIGHTS_CITYSCAPES_RATIO')
CG_DIR = Path('${AWARE_DATA_ROOT}/WEIGHTS_CITYSCAPES_GEN')

# CS-Ratio strategies and models
STRATEGIES = ['gen_TSIT', 'gen_VisualCloze', 'gen_step1x_v1p2', 'gen_flux_kontext']
MODELS = ['pspnet_r50', 'segformer_mit-b3', 'segnext_mscan-b', 'mask2former_swin-b']


def get_miou(results_json_path):
    """Extract mIoU from results.json."""
    try:
        with open(results_json_path) as f:
            data = json.load(f)
        miou = data.get('overall', {}).get('mIoU', None)
        if miou is None:
            miou = data.get('mIoU', None)
        return miou
    except:
        return None


def find_latest_result(test_dir):
    """Find latest results.json in a test_results_* directory."""
    if not test_dir.exists():
        return None
    timestamps = sorted(test_dir.iterdir())
    for ts_dir in reversed(timestamps):
        rf = ts_dir / 'results.json'
        if rf.exists():
            return get_miou(rf)
    return None


def collect_results():
    """Collect all ratio results from CS-Ratio and CG directories."""
    # key: (strategy, model, test_type) -> {ratio: miou}
    data = defaultdict(dict)

    # 1. CS-Ratio directory (ratios: 0.0, 0.25, 0.75)
    for rfile in CS_RATIO_DIR.rglob('results.json'):
        # Use relative path from base for reliable parsing
        rel = rfile.relative_to(CS_RATIO_DIR)
        parts = rel.parts
        # parts: (strategy, 'cityscapes', model_ratio, test_type, timestamp, 'results.json')
        if len(parts) < 6:
            continue
        strategy = parts[0]
        model_ratio = parts[2]

        ratio_match = re.search(r'ratio(\dp\d+)', model_ratio)
        if not ratio_match:
            continue
        ratio_str = ratio_match.group(1)
        ratio = float(ratio_str.replace('p', '.'))
        model = model_ratio.replace(f'_ratio{ratio_str}', '')

        if 'test_results_acdc' in parts[3]:
            test_type = 'ACDC'
        elif 'test_results_detailed' in parts[3]:
            test_type = 'Cityscapes'
        else:
            continue

        miou = get_miou(rfile)
        if miou is not None:
            key = (strategy, model, test_type)
            # Keep latest if duplicate
            if ratio not in data[key] or True:
                data[key][ratio] = miou

    # 2. CG directory - ratio 0.5
    for strategy in STRATEGIES:
        for model in MODELS:
            model_dir = CG_DIR / strategy / 'cityscapes' / f'{model}_ratio0p50'
            if not model_dir.exists():
                continue
            for test_dir_name, test_type in [('test_results_detailed', 'Cityscapes'),
                                              ('test_results_acdc', 'ACDC')]:
                miou = find_latest_result(model_dir / test_dir_name)
                if miou is not None:
                    data[(strategy, model, test_type)][0.5] = miou

    # 3. Baseline (ratio 1.0 = 100% real)
    for model in MODELS:
        model_dir = CG_DIR / 'baseline' / 'cityscapes' / model
        if not model_dir.exists():
            continue
        for test_dir_name, test_type in [('test_results_detailed', 'Cityscapes'),
                                          ('test_results_acdc', 'ACDC')]:
            miou = find_latest_result(model_dir / test_dir_name)
            if miou is not None:
                for strategy in STRATEGIES:
                    data[(strategy, model, test_type)][1.0] = miou

    return data


def analyze_curves(data):
    """Analyze ratio curves for non-linearity."""
    print("=" * 100)
    print("CS-RATIO ABLATION: PRELIMINARY ANALYSIS")
    print("=" * 100)

    all_ratios = sorted(set(r for ratios in data.values() for r in ratios.keys()))
    print(f"\nAvailable ratio points: {all_ratios}")
    print(f"Total curves: {len(data)}")

    # Print per-strategy/model tables
    for test_type in ['Cityscapes', 'ACDC']:
        print(f"\n{'=' * 80}")
        print(f"TEST SET: {test_type}")
        print(f"{'=' * 80}")

        # Print header
        header = f"{'Strategy':<22} {'Model':<22}"
        for r in all_ratios:
            header += f" {r:>6.2f}"
        header += "   Δ(best-baseline)  Best_ratio  Shape"
        print(header)
        print("-" * len(header))

        strategy_avgs = defaultdict(lambda: defaultdict(list))
        model_avgs = defaultdict(lambda: defaultdict(list))
        all_shapes = []

        for strategy in STRATEGIES:
            for model in MODELS:
                key = (strategy, model, test_type)
                if key not in data:
                    continue
                ratios = data[key]
                if not ratios:
                    continue

                row = f"{strategy:<22} {model:<22}"
                values = []
                for r in all_ratios:
                    if r in ratios:
                        row += f" {ratios[r]:>6.2f}"
                        values.append((r, ratios[r]))
                        strategy_avgs[strategy][r].append(ratios[r])
                        model_avgs[model][r].append(ratios[r])
                    else:
                        row += f" {'---':>6}"

                if len(values) >= 2:
                    baseline = ratios.get(1.0, None)
                    best_ratio = max(values, key=lambda x: x[1])

                    if baseline is not None:
                        delta = best_ratio[1] - baseline
                        row += f"   {delta:>+6.2f} pp"
                        row += f"       {best_ratio[0]:.2f}"

                        # Determine curve shape
                        shape = classify_shape(ratios, all_ratios)
                        row += f"      {shape}"
                        all_shapes.append(shape)
                    else:
                        row += f"   {'---':>9}"

                print(row)

        # Print strategy averages
        print(f"\n--- Strategy Averages ({test_type}) ---")
        header = f"{'Strategy':<22} {'N':>3}"
        for r in all_ratios:
            header += f" {r:>6.2f}"
        header += "   Δ(best-baseline)"
        print(header)
        print("-" * len(header))
        for strategy in STRATEGIES:
            ratios_avg = {}
            counts = {}
            for r in all_ratios:
                if r in strategy_avgs[strategy] and strategy_avgs[strategy][r]:
                    ratios_avg[r] = sum(strategy_avgs[strategy][r]) / len(strategy_avgs[strategy][r])
                    counts[r] = len(strategy_avgs[strategy][r])
            n = max(counts.values()) if counts else 0
            row = f"{strategy:<22} {n:>3}"
            for r in all_ratios:
                if r in ratios_avg:
                    row += f" {ratios_avg[r]:>6.2f}"
                else:
                    row += f" {'---':>6}"
            baseline = ratios_avg.get(1.0)
            if baseline:
                best = max(ratios_avg.values())
                best_r = [r for r, v in ratios_avg.items() if v == best][0]
                row += f"   {best - baseline:>+6.2f} pp @ {best_r:.2f}"
            print(row)

        # Print model averages
        print(f"\n--- Model Averages ({test_type}) ---")
        header = f"{'Model':<22} {'N':>3}"
        for r in all_ratios:
            header += f" {r:>6.2f}"
        header += "   Δ(best-baseline)"
        print(header)
        print("-" * len(header))
        for model in MODELS:
            ratios_avg = {}
            counts = {}
            for r in all_ratios:
                if r in model_avgs[model] and model_avgs[model][r]:
                    ratios_avg[r] = sum(model_avgs[model][r]) / len(model_avgs[model][r])
                    counts[r] = len(model_avgs[model][r])
            n = max(counts.values()) if counts else 0
            row = f"{model:<22} {n:>3}"
            for r in all_ratios:
                if r in ratios_avg:
                    row += f" {ratios_avg[r]:>6.2f}"
                else:
                    row += f" {'---':>6}"
            baseline = ratios_avg.get(1.0)
            if baseline:
                best = max(ratios_avg.values())
                best_r = [r for r, v in ratios_avg.items() if v == best][0]
                row += f"   {best - baseline:>+6.2f} pp @ {best_r:.2f}"
            print(row)

        # Shape distribution
        if all_shapes:
            print(f"\n--- Curve Shapes ({test_type}) ---")
            from collections import Counter
            for shape, count in Counter(all_shapes).most_common():
                pct = count / len(all_shapes) * 100
                print(f"  {shape}: {count}/{len(all_shapes)} ({pct:.0f}%)")

    # Non-linearity analysis
    print("\n" + "=" * 80)
    print("NON-LINEARITY ANALYSIS")
    print("=" * 80)
    analyze_nonlinearity(data, all_ratios)

    # Recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION: FINER-GRAINED RATIOS?")
    print("=" * 80)
    provide_recommendation(data, all_ratios)


def classify_shape(ratios, all_ratios):
    """Classify the shape of a ratio curve."""
    available = [(r, ratios[r]) for r in sorted(all_ratios) if r in ratios]
    if len(available) < 3:
        return "insufficient"

    values = [v for _, v in available]
    ratios_list = [r for r, _ in available]

    # Check monotonic increase
    diffs = [values[i+1] - values[i] for i in range(len(values)-1)]
    if all(d > 0 for d in diffs):
        return "↗ monotonic-up"
    if all(d < 0 for d in diffs):
        return "↘ monotonic-down"

    # Check for peak (inverted-U)
    peak_idx = values.index(max(values))
    if 0 < peak_idx < len(values) - 1:
        return f"∩ peak@{ratios_list[peak_idx]:.2f}"

    # Check for valley (U-shape)
    valley_idx = values.index(min(values))
    if 0 < valley_idx < len(values) - 1:
        return f"∪ valley@{ratios_list[valley_idx]:.2f}"

    # Check for plateau
    spread = max(values) - min(values)
    if spread < 1.0:
        return "— plateau"

    if peak_idx == 0:
        return "↘ best@0.00"
    if peak_idx == len(values) - 1:
        return "↗ best@baseline"

    return "~ irregular"


def analyze_nonlinearity(data, all_ratios):
    """Quantify non-linearity in the curves."""
    for test_type in ['Cityscapes', 'ACDC']:
        print(f"\n{test_type}:")
        deviations = []
        for key, ratios in data.items():
            if key[2] != test_type:
                continue
            # Check if we have enough points for interpolation
            available = sorted([(r, ratios[r]) for r in all_ratios if r in ratios])
            if len(available) < 4:
                continue

            # Linear interpolation between endpoints
            r_min, v_min = available[0]
            r_max, v_max = available[-1]
            slope = (v_max - v_min) / (r_max - r_min) if r_max != r_min else 0

            for r, v in available[1:-1]:  # Interior points
                expected = v_min + slope * (r - r_min)
                deviation = v - expected
                deviations.append((key[0], key[1], r, deviation))

        if deviations:
            avg_abs_dev = sum(abs(d[3]) for d in deviations) / len(deviations)
            max_dev = max(deviations, key=lambda x: abs(x[3]))
            print(f"  Mean |deviation from linear|: {avg_abs_dev:.2f} pp")
            print(f"  Max deviation: {max_dev[3]:+.2f} pp ({max_dev[0]}/{max_dev[1]} @ ratio={max_dev[2]:.2f})")
            print(f"  Interior points analyzed: {len(deviations)}")

            # Significant deviations (> 1 pp from linear)
            sig = [d for d in deviations if abs(d[3]) > 1.0]
            print(f"  Significant deviations (>1pp): {len(sig)}/{len(deviations)} ({100*len(sig)/len(deviations):.0f}%)")
            for s, m, r, d in sorted(sig, key=lambda x: -abs(x[3])):
                print(f"    {s}/{m} @ {r:.2f}: {d:+.2f} pp")


def provide_recommendation(data, all_ratios):
    """Provide recommendation on whether finer ratios are needed."""
    # Compute overall spread on ACDC (more informative)
    acdc_curves = {}
    for key, ratios in data.items():
        if key[2] != 'ACDC':
            continue
        if len(ratios) >= 3:
            acdc_curves[key] = ratios

    if not acdc_curves:
        print("Insufficient ACDC data for recommendation.")
        return

    spreads = []
    peaks = []
    for key, ratios in acdc_curves.items():
        spread = max(ratios.values()) - min(ratios.values())
        spreads.append(spread)
        best_ratio = max(ratios.items(), key=lambda x: x[1])[0]
        peaks.append(best_ratio)

    avg_spread = sum(spreads) / len(spreads)
    from collections import Counter
    peak_dist = Counter(peaks)

    print(f"\nACDC cross-domain analysis ({len(acdc_curves)} curves):")
    print(f"  Average spread: {avg_spread:.2f} pp")
    print(f"  Peak ratio distribution: {dict(peak_dist)}")

    # Key question: Is there a clear optimal ratio, or is the curve flat?
    if avg_spread < 2.0:
        print(f"\n  VERDICT: Spread is small ({avg_spread:.1f} pp). Finer ratios unlikely")
        print(f"  to reveal much. Current 5-point sampling (0, 0.25, 0.5, 0.75, 1.0) sufficient.")
        print(f"  The ratio effect is weak on Cityscapes — focus on S1 datasets instead.")
    elif avg_spread < 5.0:
        print(f"\n  VERDICT: Moderate spread ({avg_spread:.1f} pp). Finer ratios could help")
        print(f"  pinpoint optimal ratio if peaks cluster around an interior point.")
        if 0.25 in peak_dist or 0.75 in peak_dist:
            # Peaks at boundary of our sampling → finer sampling nearby could help
            print(f"  Peaks at 0.25/0.75 suggest exploring 0.125/0.375/0.625/0.875.")
        else:
            print(f"  But peaks at extremes suggest finer interior points won't help much.")
    else:
        print(f"\n  VERDICT: Large spread ({avg_spread:.1f} pp). Finer ratios recommended")
        print(f"  to accurately characterize the optimal mixing ratio.")

    # Also check: do we even need ratios at all if best is always 0.5?
    print(f"\n  Peak frequency:")
    for ratio, count in sorted(peak_dist.items()):
        pct = count / len(acdc_curves) * 100
        print(f"    ratio={ratio:.2f}: {count} curves ({pct:.0f}%)")


if __name__ == '__main__':
    data = collect_results()
    analyze_curves(data)
