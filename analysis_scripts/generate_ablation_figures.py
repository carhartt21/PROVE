#!/usr/bin/env python3
"""Generate figures for the Noise and Ratio Ablation analysis report."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json
from pathlib import Path

matplotlib.use('Agg')
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 150,
})

OUTDIR = Path('analysis_scripts/result_figures/ablation_figures')
OUTDIR.mkdir(parents=True, exist_ok=True)

ABLATION_DIR = Path('analysis_scripts/result_figures/ablation_exports')
WEIGHTS = Path('${AWARE_DATA_ROOT}/WEIGHTS')
WEIGHTS_CG = Path('${AWARE_DATA_ROOT}/WEIGHTS_CITYSCAPES_GEN')

MODEL_SHORT = {
    'deeplabv3plus_r50': 'DLv3+',
    'hrnet_hr48': 'HRNet',
    'mask2former_swin-b': 'M2F',
    'pspnet_r50': 'PSPNet',
    'segformer_mit-b3': 'SegFormer',
    'segnext_mscan-b': 'SegNeXt',
}
MODEL_COLORS = {
    'deeplabv3plus_r50': '#1f77b4',
    'hrnet_hr48': '#ff7f0e',
    'mask2former_swin-b': '#2ca02c',
    'pspnet_r50': '#d62728',
    'segformer_mit-b3': '#9467bd',
    'segnext_mscan-b': '#8c564b',
}
STRATEGY_COLORS = {
    'gen_TSIT': '#1f77b4',
    'gen_VisualCloze': '#ff7f0e',
    'gen_flux_kontext': '#2ca02c',
    'gen_step1x_v1p2': '#d62728',
}
STRATEGY_SHORT = {
    'gen_TSIT': 'TSIT',
    'gen_VisualCloze': 'VisualCloze',
    'gen_flux_kontext': 'Flux-Kontext',
    'gen_step1x_v1p2': 'Step1X',
}


def load_baselines():
    """Load Stage 1 baselines."""
    datasets = ['bdd10k', 'iddaw', 'mapillaryvistas', 'outside15k']
    models = list(MODEL_SHORT.keys())
    baseline = {}
    for ds in datasets:
        for model in models:
            wdir = WEIGHTS / 'baseline' / ds / model
            rfiles = sorted(wdir.glob('test_results_*/*/results.json')) if wdir.exists() else []
            if rfiles:
                with open(rfiles[-1]) as f:
                    data = json.load(f)
                baseline[(ds, model)] = data['overall']['mIoU']
    return baseline


def load_cg_references():
    """Load Cityscapes-Gen r=0.50 references and baselines."""
    strats = ['gen_TSIT', 'gen_VisualCloze', 'gen_flux_kontext', 'gen_step1x_v1p2']
    models = ['mask2former_swin-b', 'pspnet_r50', 'segformer_mit-b3', 'segnext_mscan-b']
    ref = {}
    for strat in strats + ['baseline']:
        for model in models:
            for suffix in [f'{model}_ratio0p50', model]:
                wdir = WEIGHTS_CG / strat / 'cityscapes' / suffix
                if not wdir.exists():
                    continue
                for td_name in sorted(wdir.glob('test_results_*')):
                    rfiles = sorted(td_name.glob('*/results.json'))
                    if rfiles:
                        with open(rfiles[-1]) as f:
                            data = json.load(f)
                        tt = 'acdc' if 'acdc' in td_name.name else 'detailed'
                        ref[(strat, model, tt)] = data['overall']['mIoU']
                break
    return ref


# ============================================================
# Figure 1: Noise vs Baseline Bar Chart
# ============================================================
def fig1_noise_vs_baseline():
    noise = pd.read_csv(ABLATION_DIR / 'noise_ablation.csv')
    baseline = load_baselines()

    models = ['hrnet_hr48', 'segformer_mit-b3', 'segnext_mscan-b', 'pspnet_r50', 'mask2former_swin-b']
    datasets = ['bdd10k', 'iddaw', 'mapillaryvistas', 'outside15k']

    noise_means = []
    bl_means = []
    labels = []
    for model in models:
        n0 = noise[(noise['model'] == model) & (noise['ratio'] == 0.0)]['mIoU'].mean()
        bls = [baseline[(ds, model)] for ds in datasets if (ds, model) in baseline]
        bl = np.mean(bls) if bls else np.nan
        noise_means.append(n0)
        bl_means.append(bl)
        labels.append(MODEL_SHORT[model])

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    w = 0.35
    bars_bl = ax.bar(x - w/2, bl_means, w, label='Baseline (real data)', color='#4c72b0', edgecolor='white')
    bars_noise = ax.bar(x + w/2, noise_means, w, label='Noise (100% random pixels)', color='#dd8452', edgecolor='white')

    # Add delta annotations
    for i, (bl, n) in enumerate(zip(bl_means, noise_means)):
        if not np.isnan(bl):
            delta = n - bl
            color = '#2ca02c' if delta > 0 else '#d62728'
            ax.annotate(f'{delta:+.1f}', xy=(x[i] + w/2, n + 0.3),
                       ha='center', va='bottom', fontsize=10, fontweight='bold', color=color)

    ax.set_ylabel('mIoU (%)')
    ax.set_title('Noise Ablation: Random Noise vs. Real-Data Baseline\n(Stage 1, mean across 4 datasets)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper right')
    ax.set_ylim(0, max(max(noise_means), max(bl_means)) * 1.12)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(OUTDIR / 'fig1_noise_vs_baseline.png')
    plt.close(fig)
    print(f"Saved fig1_noise_vs_baseline.png")


# ============================================================
# Figure 2: Noise Ratio Effect (r=0.0 vs r=0.5)
# ============================================================
def fig2_noise_ratio_effect():
    noise = pd.read_csv(ABLATION_DIR / 'noise_ablation.csv')
    models = sorted(noise['model'].unique())

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    w = 0.35

    r0_vals = [noise[(noise['model'] == m) & (noise['ratio'] == 0.0)]['mIoU'].mean() for m in models]
    r5_vals = [noise[(noise['model'] == m) & (noise['ratio'] == 0.5)]['mIoU'].mean() for m in models]

    ax.bar(x - w/2, r0_vals, w, label='r=0.0 (100% noise)', color='#e74c3c', edgecolor='white')
    ax.bar(x + w/2, r5_vals, w, label='r=0.5 (50% noise + 50% real)', color='#3498db', edgecolor='white')

    for i in range(len(models)):
        delta = r5_vals[i] - r0_vals[i]
        y = max(r0_vals[i], r5_vals[i]) + 0.3
        ax.annotate(f'Δ={delta:+.2f}', xy=(x[i], y), ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('mIoU (%)')
    ax.set_title('Effect of Noise Ratio: r=0.0 (all noise) vs r=0.5 (half noise)\n(Mean across 4 datasets)')
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_SHORT[m] for m in models])
    ax.legend()
    ax.set_ylim(25, max(max(r0_vals), max(r5_vals)) * 1.08)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(OUTDIR / 'fig2_noise_ratio_effect.png')
    plt.close(fig)
    print(f"Saved fig2_noise_ratio_effect.png")


# ============================================================
# Figure 3: Noise Per-Domain Heatmap
# ============================================================
def fig3_noise_per_domain():
    noise = pd.read_csv(ABLATION_DIR / 'noise_ablation.csv')
    domain_cols = [c for c in noise.columns if c.startswith('mIoU_')]
    domains = [c.replace('mIoU_', '') for c in domain_cols]

    # Mean across all models, ratio=0.0
    r0 = noise[noise['ratio'] == 0.0]

    # Build heatmap: dataset × domain
    datasets = sorted(r0['dataset'].unique())
    matrix = np.zeros((len(datasets), len(domains)))
    for i, ds in enumerate(datasets):
        for j, col in enumerate(domain_cols):
            vals = r0[r0['dataset'] == ds][col].dropna()
            matrix[i, j] = vals.mean() if len(vals) else np.nan

    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=20, vmax=55)

    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels([d.replace('_', '\n') for d in domains], rotation=0)
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets)

    for i in range(len(datasets)):
        for j in range(len(domains)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                       fontsize=9, color='black' if 25 < val < 50 else 'white')

    plt.colorbar(im, ax=ax, label='mIoU (%)', shrink=0.8)
    ax.set_title('Noise Ablation: Per-Domain mIoU by Dataset (r=0.0, mean across 6 models)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(OUTDIR / 'fig3_noise_per_domain_heatmap.png')
    plt.close(fig)
    print(f"Saved fig3_noise_per_domain_heatmap.png")


# ============================================================
# Figure 4: CS-Ratio Curves (ACDC)
# ============================================================
def fig4_cs_ratio_acdc():
    cs_ratio = pd.read_csv(ABLATION_DIR / 'cs_ratio_ablation.csv')
    cg_ref = load_cg_references()

    acdc = cs_ratio[cs_ratio['test_type'] == 'acdc']
    strategies = sorted(acdc['strategy'].unique())
    models = sorted(acdc['model'].unique())
    ratios_abl = [0.0, 0.25, 0.75]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, model in enumerate(models):
        ax = axes[idx]
        for strat in strategies:
            vals = []
            for r in [0.0, 0.25, 0.50, 0.75]:
                if r == 0.50:
                    v = cg_ref.get((strat, model, 'acdc'), np.nan)
                else:
                    row = acdc[(acdc['strategy'] == strat) & (acdc['model'] == model) & (acdc['ratio'] == r)]
                    v = row['mIoU'].values[0] if len(row) else np.nan
                vals.append(v)

            ax.plot([0.0, 0.25, 0.50, 0.75], vals, 'o-',
                   color=STRATEGY_COLORS[strat], label=STRATEGY_SHORT[strat],
                   markersize=6, linewidth=2)

        # Baseline
        bl = cg_ref.get(('baseline', model, 'acdc'), None)
        if bl:
            ax.axhline(bl, color='gray', linestyle='--', alpha=0.7, label=f'Baseline ({bl:.1f})')

        ax.set_title(MODEL_SHORT[model], fontweight='bold')
        ax.set_xlabel('Real/Gen Ratio')
        ax.set_ylabel('mIoU (%)')
        ax.set_xticks([0.0, 0.25, 0.50, 0.75])
        ax.set_xticklabels(['0.00\n(all gen)', '0.25', '0.50', '0.75\n(mostly real)'])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='best')

    fig.suptitle('CS-Ratio Ablation: ACDC (Cross-Domain) mIoU by Ratio', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUTDIR / 'fig4_cs_ratio_acdc_curves.png')
    plt.close(fig)
    print(f"Saved fig4_cs_ratio_acdc_curves.png")


# ============================================================
# Figure 5: CS-Ratio Curves (Cityscapes Detailed)
# ============================================================
def fig5_cs_ratio_detailed():
    cs_ratio = pd.read_csv(ABLATION_DIR / 'cs_ratio_ablation.csv')
    cg_ref = load_cg_references()

    det = cs_ratio[cs_ratio['test_type'] == 'detailed']
    strategies = sorted(det['strategy'].unique())
    models = sorted(det['model'].unique())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, model in enumerate(models):
        ax = axes[idx]
        for strat in strategies:
            vals = []
            for r in [0.0, 0.25, 0.50, 0.75]:
                if r == 0.50:
                    v = cg_ref.get((strat, model, 'detailed'), np.nan)
                else:
                    row = det[(det['strategy'] == strat) & (det['model'] == model) & (det['ratio'] == r)]
                    v = row['mIoU'].values[0] if len(row) else np.nan
                vals.append(v)

            ax.plot([0.0, 0.25, 0.50, 0.75], vals, 'o-',
                   color=STRATEGY_COLORS[strat], label=STRATEGY_SHORT[strat],
                   markersize=6, linewidth=2)

        bl = cg_ref.get(('baseline', model, 'detailed'), None)
        if bl:
            ax.axhline(bl, color='gray', linestyle='--', alpha=0.7, label=f'Baseline ({bl:.1f})')

        ax.set_title(MODEL_SHORT[model], fontweight='bold')
        ax.set_xlabel('Real/Gen Ratio')
        ax.set_ylabel('mIoU (%)')
        ax.set_xticks([0.0, 0.25, 0.50, 0.75])
        ax.set_xticklabels(['0.00\n(all gen)', '0.25', '0.50', '0.75\n(mostly real)'])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='best')

    fig.suptitle('CS-Ratio Ablation: Cityscapes Detailed (In-Domain) mIoU by Ratio', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUTDIR / 'fig5_cs_ratio_detailed_curves.png')
    plt.close(fig)
    print(f"Saved fig5_cs_ratio_detailed_curves.png")


# ============================================================
# Figure 6: In-Domain vs Cross-Domain Trade-off
# ============================================================
def fig6_indomain_vs_crossdomain():
    cs_ratio = pd.read_csv(ABLATION_DIR / 'cs_ratio_ablation.csv')
    cg_ref = load_cg_references()

    models = ['mask2former_swin-b', 'pspnet_r50', 'segformer_mit-b3', 'segnext_mscan-b']
    strategies = ['gen_TSIT', 'gen_VisualCloze', 'gen_flux_kontext', 'gen_step1x_v1p2']
    ratios = [0.0, 0.25, 0.50, 0.75]

    # Compute mean across models×strategies for each ratio (excluding VisualCloze outlier)
    acdc_means = []
    det_means = []
    for r in ratios:
        acdc_vals = []
        det_vals = []
        for strat in strategies:
            for model in models:
                # Skip VisualCloze × mask2former at r=0.25
                if strat == 'gen_VisualCloze' and model == 'mask2former_swin-b' and r == 0.25:
                    continue
                if r == 0.50:
                    va = cg_ref.get((strat, model, 'acdc'), None)
                    vd = cg_ref.get((strat, model, 'detailed'), None)
                else:
                    ra = cs_ratio[(cs_ratio['strategy'] == strat) & (cs_ratio['model'] == model) &
                                  (cs_ratio['test_type'] == 'acdc') & (cs_ratio['ratio'] == r)]
                    rd = cs_ratio[(cs_ratio['strategy'] == strat) & (cs_ratio['model'] == model) &
                                  (cs_ratio['test_type'] == 'detailed') & (cs_ratio['ratio'] == r)]
                    va = ra['mIoU'].values[0] if len(ra) else None
                    vd = rd['mIoU'].values[0] if len(rd) else None
                if va is not None:
                    acdc_vals.append(va)
                if vd is not None:
                    det_vals.append(vd)
        acdc_means.append(np.mean(acdc_vals))
        det_means.append(np.mean(det_vals))

    fig, ax1 = plt.subplots(figsize=(9, 6))

    color_acdc = '#e74c3c'
    color_det = '#3498db'

    ax1.plot(ratios, acdc_means, 'o-', color=color_acdc, linewidth=2.5, markersize=10, label='ACDC (cross-domain)')
    ax1.plot(ratios, det_means, 's-', color=color_det, linewidth=2.5, markersize=10, label='Cityscapes (in-domain)')

    # Annotate values
    for i, r in enumerate(ratios):
        ax1.annotate(f'{acdc_means[i]:.2f}', xy=(r, acdc_means[i]),
                    textcoords='offset points', xytext=(0, -18), ha='center', fontsize=9, color=color_acdc)
        ax1.annotate(f'{det_means[i]:.2f}', xy=(r, det_means[i]),
                    textcoords='offset points', xytext=(0, 12), ha='center', fontsize=9, color=color_det)

    # Add arrows showing optimal direction
    ax1.annotate('← More generated\n   (better cross-domain)', xy=(0.05, acdc_means[0] + 0.3),
                fontsize=8, color=color_acdc, alpha=0.7)
    ax1.annotate('More real →\n(better in-domain)', xy=(0.55, det_means[3] + 0.3),
                fontsize=8, color=color_det, alpha=0.7)

    ax1.set_xlabel('Real/Generated Ratio', fontsize=12)
    ax1.set_ylabel('mIoU (%)', fontsize=12)
    ax1.set_title('In-Domain vs. Cross-Domain: Ratio Trade-off\n(CS-Ratio Ablation, mean across 4 strategies × 4 models)', fontsize=13)
    ax1.set_xticks(ratios)
    ax1.set_xticklabels(['0.00\n(all gen)', '0.25', '0.50', '0.75\n(mostly real)'])
    ax1.legend(fontsize=11, loc='center left')
    ax1.grid(alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    fig.savefig(OUTDIR / 'fig6_indomain_vs_crossdomain_tradeoff.png')
    plt.close(fig)
    print(f"Saved fig6_indomain_vs_crossdomain_tradeoff.png")


# ============================================================
# Figure 7: Pretrained vs From-Scratch Noise Comparison
# ============================================================
def fig7_pretrained_vs_fromscratch():
    noise = pd.read_csv(ABLATION_DIR / 'noise_ablation.csv')
    from_scratch = pd.read_csv(ABLATION_DIR / 'from_scratch_ablation.csv')
    baseline = load_baselines()

    # Data for pspnet/bdd10k
    pretrained_noise_r0 = noise[(noise['model'] == 'pspnet_r50') & (noise['dataset'] == 'bdd10k') & (noise['ratio'] == 0.0)]['mIoU'].values[0]
    pretrained_bl = baseline.get(('bdd10k', 'pspnet_r50'), np.nan)

    fs_noise_row = from_scratch[(from_scratch['strategy'] == 'gen_random_noise') & (from_scratch['model'] == 'pspnet_r50') & (from_scratch['dataset'] == 'bdd10k')]
    fs_noise = fs_noise_row['mIoU'].values[0] if len(fs_noise_row) else np.nan
    fs_bl_row = from_scratch[(from_scratch['strategy'] == 'baseline') & (from_scratch['model'] == 'pspnet_r50') & (from_scratch['dataset'] == 'bdd10k')]
    fs_bl = fs_bl_row['mIoU'].values[0] if len(fs_bl_row) else np.nan

    fig, ax = plt.subplots(figsize=(8, 5))

    categories = ['Pretrained\nBackbone', 'From\nScratch']
    x = np.arange(len(categories))
    w = 0.3

    baseline_vals = [pretrained_bl, fs_bl]
    noise_vals = [pretrained_noise_r0, fs_noise]

    bars_bl = ax.bar(x - w/2, baseline_vals, w, label='Baseline (real data)', color='#4c72b0', edgecolor='white')
    bars_noise = ax.bar(x + w/2, noise_vals, w, label='Noise (random pixels)', color='#dd8452', edgecolor='white')

    # Annotate values and deltas
    for i in range(len(categories)):
        ax.text(x[i] - w/2, baseline_vals[i] + 0.3, f'{baseline_vals[i]:.1f}', ha='center', va='bottom', fontsize=10)
        ax.text(x[i] + w/2, noise_vals[i] + 0.3, f'{noise_vals[i]:.1f}', ha='center', va='bottom', fontsize=10)
        delta = noise_vals[i] - baseline_vals[i]
        color = '#2ca02c' if delta > 0 else '#d62728'
        mid_y = max(baseline_vals[i], noise_vals[i]) + 2
        ax.annotate(f'Δ = {delta:+.2f}', xy=(x[i], mid_y), ha='center', fontsize=11, fontweight='bold', color=color)

    ax.set_ylabel('mIoU (%)')
    ax.set_title('Noise Effect: Pretrained vs. From-Scratch\n(PSPNet × BDD10k)', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc='upper right')
    ax.set_ylim(0, max(max(baseline_vals), max(noise_vals)) * 1.15)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(OUTDIR / 'fig7_pretrained_vs_fromscratch_noise.png')
    plt.close(fig)
    print(f"Saved fig7_pretrained_vs_fromscratch_noise.png")


# ============================================================
# Figure 8: S1-Ratio Ablation
# ============================================================
def fig8_s1_ratio():
    s1 = pd.read_csv(ABLATION_DIR / 's1_ratio_ablation.csv')

    strategies = sorted(s1['strategy'].unique())
    combos = s1.groupby(['dataset', 'model']).size().reset_index()
    combo_list = list(zip(combos['dataset'], combos['model']))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for idx, (ds, model) in enumerate(combo_list):
        ax = axes[idx]
        for strat in strategies:
            subset = s1[(s1['dataset'] == ds) & (s1['model'] == model) & (s1['strategy'] == strat)]
            subset = subset.sort_values('ratio')
            ax.plot(subset['ratio'], subset['mIoU'], 'o-',
                   color=STRATEGY_COLORS[strat], label=STRATEGY_SHORT[strat],
                   markersize=8, linewidth=2)

        ax.set_title(f'{ds} × {MODEL_SHORT.get(model, model)}', fontweight='bold')
        ax.set_xlabel('Real/Gen Ratio')
        ax.set_ylabel('mIoU (%)')
        ax.set_xticks([0.0, 0.25, 0.75])
        ax.set_xticklabels(['0.00\n(all gen)', '0.25', '0.75\n(mostly real)'])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle('S1-Ratio Ablation: Stage 1 Training (BDD10k & IDD-AW)', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUTDIR / 'fig8_s1_ratio_curves.png')
    plt.close(fig)
    print(f"Saved fig8_s1_ratio_curves.png")


# ============================================================
# Figure 9: Model Sensitivity to Ratio (ACDC)
# ============================================================
def fig9_model_sensitivity():
    cs_ratio = pd.read_csv(ABLATION_DIR / 'cs_ratio_ablation.csv')
    cg_ref = load_cg_references()

    models = ['mask2former_swin-b', 'pspnet_r50', 'segformer_mit-b3', 'segnext_mscan-b']
    strategies = ['gen_TSIT', 'gen_VisualCloze', 'gen_flux_kontext', 'gen_step1x_v1p2']
    ratios = [0.0, 0.25, 0.50, 0.75]

    fig, ax = plt.subplots(figsize=(10, 6))

    for model in models:
        means = []
        stds = []
        for r in ratios:
            vals = []
            for strat in strategies:
                # Skip outlier
                if strat == 'gen_VisualCloze' and model == 'mask2former_swin-b' and r == 0.25:
                    continue
                if r == 0.50:
                    v = cg_ref.get((strat, model, 'acdc'), None)
                else:
                    row = cs_ratio[(cs_ratio['strategy'] == strat) & (cs_ratio['model'] == model) &
                                  (cs_ratio['test_type'] == 'acdc') & (cs_ratio['ratio'] == r)]
                    v = row['mIoU'].values[0] if len(row) else None
                if v is not None:
                    vals.append(v)
            means.append(np.mean(vals))
            stds.append(np.std(vals))

        ax.errorbar(ratios, means, yerr=stds, fmt='o-',
                   color=MODEL_COLORS[model], label=MODEL_SHORT[model],
                   markersize=8, linewidth=2, capsize=4)

    ax.set_xlabel('Real/Generated Ratio')
    ax.set_ylabel('mIoU (%)')
    ax.set_title('Model Sensitivity to Real/Gen Ratio (ACDC)\n(Mean ± std across 4 strategies)', fontsize=13)
    ax.set_xticks(ratios)
    ax.set_xticklabels(['0.00\n(all gen)', '0.25', '0.50', '0.75\n(mostly real)'])
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(OUTDIR / 'fig9_model_sensitivity_acdc.png')
    plt.close(fig)
    print(f"Saved fig9_model_sensitivity_acdc.png")


# ============================================================
# Run all
# ============================================================
if __name__ == '__main__':
    print("Generating ablation figures...")
    fig1_noise_vs_baseline()
    fig2_noise_ratio_effect()
    fig3_noise_per_domain()
    fig4_cs_ratio_acdc()
    fig5_cs_ratio_detailed()
    fig6_indomain_vs_crossdomain()
    fig7_pretrained_vs_fromscratch()
    fig8_s1_ratio()
    fig9_model_sensitivity()
    print(f"\nAll figures saved to {OUTDIR}/")
