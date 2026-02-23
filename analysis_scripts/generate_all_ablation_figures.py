#!/usr/bin/env python3
"""
Generate comprehensive figures from ALL ablation CSV files.

CSVs covered:
  1. ratio_ablation_synthesized.csv  — 21 strategies × 4 datasets × 4 models × 9 ratios
  2. noise_ablation.csv              — noise vs baseline (6 models × 4 datasets × 2 ratios)
  3. cs_ratio_ablation.csv           — Cityscapes ratio ablation (4 strategies × 4 models × 3 ratios × 2 test types)
  4. s1_ratio_ablation.csv           — Stage 1 ratio ablation (2 strategies × 2 models × 2 datasets × 3 ratios)
  5. loss_ablation.csv               — Loss function comparison
  6. combination_ablation.csv        — Strategy combination ablation
  7. from_scratch_ablation.csv       — Pretrained vs from-scratch
  8. extended_training_ablation.csv  — Training duration study
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from pathlib import Path
import json

matplotlib.use('Agg')

# ---- Global style ----
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
})

ABLATION_DIR = Path('analysis_scripts/result_figures/ablation_exports')
OUTDIR = Path('analysis_scripts/result_figures/ablation_figures')
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---- Name mappings ----
MODEL_SHORT = {
    'deeplabv3plus_r50': 'DLv3+',
    'hrnet_hr48': 'HRNet',
    'mask2former_swin-b': 'M2F',
    'pspnet_r50': 'PSPNet',
    'segformer_mit-b3': 'SegFormer',
    'segnext_mscan-b': 'SegNeXt',
}

DATASET_SHORT = {
    'bdd10k': 'BDD10k',
    'iddaw': 'IDD-AW',
    'mapillaryvistas': 'MV',
    'outside15k': 'O15k',
    'cityscapes': 'CS',
}

STRATEGY_SHORT = {
    'gen_Attribute_Hallucination': 'AttrHalluc',
    'gen_CNetSeg': 'CNetSeg',
    'gen_CUT': 'CUT',
    'gen_IP2P': 'IP2P',
    'gen_Img2Img': 'Img2Img',
    'gen_LANIT': 'LANIT',
    'gen_Qwen_Image_Edit': 'Qwen',
    'gen_SUSTechGAN': 'SUSTech',
    'gen_TSIT': 'TSIT',
    'gen_UniControl': 'UniCtrl',
    'gen_VisualCloze': 'VisCloze',
    'gen_Weather_Effect_Generator': 'WxEffect',
    'gen_albumentations_weather': 'AlbWx',
    'gen_augmenters': 'Augm',
    'gen_automold': 'AutoMold',
    'gen_cycleGAN': 'CycleGAN',
    'gen_cyclediffusion': 'CycleDiff',
    'gen_flux_kontext': 'Flux',
    'gen_stargan_v2': 'StarV2',
    'gen_step1x_new': 'Step1X-n',
    'gen_step1x_v1p2': 'Step1X',
    'baseline': 'Baseline',
}

# Strategy families
STRATEGY_FAMILIES = {
    'GAN-based': ['gen_CUT', 'gen_cycleGAN', 'gen_TSIT', 'gen_LANIT', 'gen_stargan_v2', 'gen_SUSTechGAN', 'gen_CNetSeg'],
    'Diffusion-based': ['gen_IP2P', 'gen_Img2Img', 'gen_cyclediffusion', 'gen_flux_kontext', 'gen_step1x_new', 'gen_step1x_v1p2', 'gen_VisualCloze', 'gen_Qwen_Image_Edit', 'gen_Attribute_Hallucination', 'gen_UniControl'],
    'Classical': ['gen_albumentations_weather', 'gen_augmenters', 'gen_automold', 'gen_Weather_Effect_Generator'],
}

# Colors for strategies
FAMILY_COLORS = {
    'GAN-based': '#e74c3c',
    'Diffusion-based': '#3498db',
    'Classical': '#2ecc71',
}

MODEL_COLORS = {
    'pspnet_r50': '#d62728',
    'segformer_mit-b3': '#9467bd',
    'segnext_mscan-b': '#8c564b',
    'mask2former_swin-b': '#2ca02c',
}

DATASET_COLORS = {
    'bdd10k': '#1f77b4',
    'iddaw': '#ff7f0e',
    'mapillaryvistas': '#2ca02c',
    'outside15k': '#d62728',
}


def get_family(strategy):
    for fam, strats in STRATEGY_FAMILIES.items():
        if strategy in strats:
            return fam
    return 'Unknown'


# ============================================================
# SYNTHESIZED RATIO ABLATION FIGURES
# ============================================================

def fig_ratio_overview_curves():
    """Fig R1: Ratio curves averaged across all strategies, faceted by dataset."""
    df = pd.read_csv(ABLATION_DIR / 'ratio_ablation_synthesized.csv')
    models = ['pspnet_r50', 'segformer_mit-b3', 'segnext_mscan-b', 'mask2former_swin-b']
    datasets = ['bdd10k', 'iddaw', 'mapillaryvistas', 'outside15k']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        for model in models:
            sub = df[(df['dataset'] == ds) & (df['model'] == model)]
            # Average across all strategies for each ratio
            grouped = sub.groupby('ratio')['mIoU'].agg(['mean', 'std']).reset_index()
            ax.plot(grouped['ratio'], grouped['mean'], 'o-',
                    color=MODEL_COLORS[model], label=MODEL_SHORT[model],
                    markersize=6, linewidth=2)
            ax.fill_between(grouped['ratio'],
                            grouped['mean'] - grouped['std'],
                            grouped['mean'] + grouped['std'],
                            alpha=0.15, color=MODEL_COLORS[model])

        ax.set_title(DATASET_SHORT[ds], fontweight='bold', fontsize=13)
        ax.set_xlabel('Real Image Ratio')
        ax.set_ylabel('mIoU (%)')
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(['0.0\n(all gen)', '0.25', '0.50', '0.75', '1.0\n(baseline)'])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc='best')

    fig.suptitle('Ratio Ablation: mIoU vs. Real Image Ratio\n(Mean ± Std across 21 strategies)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUTDIR / 'figR1_ratio_overview_by_dataset.png', bbox_inches='tight')
    plt.close(fig)
    print("Saved figR1_ratio_overview_by_dataset.png")


def fig_ratio_by_family():
    """Fig R2: Ratio curves by strategy family (GAN vs Diffusion vs Classical)."""
    df = pd.read_csv(ABLATION_DIR / 'ratio_ablation_synthesized.csv')
    df['family'] = df['strategy'].apply(get_family)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    datasets = ['bdd10k', 'iddaw', 'mapillaryvistas', 'outside15k']

    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        for fam in ['GAN-based', 'Diffusion-based', 'Classical']:
            sub = df[(df['dataset'] == ds) & (df['family'] == fam)]
            grouped = sub.groupby('ratio')['mIoU'].agg(['mean', 'std']).reset_index()
            ax.plot(grouped['ratio'], grouped['mean'], 'o-',
                    color=FAMILY_COLORS[fam], label=fam,
                    markersize=5, linewidth=2)
            ax.fill_between(grouped['ratio'],
                            grouped['mean'] - grouped['std'],
                            grouped['mean'] + grouped['std'],
                            alpha=0.12, color=FAMILY_COLORS[fam])

        ax.set_title(DATASET_SHORT[ds], fontweight='bold')
        ax.set_xlabel('Real Image Ratio')
        if idx == 0:
            ax.set_ylabel('mIoU (%)')
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='best')

    fig.suptitle('Ratio Ablation by Strategy Family\n(Mean ± Std across models)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUTDIR / 'figR2_ratio_by_family.png', bbox_inches='tight')
    plt.close(fig)
    print("Saved figR2_ratio_by_family.png")


def fig_ratio_heatmap():
    """Fig R3: Heatmap of mIoU delta from baseline at r=0.50 per strategy × dataset."""
    df = pd.read_csv(ABLATION_DIR / 'ratio_ablation_synthesized.csv')

    # Get r=0.50 and r=1.0 (baseline) for each strategy/dataset
    strategies = sorted(df['strategy'].unique())
    datasets = ['bdd10k', 'iddaw', 'mapillaryvistas', 'outside15k']

    delta_matrix = np.zeros((len(strategies), len(datasets)))
    for i, strat in enumerate(strategies):
        for j, ds in enumerate(datasets):
            sub = df[(df['strategy'] == strat) & (df['dataset'] == ds)]
            r050 = sub[sub['ratio'] == 0.5]['mIoU'].mean()
            r100 = sub[sub['ratio'] == 1.0]['mIoU'].mean()
            delta_matrix[i, j] = r050 - r100

    # Sort strategies by average delta (descending)
    avg_deltas = delta_matrix.mean(axis=1)
    sort_idx = np.argsort(-avg_deltas)
    delta_matrix = delta_matrix[sort_idx]
    strategies = [strategies[i] for i in sort_idx]

    fig, ax = plt.subplots(figsize=(10, 12))
    vmax = max(abs(delta_matrix.min()), abs(delta_matrix.max()))
    im = ax.imshow(delta_matrix, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([DATASET_SHORT[ds] for ds in datasets], fontsize=11)
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels([STRATEGY_SHORT.get(s, s) for s in strategies], fontsize=10)

    for i in range(len(strategies)):
        for j in range(len(datasets)):
            val = delta_matrix[i, j]
            color = 'white' if abs(val) > vmax * 0.6 else 'black'
            ax.text(j, i, f'{val:+.1f}', ha='center', va='center', fontsize=9, color=color)

    plt.colorbar(im, ax=ax, label='Δ mIoU from Baseline (pp)', shrink=0.6, pad=0.02)
    ax.set_title('Augmentation Effect at r=0.50\n(mIoU of strategy − baseline, averaged across 4 models)',
                 fontsize=13, fontweight='bold')

    fig.tight_layout()
    fig.savefig(OUTDIR / 'figR3_ratio_heatmap_delta.png', bbox_inches='tight')
    plt.close(fig)
    print("Saved figR3_ratio_heatmap_delta.png")


def fig_ratio_sensitivity():
    """Fig R4: How sensitive is each strategy to ratio changes? (spread metric)."""
    df = pd.read_csv(ABLATION_DIR / 'ratio_ablation_synthesized.csv')

    strategies = sorted(df['strategy'].unique())
    datasets = ['bdd10k', 'iddaw', 'mapillaryvistas', 'outside15k']

    # Compute spread = max(mIoU) - min(mIoU) across ratios, averaged across datasets×models
    spreads = []
    labels = []
    families = []
    for strat in strategies:
        sub = df[df['strategy'] == strat]
        combo_spreads = []
        for ds in datasets:
            for model in df['model'].unique():
                vals = sub[(sub['dataset'] == ds) & (sub['model'] == model)]['mIoU']
                if len(vals) > 0:
                    combo_spreads.append(vals.max() - vals.min())
        spreads.append(np.mean(combo_spreads))
        labels.append(STRATEGY_SHORT.get(strat, strat))
        families.append(get_family(strat))

    # Sort by spread
    sort_idx = np.argsort(spreads)[::-1]
    spreads = [spreads[i] for i in sort_idx]
    labels = [labels[i] for i in sort_idx]
    families = [families[i] for i in sort_idx]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [FAMILY_COLORS.get(f, '#888888') for f in families]
    bars = ax.barh(range(len(labels)), spreads, color=colors, edgecolor='white', height=0.7)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('mIoU Spread Across Ratios (pp)', fontsize=12)
    ax.set_title('Ratio Sensitivity per Strategy\n(Max−Min mIoU across r=0.0 to r=1.0, mean over datasets×models)',
                 fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend for families
    legend_elements = [Line2D([0], [0], marker='s', color='w', markerfacecolor=FAMILY_COLORS[f],
                              markersize=12, label=f) for f in FAMILY_COLORS]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    # Annotate values
    for i, (s, l) in enumerate(zip(spreads, labels)):
        ax.text(s + 0.05, i, f'{s:.1f}', va='center', fontsize=9)

    fig.tight_layout()
    fig.savefig(OUTDIR / 'figR4_ratio_sensitivity_per_strategy.png', bbox_inches='tight')
    plt.close(fig)
    print("Saved figR4_ratio_sensitivity_per_strategy.png")


def fig_ratio_best_strategies():
    """Fig R5: Top-5 and Bottom-5 strategies at r=0.50 per dataset."""
    df = pd.read_csv(ABLATION_DIR / 'ratio_ablation_synthesized.csv')
    datasets = ['bdd10k', 'iddaw', 'mapillaryvistas', 'outside15k']

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        sub = df[(df['dataset'] == ds) & (df['ratio'].isin([0.5, 1.0]))]
        pivot = sub.pivot_table(values='mIoU', index='strategy', columns='ratio', aggfunc='mean')

        if 1.0 in pivot.columns and 0.5 in pivot.columns:
            pivot['delta'] = pivot[0.5] - pivot[1.0]
            pivot = pivot.sort_values('delta', ascending=False)

            # Top 5 and bottom 5
            top5 = pivot.head(5)
            bot5 = pivot.tail(5)
            combined = pd.concat([top5, bot5])

            labels_s = [STRATEGY_SHORT.get(s, s) for s in combined.index]
            deltas = combined['delta'].values
            colors = ['#2ecc71' if d > 0 else '#e74c3c' for d in deltas]

            bars = ax.barh(range(len(labels_s)), deltas, color=colors, edgecolor='white', height=0.7)
            ax.set_yticks(range(len(labels_s)))
            ax.set_yticklabels(labels_s, fontsize=10)
            ax.axvline(0, color='black', linewidth=0.8)

            for i, d in enumerate(deltas):
                ax.text(d + (0.1 if d >= 0 else -0.1), i, f'{d:+.1f}',
                        va='center', ha='left' if d >= 0 else 'right', fontsize=9)

        ax.set_title(DATASET_SHORT[ds], fontweight='bold', fontsize=13)
        ax.set_xlabel('Δ mIoU from Baseline (pp)')
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('Top-5 & Bottom-5 Strategies at r=0.50\n(Δ mIoU from baseline, mean across 4 models)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUTDIR / 'figR5_ratio_top_bottom_strategies.png', bbox_inches='tight')
    plt.close(fig)
    print("Saved figR5_ratio_top_bottom_strategies.png")


def fig_ratio_individual_strategies():
    """Fig R6: Individual ratio curves for selected strategies (3×3 grid)."""
    df = pd.read_csv(ABLATION_DIR / 'ratio_ablation_synthesized.csv')

    # Select 9 representative strategies
    selected = [
        'gen_TSIT', 'gen_cycleGAN', 'gen_CUT',
        'gen_flux_kontext', 'gen_step1x_new', 'gen_VisualCloze',
        'gen_albumentations_weather', 'gen_automold', 'gen_IP2P',
    ]

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    axes = axes.flatten()

    for idx, strat in enumerate(selected):
        ax = axes[idx]
        sub = df[df['strategy'] == strat]

        for ds in ['bdd10k', 'iddaw', 'mapillaryvistas', 'outside15k']:
            ds_sub = sub[sub['dataset'] == ds]
            grouped = ds_sub.groupby('ratio')['mIoU'].mean().reset_index()
            ax.plot(grouped['ratio'], grouped['mIoU'], 'o-',
                    color=DATASET_COLORS[ds], label=DATASET_SHORT[ds],
                    markersize=5, linewidth=1.8)

            # Mark real vs synthesized
            real_rows = ds_sub[ds_sub['source'] == 'real'].groupby('ratio')['mIoU'].mean().reset_index()
            if len(real_rows) > 0:
                ax.scatter(real_rows['ratio'], real_rows['mIoU'],
                           color=DATASET_COLORS[ds], s=50, zorder=5, edgecolors='black', linewidth=0.5)

        ax.set_title(STRATEGY_SHORT.get(strat, strat), fontweight='bold', fontsize=12)
        ax.set_xlabel('Real Image Ratio')
        if idx % 3 == 0:
            ax.set_ylabel('mIoU (%)')
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8, loc='best')

    fig.suptitle('Ratio Curves for Selected Strategies\n(Mean across 4 models; ● = real data points)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUTDIR / 'figR6_ratio_individual_strategies.png', bbox_inches='tight')
    plt.close(fig)
    print("Saved figR6_ratio_individual_strategies.png")


def fig_ratio_model_comparison():
    """Fig R7: Per-model ratio curves averaged across all strategies."""
    df = pd.read_csv(ABLATION_DIR / 'ratio_ablation_synthesized.csv')
    models = ['pspnet_r50', 'segformer_mit-b3', 'segnext_mscan-b', 'mask2former_swin-b']

    fig, ax = plt.subplots(figsize=(10, 6))

    for model in models:
        sub = df[df['model'] == model]
        grouped = sub.groupby('ratio')['mIoU'].agg(['mean', 'std']).reset_index()
        ax.plot(grouped['ratio'], grouped['mean'], 'o-',
                color=MODEL_COLORS[model], label=MODEL_SHORT[model],
                markersize=7, linewidth=2.5)
        ax.fill_between(grouped['ratio'],
                        grouped['mean'] - grouped['std'],
                        grouped['mean'] + grouped['std'],
                        alpha=0.1, color=MODEL_COLORS[model])

    ax.set_xlabel('Real Image Ratio', fontsize=12)
    ax.set_ylabel('mIoU (%)', fontsize=12)
    ax.set_title('Per-Model Ratio Response\n(Mean ± Std across 21 strategies × 4 datasets)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks([0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0])
    ax.set_xticklabels(['0.0', '', '0.25', '', '0.50', '', '0.75', '', '1.0'])
    ax.legend(fontsize=11, loc='best')
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUTDIR / 'figR7_ratio_per_model.png', bbox_inches='tight')
    plt.close(fig)
    print("Saved figR7_ratio_per_model.png")


# ============================================================
# LOSS ABLATION FIGURES
# ============================================================

def fig_loss_ablation():
    """Fig L1: Loss function comparison bar chart."""
    df = pd.read_csv(ABLATION_DIR / 'loss_ablation.csv')
    if df.empty:
        print("Skipping loss ablation figure (no data)")
        return

    # Extract loss type from model name (e.g., deeplabv3plus_r50_aux-boundary)
    df['loss_variant'] = df['model'].str.extract(r'_(aux-\w+|loss-\w+)', expand=False)
    df['base_model'] = df['model'].str.replace(r'_(aux-\w+|loss-\w+)', '', regex=True)

    models = sorted(df['base_model'].unique())
    loss_variants = sorted(df['loss_variant'].dropna().unique())
    datasets = sorted(df['dataset'].unique())

    if not loss_variants:
        print("Skipping loss ablation figure (no loss variants found)")
        return

    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]

    loss_colors = plt.cm.Set2(np.linspace(0, 1, len(loss_variants) + 1))

    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        sub = df[df['dataset'] == ds]
        x = np.arange(len(loss_variants))
        width = 0.6

        vals = []
        for lv in loss_variants:
            lv_data = sub[sub['loss_variant'] == lv]
            vals.append(lv_data['mIoU'].mean())

        bars = ax.bar(x, vals, width, color=loss_colors[:len(loss_variants)], edgecolor='white')

        # Add baseline reference
        bl = sub[sub['loss_variant'].isna()]
        if len(bl) > 0:
            bl_mean = bl['mIoU'].mean()
            ax.axhline(bl_mean, color='gray', linestyle='--', alpha=0.7, label=f'Standard ({bl_mean:.1f})')

        for i, v in enumerate(vals):
            ax.text(i, v + 0.3, f'{v:.1f}', ha='center', va='bottom', fontsize=10)

        ax.set_title(DATASET_SHORT.get(ds, ds), fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([lv.replace('_', '\n') for lv in loss_variants], fontsize=9)
        ax.set_ylabel('mIoU (%)')
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle('Loss Function Ablation', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUTDIR / 'figL1_loss_ablation.png', bbox_inches='tight')
    plt.close(fig)
    print("Saved figL1_loss_ablation.png")


# ============================================================
# COMBINATION ABLATION FIGURES
# ============================================================

def fig_combination_ablation():
    """Fig C1: Combination of strategies vs. individual strategies."""
    df = pd.read_csv(ABLATION_DIR / 'combination_ablation.csv')
    if df.empty:
        print("Skipping combination ablation figure (no data)")
        return

    models = sorted(df['model'].unique())

    fig, ax = plt.subplots(figsize=(14, 6))

    strategies = sorted(df['strategy'].unique())
    x = np.arange(len(strategies))
    width = 0.6 / len(models)

    for m_idx, model in enumerate(models):
        vals = []
        for strat in strategies:
            sub = df[(df['strategy'] == strat) & (df['model'] == model)]
            vals.append(sub['mIoU'].mean() if len(sub) > 0 else 0)
        offset = (m_idx - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=MODEL_SHORT.get(model, model),
               color=MODEL_COLORS.get(model, f'C{m_idx}'), edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('+', '\n+\n') for s in strategies], fontsize=8, rotation=0)
    ax.set_ylabel('mIoU (%)')
    ax.set_title('Strategy Combination Ablation\n(Cityscapes, Detailed Test)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUTDIR / 'figC1_combination_ablation.png', bbox_inches='tight')
    plt.close(fig)
    print("Saved figC1_combination_ablation.png")


# ============================================================
# FROM-SCRATCH ABLATION FIGURES
# ============================================================

def fig_from_scratch():
    """Fig FS1: Pretrained vs. from-scratch comparison across strategies."""
    df = pd.read_csv(ABLATION_DIR / 'from_scratch_ablation.csv')
    if df.empty:
        print("Skipping from-scratch figure (no data)")
        return

    # Focus on key strategies and BDD10k dataset
    key_strats = ['baseline', 'gen_cycleGAN', 'gen_TSIT', 'gen_IP2P',
                  'gen_flux_kontext', 'gen_step1x_new', 'gen_VisualCloze',
                  'gen_albumentations_weather', 'gen_automold']
    ds = 'bdd10k'
    sub = df[(df['dataset'] == ds) & (df['strategy'].isin(key_strats))]

    if sub.empty:
        # Try all strategies
        sub = df[df['dataset'] == ds]
        key_strats = sorted(sub['strategy'].unique())

    models = sorted(sub['model'].unique())

    fig, ax = plt.subplots(figsize=(14, 6))

    strats_in_data = [s for s in key_strats if s in sub['strategy'].values]
    x = np.arange(len(strats_in_data))
    width = 0.6 / max(len(models), 1)

    for m_idx, model in enumerate(models):
        vals = []
        for strat in strats_in_data:
            row = sub[(sub['strategy'] == strat) & (sub['model'] == model)]
            vals.append(row['mIoU'].mean() if len(row) > 0 else 0)
        offset = (m_idx - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width,
               label=MODEL_SHORT.get(model, model),
               color=MODEL_COLORS.get(model, f'C{m_idx}'),
               edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGY_SHORT.get(s, s) for s in strats_in_data], fontsize=9, rotation=30, ha='right')
    ax.set_ylabel('mIoU (%)')
    ax.set_title(f'From-Scratch Training: {DATASET_SHORT.get(ds, ds)}\n(No pretrained backbone)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUTDIR / 'figFS1_from_scratch.png', bbox_inches='tight')
    plt.close(fig)
    print("Saved figFS1_from_scratch.png")


def fig_pretrained_vs_scratch_delta():
    """Fig FS2: Delta (pretrained - from_scratch) showing backbone benefit per strategy."""
    df_scratch = pd.read_csv(ABLATION_DIR / 'from_scratch_ablation.csv')
    df_ratio = pd.read_csv(ABLATION_DIR / 'ratio_ablation_synthesized.csv')

    if df_scratch.empty:
        print("Skipping pretrained vs scratch delta (no scratch data)")
        return

    # Get shared strategies × datasets × models
    strategies_s = set(df_scratch['strategy'].unique())
    strategies_r = set(df_ratio['strategy'].unique()) | {'baseline'}

    common_strats = sorted(strategies_s & strategies_r)
    ds = 'bdd10k'  # Focus on BDD10k

    deltas = []
    strat_labels = []

    for strat in common_strats:
        scratch_rows = df_scratch[(df_scratch['strategy'] == strat) & (df_scratch['dataset'] == ds)]
        if strat == 'baseline':
            pretrained_rows = df_ratio[(df_ratio['strategy'].isin(df_ratio['strategy'].unique())) &
                                        (df_ratio['dataset'] == ds) &
                                        (df_ratio['ratio'] == 1.0)]
        else:
            pretrained_rows = df_ratio[(df_ratio['strategy'] == strat) &
                                        (df_ratio['dataset'] == ds) &
                                        (df_ratio['ratio'] == 0.5)]

        if len(scratch_rows) > 0 and len(pretrained_rows) > 0:
            s_mean = scratch_rows['mIoU'].mean()
            p_mean = pretrained_rows['mIoU'].mean()
            deltas.append(p_mean - s_mean)
            strat_labels.append(STRATEGY_SHORT.get(strat, strat))

    if not deltas:
        print("Skipping pretrained vs scratch delta (no matching data)")
        return

    # Sort by delta
    sort_idx = np.argsort(deltas)[::-1]
    deltas = [deltas[i] for i in sort_idx]
    strat_labels = [strat_labels[i] for i in sort_idx]

    fig, ax = plt.subplots(figsize=(10, max(4, len(strat_labels) * 0.4)))
    colors = ['#2ecc71' if d > 0 else '#e74c3c' for d in deltas]
    ax.barh(range(len(strat_labels)), deltas, color=colors, edgecolor='white', height=0.6)
    ax.set_yticks(range(len(strat_labels)))
    ax.set_yticklabels(strat_labels, fontsize=10)
    ax.set_xlabel('Δ mIoU: Pretrained − From-Scratch (pp)')
    ax.set_title(f'Backbone Pretraining Benefit per Strategy\n({DATASET_SHORT.get(ds, ds)})',
                 fontsize=13, fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for i, d in enumerate(deltas):
        ax.text(d + 0.2, i, f'{d:+.1f}', va='center', fontsize=9)

    fig.tight_layout()
    fig.savefig(OUTDIR / 'figFS2_pretrained_vs_scratch_delta.png', bbox_inches='tight')
    plt.close(fig)
    print("Saved figFS2_pretrained_vs_scratch_delta.png")


# ============================================================
# EXTENDED TRAINING FIGURES
# ============================================================

def fig_extended_training():
    """Fig E1: mIoU vs. iteration for extended training study."""
    df = pd.read_csv(ABLATION_DIR / 'extended_training_ablation.csv')
    if df.empty:
        print("Skipping extended training figure (no data)")
        return

    strategies = sorted(df['strategy'].unique())
    models = sorted(df['model'].unique())
    stages = sorted(df['stage'].unique())

    # Focus on detailed test type
    df_det = df[df['test_type'] == 'detailed']
    if df_det.empty:
        df_det = df

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), squeeze=False)
    axes = axes.flatten()

    strat_colors = plt.cm.tab10(np.linspace(0, 0.8, len(strategies)))

    for m_idx, model in enumerate(models):
        ax = axes[m_idx]
        for s_idx, strat in enumerate(strategies):
            sub = df_det[(df_det['model'] == model) & (df_det['strategy'] == strat)]
            if sub.empty:
                continue
            sub = sub.sort_values('iteration')
            ax.plot(sub['iteration'] / 1000, sub['mIoU'], 'o-',
                    color=strat_colors[s_idx],
                    label=STRATEGY_SHORT.get(strat, strat),
                    markersize=4, linewidth=1.5)

        ax.set_title(MODEL_SHORT.get(model, model), fontweight='bold')
        ax.set_xlabel('Iteration (×1000)')
        if m_idx == 0:
            ax.set_ylabel('mIoU (%)')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc='best')

    fig.suptitle('Extended Training: mIoU vs. Training Duration', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUTDIR / 'figE1_extended_training.png', bbox_inches='tight')
    plt.close(fig)
    print("Saved figE1_extended_training.png")


# ============================================================
# NOISE ABLATION (additional figures not in generate_ablation_figures.py)
# ============================================================

def fig_noise_per_dataset_model():
    """Fig N1: Noise vs baseline bar chart per dataset×model (full detail)."""
    df = pd.read_csv(ABLATION_DIR / 'noise_ablation.csv')
    models = ['pspnet_r50', 'segformer_mit-b3', 'segnext_mscan-b', 'mask2former_swin-b']
    datasets = ['bdd10k', 'iddaw', 'mapillaryvistas', 'outside15k']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        sub = df[(df['dataset'] == ds) & (df['model'].isin(models))]
        models_in = sorted(sub['model'].unique())

        x = np.arange(len(models_in))
        w = 0.25

        r0 = [sub[(sub['model'] == m) & (sub['ratio'] == 0.0)]['mIoU'].values[0]
              if len(sub[(sub['model'] == m) & (sub['ratio'] == 0.0)]) > 0 else 0
              for m in models_in]
        r5 = [sub[(sub['model'] == m) & (sub['ratio'] == 0.5)]['mIoU'].values[0]
              if len(sub[(sub['model'] == m) & (sub['ratio'] == 0.5)]) > 0 else 0
              for m in models_in]

        ax.bar(x - w/2, r0, w, label='r=0.0 (100% noise)', color='#e74c3c', edgecolor='white')
        ax.bar(x + w/2, r5, w, label='r=0.5 (50% noise)', color='#3498db', edgecolor='white')

        for i in range(len(models_in)):
            delta = r5[i] - r0[i]
            y = max(r0[i], r5[i]) + 0.3
            ax.text(x[i], y, f'Δ={delta:+.1f}', ha='center', fontsize=8)

        ax.set_title(DATASET_SHORT[ds], fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_SHORT.get(m, m) for m in models_in], fontsize=10)
        ax.set_ylabel('mIoU (%)')
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle('Noise Ablation: Per Dataset × Model Detail', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUTDIR / 'figN1_noise_per_dataset_model.png', bbox_inches='tight')
    plt.close(fig)
    print("Saved figN1_noise_per_dataset_model.png")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Generating comprehensive ablation figures...")
    print("=" * 60)

    # Synthesized Ratio Ablation (from ratio_ablation_synthesized.csv)
    print("\n--- Synthesized Ratio Ablation ---")
    fig_ratio_overview_curves()
    fig_ratio_by_family()
    fig_ratio_heatmap()
    fig_ratio_sensitivity()
    fig_ratio_best_strategies()
    fig_ratio_individual_strategies()
    fig_ratio_model_comparison()

    # Loss Ablation
    print("\n--- Loss Ablation ---")
    fig_loss_ablation()

    # Combination Ablation
    print("\n--- Combination Ablation ---")
    fig_combination_ablation()

    # From-Scratch
    print("\n--- From-Scratch Ablation ---")
    fig_from_scratch()
    fig_pretrained_vs_scratch_delta()

    # Extended Training
    print("\n--- Extended Training ---")
    fig_extended_training()

    # Noise (additional detail)
    print("\n--- Noise Ablation (additional) ---")
    fig_noise_per_dataset_model()

    print(f"\n{'=' * 60}")
    print(f"All figures saved to {OUTDIR}/")
    print(f"{'=' * 60}")
