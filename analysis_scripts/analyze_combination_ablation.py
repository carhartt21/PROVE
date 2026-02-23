#!/usr/bin/env python3
"""
Combination Strategy Ablation Study Analysis

Analyzes the results of combining different data augmentation strategies.
This is an ablation study to understand the effects of combining generative
and standard augmentation methods.

Combination Types:
- Generative + Standard: gen_CUT+std_mixup, gen_cycleGAN+std_randaugment, etc.
- Standard + Standard: std_randaugment+std_mixup, std_cutmix+std_autoaugment, etc.
- Baseline + Standard: baseline+std_cutmix

Results are stored in WEIGHTS_COMBINATIONS folder (separated from single strategies).

Usage:
    mamba run -n prove python analyze_combination_ablation.py

Output:
    result_figures/combination_ablation/
    
See docs/COMBINATION_ABLATION.md for detailed documentation.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Output directory
OUTPUT_DIR = Path("result_figures/combination_ablation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Weights directories
WEIGHTS_COMBINATIONS = "${AWARE_DATA_ROOT}/WEIGHTS_COMBINATION_ABLATION"
WEIGHTS_SINGLE = "${AWARE_DATA_ROOT}/WEIGHTS_CITYSCAPES_GEN"

# Strategy family definitions for components
COMPONENT_FAMILIES = {
    "Generative (Augmenters)": ["gen_augmenters"],
    "Generative (Diffusion)": ["gen_Img2Img"],
    "Generative (Multimodal)": ["gen_Qwen_Image_Edit"],
    "Standard Augmentation": ["std_autoaugment", "std_randaugment"],
    "Standard Mixing": ["std_cutmix", "std_mixup"],
    "Baseline": ["baseline"]
}

# Reverse mapping
COMPONENT_TO_FAMILY = {}
for family, components in COMPONENT_FAMILIES.items():
    for comp in components:
        COMPONENT_TO_FAMILY[comp] = family


def parse_combination_name(name: str) -> dict:
    """Parse a combination strategy name into components."""
    parts = name.split('+')
    result = {
        'components': parts,
        'n_components': len(parts),
        'component_families': []
    }
    
    for part in parts:
        family = COMPONENT_TO_FAMILY.get(part, "Unknown")
        if family not in result['component_families']:
            result['component_families'].append(family)
    
    # Classify combination type
    if any(p.startswith("gen_") for p in parts) and any(p.startswith("std_") for p in parts):
        result['combination_type'] = "Generative + Standard"
    elif sum(1 for p in parts if p.startswith("std_")) >= 2:
        result['combination_type'] = "Standard + Standard"
    elif "baseline" in name:
        result['combination_type'] = "Baseline + Standard"
    else:
        result['combination_type'] = "Other"
    
    return result


def load_combination_results(combinations_root: str, downstream_csv: str = None) -> pd.DataFrame:
    """Load results for combination strategies."""
    
    results = []
    weights_path = Path(combinations_root)
    
    if not weights_path.exists():
        print(f"Combinations directory not found: {combinations_root}")
        return pd.DataFrame()
    
    for strategy_dir in weights_path.iterdir():
        if not strategy_dir.is_dir():
            continue
        
        strategy = strategy_dir.name
        combo_info = parse_combination_name(strategy)
        
        for dataset_dir in strategy_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            dataset = dataset_dir.name
            
            for model_dir in dataset_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                model = model_dir.name
                
                # Look for test results
                test_dirs = [
                    model_dir / "test_results",
                    model_dir / "test_results_detailed"
                ]
                
                miou = None
                macc = None
                
                for test_dir in test_dirs:
                    if not test_dir.exists():
                        continue
                    
                    # Find most recent test result
                    for subdir in sorted(test_dir.iterdir(), reverse=True):
                        if not subdir.is_dir():
                            continue
                        
                        # Look for JSON results
                        for json_file in subdir.glob("*.json"):
                            try:
                                with open(json_file) as f:
                                    data = json.load(f)
                                if isinstance(data, dict):
                                    miou = data.get('mIoU', data.get('miou'))
                                    macc = data.get('mAcc', data.get('macc'))
                                    if miou is None and 'overall' in data:
                                        miou = data['overall'].get('mIoU', data['overall'].get('miou'))
                                        macc = data['overall'].get('mAcc', data['overall'].get('macc'))
                                    if miou is not None:
                                        break
                            except:
                                continue
                        
                        # Look for test_report.txt
                        report_file = subdir / "test_report.txt"
                        if report_file.exists() and miou is None:
                            try:
                                content = report_file.read_text()
                                for line in content.split('\n'):
                                    if 'mIoU:' in line and 'Per-Class' not in content[:content.index(line)]:
                                        parts = line.split(':')
                                        if len(parts) >= 2:
                                            miou = float(parts[1].strip().split()[0])
                                            break
                            except:
                                pass
                        
                        if miou is not None:
                            break
                    
                    if miou is not None:
                        break
                
                if miou is not None:
                    # Normalize model name: strip _ratio* suffix for cross-comparison
                    import re as _re
                    model_clean = _re.sub(r'_ratio\dp\d+$', '', model)
                    result = {
                        'strategy': strategy,
                        'dataset': dataset,
                        'model': model_clean,
                        'model_raw': model,
                        'mIoU': miou,
                        'mAcc': macc,
                        'combination_type': combo_info['combination_type'],
                        'n_components': combo_info['n_components'],
                        'components': '+'.join(combo_info['components']),
                        'component_families': '+'.join(combo_info['component_families']),
                        'component_1': combo_info['components'][0],
                        'component_2': combo_info['components'][1] if len(combo_info['components']) > 1 else None
                    }
                    results.append(result)
    
    # Also try to load from downstream_results.csv if provided
    if downstream_csv and os.path.exists(downstream_csv):
        try:
            df_downstream = pd.read_csv(downstream_csv)
            # Filter for combination strategies
            df_combinations = df_downstream[df_downstream['strategy'].str.contains(r'\+', regex=True)]
            
            for _, row in df_combinations.iterrows():
                strategy = row['strategy']
                combo_info = parse_combination_name(strategy)
                
                result = {
                    'strategy': strategy,
                    'dataset': row.get('dataset', 'unknown'),
                    'model': row.get('model', 'unknown'),
                    'mIoU': row.get('mIoU'),
                    'mAcc': row.get('mAcc'),
                    'combination_type': combo_info['combination_type'],
                    'n_components': combo_info['n_components'],
                    'components': '+'.join(combo_info['components']),
                    'component_families': '+'.join(combo_info['component_families']),
                    'component_1': combo_info['components'][0],
                    'component_2': combo_info['components'][1] if len(combo_info['components']) > 1 else None
                }
                results.append(result)
        except Exception as e:
            print(f"Error loading downstream CSV: {e}")
    
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.drop_duplicates(subset=['strategy', 'dataset', 'model'], keep='first')
    
    return df


def load_single_strategy_results(weights_root: str, strategies: list) -> pd.DataFrame:
    """Load results for single (non-combined) strategies for comparison."""
    
    results = []
    weights_path = Path(weights_root)
    
    for strategy in strategies:
        strategy_dir = weights_path / strategy
        if not strategy_dir.exists():
            continue
        
        for dataset_dir in strategy_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            dataset = dataset_dir.name
            
            for model_dir in dataset_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                model = model_dir.name
                
                # Look for test results
                test_dirs = [model_dir / "test_results", model_dir / "test_results_detailed"]
                
                miou = None
                
                for test_dir in test_dirs:
                    if not test_dir.exists():
                        continue
                    
                    for subdir in sorted(test_dir.iterdir(), reverse=True):
                        if not subdir.is_dir():
                            continue
                        
                        for json_file in subdir.glob("*.json"):
                            try:
                                with open(json_file) as f:
                                    data = json.load(f)
                                if isinstance(data, dict):
                                    miou = data.get('mIoU', data.get('miou'))
                                    if miou is None and 'overall' in data:
                                        miou = data['overall'].get('mIoU', data['overall'].get('miou'))
                                    if miou is not None:
                                        break
                            except:
                                continue
                        
                        if miou is not None:
                            break
                    
                    if miou is not None:
                        break
                
                if miou is not None:
                    # Normalize model name: strip _ratio* suffix for cross-comparison
                    import re as _re
                    model_clean = _re.sub(r'_ratio\dp\d+$', '', model)
                    results.append({
                        'strategy': strategy,
                        'dataset': dataset,
                        'model': model_clean,
                        'mIoU': miou
                    })
    
    return pd.DataFrame(results)


def compute_combination_effect(combo_df: pd.DataFrame, single_df: pd.DataFrame) -> pd.DataFrame:
    """Compute the effect of combining strategies vs individual components."""
    
    results = []
    
    for _, row in combo_df.iterrows():
        components = row['components'].split('+')
        dataset = row['dataset']
        model = row['model']
        combo_miou = row['mIoU']
        
        # Get individual component performance
        component_mious = []
        for comp in components:
            match = single_df[
                (single_df['strategy'] == comp) &
                (single_df['dataset'] == dataset) &
                (single_df['model'] == model)
            ]
            if len(match) > 0:
                component_mious.append((comp, match['mIoU'].iloc[0]))
        
        if len(component_mious) >= 2:
            best_single = max([m for _, m in component_mious])
            avg_single = np.mean([m for _, m in component_mious])
            
            result = row.to_dict()
            result['best_component_mIoU'] = best_single
            result['avg_component_mIoU'] = avg_single
            result['improvement_over_best'] = combo_miou - best_single
            result['improvement_over_avg'] = combo_miou - avg_single
            result['synergy'] = 'Positive' if combo_miou > best_single else ('Neutral' if combo_miou == best_single else 'Negative')
            
            for i, (comp, miou) in enumerate(component_mious):
                result[f'comp{i+1}_name'] = comp
                result[f'comp{i+1}_mIoU'] = miou
            
            results.append(result)
    
    return pd.DataFrame(results)


def create_combination_overview(df: pd.DataFrame, output_path: Path):
    """Create overview of combination strategies performance."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Plot 1: Performance by combination type
    ax1 = axes[0, 0]
    type_means = df.groupby('combination_type')['mIoU'].agg(['mean', 'std', 'count']).reset_index()
    type_means = type_means.sort_values('mean', ascending=True)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(type_means)))
    ax1.barh(range(len(type_means)), type_means['mean'], 
             xerr=type_means['std'], capsize=5, color=colors)
    ax1.set_yticks(range(len(type_means)))
    ax1.set_yticklabels(type_means['combination_type'])
    ax1.set_xlabel('Mean mIoU')
    ax1.set_title('Performance by Combination Type', fontsize=12, fontweight='bold')
    
    # Add count annotations
    for i, (mean, count) in enumerate(zip(type_means['mean'], type_means['count'])):
        ax1.annotate(f'n={count}', xy=(mean + type_means['std'].iloc[i] + 0.5, i), va='center', fontsize=9)
    
    # Plot 2: Performance by combination strategy
    ax2 = axes[0, 1]
    strategy_means = df.groupby('strategy')['mIoU'].mean().sort_values(ascending=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(strategy_means)))
    ax2.barh(range(len(strategy_means)), strategy_means.values, color=colors)
    ax2.set_yticks(range(len(strategy_means)))
    ax2.set_yticklabels(strategy_means.index, fontsize=8)
    ax2.set_xlabel('Mean mIoU')
    ax2.set_title('Performance by Combination Strategy', fontsize=12, fontweight='bold')
    
    # Plot 3: Component frequency
    ax3 = axes[1, 0]
    component_counts = pd.concat([df['component_1'], df['component_2']]).value_counts()
    ax3.bar(range(len(component_counts)), component_counts.values, color='steelblue')
    ax3.set_xticks(range(len(component_counts)))
    ax3.set_xticklabels(component_counts.index, rotation=45, ha='right')
    ax3.set_ylabel('Frequency in Combinations')
    ax3.set_title('Component Frequency', fontsize=12, fontweight='bold')
    
    # Plot 4: Performance distribution
    ax4 = axes[1, 1]
    for ctype in df['combination_type'].unique():
        subset = df[df['combination_type'] == ctype]['mIoU']
        ax4.hist(subset, bins=15, alpha=0.5, label=ctype)
    ax4.set_xlabel('mIoU')
    ax4.set_ylabel('Frequency')
    ax4.set_title('mIoU Distribution by Combination Type', fontsize=12, fontweight='bold')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_synergy_analysis(effect_df: pd.DataFrame, output_path: Path):
    """Create analysis of combination synergy effects."""
    
    if len(effect_df) == 0:
        print("No effect data available for synergy analysis")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Plot 1: Improvement over best component
    ax1 = axes[0, 0]
    strategy_imp = effect_df.groupby('strategy')['improvement_over_best'].mean().sort_values(ascending=True)
    colors = ['green' if x > 0 else 'red' for x in strategy_imp.values]
    ax1.barh(range(len(strategy_imp)), strategy_imp.values, color=colors)
    ax1.set_yticks(range(len(strategy_imp)))
    ax1.set_yticklabels(strategy_imp.index, fontsize=8)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('Improvement over Best Component (mIoU)')
    ax1.set_title('Synergy Effect: Combination vs Best Component', fontsize=12, fontweight='bold')
    
    # Plot 2: Synergy distribution
    ax2 = axes[0, 1]
    synergy_counts = effect_df['synergy'].value_counts()
    colors_pie = {'Positive': '#2ecc71', 'Neutral': '#f39c12', 'Negative': '#e74c3c'}
    ax2.pie(synergy_counts.values, labels=synergy_counts.index, autopct='%1.1f%%',
            colors=[colors_pie.get(s, 'gray') for s in synergy_counts.index])
    ax2.set_title('Synergy Distribution', fontsize=12, fontweight='bold')
    
    # Plot 3: Combination vs components scatter
    ax3 = axes[1, 0]
    ax3.scatter(effect_df['best_component_mIoU'], effect_df['mIoU'], 
                c=effect_df['combination_type'].astype('category').cat.codes, 
                cmap='Set2', alpha=0.7, s=50)
    
    # Add diagonal line
    lims = [min(ax3.get_xlim()[0], ax3.get_ylim()[0]), max(ax3.get_xlim()[1], ax3.get_ylim()[1])]
    ax3.plot(lims, lims, 'k--', alpha=0.5, label='Parity')
    ax3.set_xlabel('Best Component mIoU')
    ax3.set_ylabel('Combination mIoU')
    ax3.set_title('Combination vs Best Component', fontsize=12, fontweight='bold')
    ax3.legend()
    
    # Plot 4: Improvement by combination type
    ax4 = axes[1, 1]
    type_groups = effect_df.groupby('combination_type')['improvement_over_best'].agg(['mean', 'std'])
    type_groups = type_groups.sort_values('mean', ascending=True)
    
    colors = ['green' if x > 0 else 'red' for x in type_groups['mean'].values]
    ax4.barh(range(len(type_groups)), type_groups['mean'], 
             xerr=type_groups['std'], capsize=5, color=colors)
    ax4.set_yticks(range(len(type_groups)))
    ax4.set_yticklabels(type_groups.index)
    ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_xlabel('Mean Improvement over Best Component')
    ax4.set_title('Synergy by Combination Type', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_component_interaction_heatmap(effect_df: pd.DataFrame, output_path: Path):
    """Create heatmap showing interaction effects between components."""
    
    if len(effect_df) == 0:
        return
    
    # Create interaction matrix
    components = list(set(effect_df['component_1'].unique()) | set(effect_df['component_2'].dropna().unique()))
    components = sorted(components)
    
    interaction_matrix = pd.DataFrame(index=components, columns=components, dtype=float)
    
    for _, row in effect_df.iterrows():
        c1, c2 = row['component_1'], row['component_2']
        if c1 and c2:
            imp = row['improvement_over_best']
            if pd.isna(interaction_matrix.loc[c1, c2]):
                interaction_matrix.loc[c1, c2] = imp
                interaction_matrix.loc[c2, c1] = imp
            else:
                # Average multiple results
                interaction_matrix.loc[c1, c2] = (interaction_matrix.loc[c1, c2] + imp) / 2
                interaction_matrix.loc[c2, c1] = interaction_matrix.loc[c1, c2]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    mask = interaction_matrix.isna()
    sns.heatmap(interaction_matrix.astype(float), annot=True, fmt='.2f', 
                cmap='RdYlGn', center=0, ax=ax, mask=mask,
                cbar_kws={'label': 'Synergy (Δ mIoU vs Best)'})
    ax.set_title('Component Interaction Effects', fontsize=14, fontweight='bold')
    ax.set_xlabel('Component 2')
    ax.set_ylabel('Component 1')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_summary_table(df: pd.DataFrame, effect_df: pd.DataFrame = None) -> pd.DataFrame:
    """Create summary table for combination ablation study."""
    
    summary = []
    
    for strategy in df['strategy'].unique():
        strat_df = df[df['strategy'] == strategy]
        
        row = {
            'Strategy': strategy,
            'N_Results': len(strat_df),
            'Mean_mIoU': strat_df['mIoU'].mean(),
            'Std_mIoU': strat_df['mIoU'].std(),
            'Combination_Type': strat_df['combination_type'].iloc[0],
            'Component_1': strat_df['component_1'].iloc[0],
            'Component_2': strat_df['component_2'].iloc[0]
        }
        
        if effect_df is not None and len(effect_df) > 0:
            effect_match = effect_df[effect_df['strategy'] == strategy]
            if len(effect_match) > 0:
                row['Best_Component_mIoU'] = effect_match['best_component_mIoU'].mean()
                row['Improvement_vs_Best'] = effect_match['improvement_over_best'].mean()
                row['Synergy'] = effect_match['synergy'].mode().iloc[0] if len(effect_match) > 0 else 'Unknown'
        
        summary.append(row)
    
    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.sort_values('Mean_mIoU', ascending=False)
    
    return summary_df


def main():
    """Main function to run combination ablation study analysis."""
    
    print("=" * 60)
    print("Combination Strategy Ablation Study")
    print("=" * 60)
    
    # Paths
    downstream_csv = "${HOME}/repositories/PROVE/downstream_results.csv"
    
    # Load combination results
    print("\nLoading combination strategy results...")
    combo_df = load_combination_results(WEIGHTS_COMBINATIONS, downstream_csv)
    print(f"Loaded {len(combo_df)} results from {combo_df['strategy'].nunique()} combination strategies")
    
    if len(combo_df) == 0:
        print("No combination results found!")
        return
    
    # Print combination strategies
    print("\nCombination strategies:")
    for strategy in sorted(combo_df['strategy'].unique()):
        count = len(combo_df[combo_df['strategy'] == strategy])
        ctype = combo_df[combo_df['strategy'] == strategy]['combination_type'].iloc[0]
        print(f"  {strategy} ({ctype}): {count} results")
    
    # Get unique components for single strategy loading
    components = set()
    for _, row in combo_df.iterrows():
        if row['component_1']:
            components.add(row['component_1'])
        if row['component_2']:
            components.add(row['component_2'])
    
    # Load single strategy results for comparison
    print("\nLoading single strategy results for comparison...")
    single_df = load_single_strategy_results(WEIGHTS_SINGLE, list(components))
    print(f"Loaded {len(single_df)} single strategy results")
    
    # Compute combination effects
    effect_df = pd.DataFrame()
    if len(single_df) > 0:
        print("\nComputing combination effects...")
        effect_df = compute_combination_effect(combo_df, single_df)
        print(f"Computed effects for {len(effect_df)} combinations")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # 1. Overview
    print("  - Combination overview...")
    create_combination_overview(combo_df, OUTPUT_DIR / "combination_overview.png")
    
    # 2. Synergy analysis
    if len(effect_df) > 0:
        print("  - Synergy analysis...")
        create_synergy_analysis(effect_df, OUTPUT_DIR / "synergy_analysis.png")
        
        # 3. Component interaction heatmap
        print("  - Component interaction heatmap...")
        create_component_interaction_heatmap(effect_df, OUTPUT_DIR / "component_interaction.png")
    
    # Create summary
    print("\nCreating summary...")
    summary_df = create_summary_table(combo_df, effect_df)
    summary_df.to_csv(OUTPUT_DIR / "combination_ablation_summary.csv", index=False)
    
    # Save raw data
    combo_df.to_csv(OUTPUT_DIR / "combination_results.csv", index=False)
    if len(effect_df) > 0:
        effect_df.to_csv(OUTPUT_DIR / "combination_effects.csv", index=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("COMBINATION ABLATION SUMMARY")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    
    # Print key findings
    if len(effect_df) > 0:
        print("\n" + "=" * 60)
        print("KEY FINDINGS")
        print("=" * 60)
        
        positive_synergy = effect_df[effect_df['synergy'] == 'Positive']
        negative_synergy = effect_df[effect_df['synergy'] == 'Negative']
        
        print(f"\nPositive synergy combinations: {len(positive_synergy['strategy'].unique())}")
        print(f"Negative synergy combinations: {len(negative_synergy['strategy'].unique())}")
        
        if len(positive_synergy) > 0:
            best_combo = positive_synergy.groupby('strategy')['improvement_over_best'].mean().idxmax()
            best_improvement = positive_synergy.groupby('strategy')['improvement_over_best'].mean().max()
            print(f"\nBest synergy: {best_combo} (+{best_improvement:.2f} mIoU)")
        
        if len(negative_synergy) > 0:
            worst_combo = negative_synergy.groupby('strategy')['improvement_over_best'].mean().idxmin()
            worst_improvement = negative_synergy.groupby('strategy')['improvement_over_best'].mean().min()
            print(f"Worst synergy: {worst_combo} ({worst_improvement:.2f} mIoU)")
    
    # Save text report
    with open(OUTPUT_DIR / "combination_ablation_report.txt", 'w') as f:
        f.write("Combination Strategy Ablation Study Report\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("1. COMBINATION STRATEGIES\n")
        f.write("-" * 40 + "\n")
        for strategy in sorted(combo_df['strategy'].unique()):
            strat_df = combo_df[combo_df['strategy'] == strategy]
            f.write(f"\n{strategy}:\n")
            f.write(f"  Type: {strat_df['combination_type'].iloc[0]}\n")
            f.write(f"  Results: {len(strat_df)}\n")
            f.write(f"  Mean mIoU: {strat_df['mIoU'].mean():.2f}\n")
        
        f.write("\n\n2. SUMMARY TABLE\n")
        f.write("-" * 40 + "\n")
        f.write(summary_df.to_string(index=False))
        
        if len(effect_df) > 0:
            f.write("\n\n3. SYNERGY ANALYSIS\n")
            f.write("-" * 40 + "\n")
            synergy_stats = effect_df.groupby('synergy').size()
            f.write(f"Positive synergy: {synergy_stats.get('Positive', 0)}\n")
            f.write(f"Neutral: {synergy_stats.get('Neutral', 0)}\n")
            f.write(f"Negative synergy: {synergy_stats.get('Negative', 0)}\n")
    
    print(f"\nResults saved to {OUTPUT_DIR}/")
    print("Generated files:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
