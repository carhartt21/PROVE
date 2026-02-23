#!/usr/bin/env python3
"""
Generate per-domain analysis: Normal vs Adverse weather performance.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import defaultdict

# Weather domain classifications
NORMAL_DOMAINS = ['clear_day', 'cloudy']  # Good weather conditions
ADVERSE_DOMAINS = ['foggy', 'night', 'rainy', 'snowy']  # Challenging conditions
TRANSITION_DOMAINS = ['dawn_dusk']  # Low light but not adverse
ALL_DOMAINS = ['clear_day', 'cloudy', 'dawn_dusk', 'foggy', 'night', 'rainy', 'snowy']


def parse_per_domain_metrics(per_domain_str):
    """Parse the per_domain_metrics string (Python literal or JSON)."""
    if pd.isna(per_domain_str) or per_domain_str == '':
        return None
    try:
        # Try ast.literal_eval first (Python dict format)
        import ast
        return ast.literal_eval(per_domain_str)
    except:
        try:
            # Fall back to JSON
            return json.loads(per_domain_str)
        except:
            return None


def extract_domain_miou(per_domain_metrics):
    """Extract mIoU for each domain from per_domain_metrics."""
    if not per_domain_metrics:
        return {}
    
    domain_miou = {}
    for domain, metrics in per_domain_metrics.items():
        if isinstance(metrics, dict):
            # Handle nested structure: {"summary": {"mIoU": ...}, ...}
            if 'summary' in metrics and isinstance(metrics['summary'], dict):
                miou = metrics['summary'].get('mIoU', metrics['summary'].get('miou'))
            else:
                miou = metrics.get('mIoU', metrics.get('miou'))
            if miou is not None:
                domain_miou[domain.lower().replace('-', '_').replace(' ', '_')] = float(miou)
    return domain_miou


def compute_weather_metrics(domain_miou):
    """Compute normal, adverse, and transition weather metrics."""
    normal_values = [domain_miou.get(d) for d in NORMAL_DOMAINS if domain_miou.get(d) is not None]
    adverse_values = [domain_miou.get(d) for d in ADVERSE_DOMAINS if domain_miou.get(d) is not None]
    transition_values = [domain_miou.get(d) for d in TRANSITION_DOMAINS if domain_miou.get(d) is not None]
    
    normal_miou = np.mean(normal_values) if normal_values else None
    adverse_miou = np.mean(adverse_values) if adverse_values else None
    transition_miou = np.mean(transition_values) if transition_values else None
    
    # Domain gap = Normal - Adverse
    domain_gap = (normal_miou - adverse_miou) if (normal_miou and adverse_miou) else None
    
    return {
        'normal_miou': normal_miou,
        'adverse_miou': adverse_miou,
        'transition_miou': transition_miou,
        'domain_gap': domain_gap,
    }


def main():
    df = pd.read_csv('${HOME}/repositories/PROVE/downstream_results.csv')
    
    print("=" * 80)
    print("PER-DOMAIN ANALYSIS: Normal vs Adverse Weather Performance")
    print("=" * 80)
    
    # Extract per-domain metrics
    results = []
    for _, row in df.iterrows():
        per_domain = parse_per_domain_metrics(row.get('per_domain_metrics', ''))
        domain_miou = extract_domain_miou(per_domain)
        
        if domain_miou:
            weather_metrics = compute_weather_metrics(domain_miou)
            results.append({
                'strategy': row['strategy'],
                'dataset': row['dataset'],
                'model': row['model'],
                'overall_miou': row['mIoU'],
                **weather_metrics,
                **{f"domain_{d}": domain_miou.get(d) for d in ALL_DOMAINS}
            })
    
    if not results:
        print("\nNo per-domain metrics available in the results.")
        print("This data will be available after retest jobs complete.")
        return
    
    results_df = pd.DataFrame(results)
    
    # Filter out rows without weather breakdown
    valid_df = results_df.dropna(subset=['normal_miou', 'adverse_miou'])
    print(f"\nResults with per-domain breakdown: {len(valid_df)} / {len(df)}")
    
    if len(valid_df) == 0:
        print("\nNo results have per-domain breakdown yet.")
        print("This data will be available after retest jobs complete.")
        return
    
    # Strategy-level summary
    print("\n" + "=" * 80)
    print("STRATEGY WEATHER PERFORMANCE")
    print("=" * 80)
    print(f"\n{'Strategy':<35} {'Normal':>10} {'Adverse':>10} {'Gap':>10} {'Count':>8}")
    print("-" * 80)
    
    strategy_stats = valid_df.groupby('strategy').agg({
        'normal_miou': 'mean',
        'adverse_miou': 'mean',
        'domain_gap': 'mean',
        'overall_miou': 'count'
    }).sort_values('adverse_miou', ascending=False)
    
    for strategy, row in strategy_stats.iterrows():
        gap = f"+{row['domain_gap']:.2f}" if row['domain_gap'] > 0 else f"{row['domain_gap']:.2f}"
        print(f"{strategy:<35} {row['normal_miou']:>10.2f} {row['adverse_miou']:>10.2f} {gap:>10} {int(row['overall_miou']):>8}")
    
    # Best strategies for adverse weather
    print("\n" + "=" * 80)
    print("TOP 10 STRATEGIES FOR ADVERSE WEATHER (foggy, night, rainy, snowy)")
    print("=" * 80)
    print(f"\n{'Rank':<5} {'Strategy':<35} {'Adverse mIoU':>15} {'Normal mIoU':>15} {'Gap':>10}")
    print("-" * 80)
    
    top_adverse = strategy_stats.head(10)
    for i, (strategy, row) in enumerate(top_adverse.iterrows()):
        gap = f"+{row['domain_gap']:.2f}" if row['domain_gap'] > 0 else f"{row['domain_gap']:.2f}"
        print(f"{i+1:<5} {strategy:<35} {row['adverse_miou']:>15.2f} {row['normal_miou']:>15.2f} {gap:>10}")
    
    # Best strategies for reducing domain gap
    print("\n" + "=" * 80)
    print("TOP 10 STRATEGIES FOR SMALLEST DOMAIN GAP (Normal - Adverse)")
    print("=" * 80)
    print(f"\n{'Rank':<5} {'Strategy':<35} {'Domain Gap':>12} {'Normal':>10} {'Adverse':>10}")
    print("-" * 80)
    
    smallest_gap = strategy_stats.sort_values('domain_gap').head(10)
    for i, (strategy, row) in enumerate(smallest_gap.iterrows()):
        print(f"{i+1:<5} {strategy:<35} {row['domain_gap']:>12.2f} {row['normal_miou']:>10.2f} {row['adverse_miou']:>10.2f}")
    
    # Per-domain breakdown
    print("\n" + "=" * 80)
    print("PER-DOMAIN BREAKDOWN (Top 5 strategies per domain)")
    print("=" * 80)
    
    for domain in ALL_DOMAINS:
        col = f"domain_{domain}"
        if col not in valid_df.columns:
            continue
        
        domain_data = valid_df.dropna(subset=[col])
        if len(domain_data) == 0:
            continue
            
        print(f"\n### {domain.upper()}")
        domain_stats = domain_data.groupby('strategy')[col].mean().sort_values(ascending=False).head(5)
        for i, (strategy, miou) in enumerate(domain_stats.items()):
            print(f"  {i+1}. {strategy:<30} {miou:.2f}%")
    
    # Save to markdown
    output_path = Path('${HOME}/repositories/PROVE/result_figures/per_domain_analysis.md')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("# Per-Domain Analysis: Normal vs Adverse Weather\n\n")
        f.write(f"**Results with per-domain breakdown: {len(valid_df)} / {len(df)}**\n\n")
        
        f.write("## Weather Domain Categories\n")
        f.write(f"- **Normal**: {', '.join(NORMAL_DOMAINS)}\n")
        f.write(f"- **Adverse**: {', '.join(ADVERSE_DOMAINS)}\n")
        f.write(f"- **Transition**: {', '.join(TRANSITION_DOMAINS)}\n\n")
        
        f.write("## Strategy Weather Performance\n\n")
        f.write("| Strategy | Normal mIoU | Adverse mIoU | Domain Gap | Count |\n")
        f.write("|----------|-------------|--------------|------------|-------|\n")
        for strategy, row in strategy_stats.iterrows():
            gap = f"+{row['domain_gap']:.2f}" if row['domain_gap'] > 0 else f"{row['domain_gap']:.2f}"
            f.write(f"| {strategy} | {row['normal_miou']:.2f} | {row['adverse_miou']:.2f} | {gap} | {int(row['overall_miou'])} |\n")
    
    print(f"\n\nAnalysis saved to: {output_path}")


if __name__ == '__main__':
    main()
