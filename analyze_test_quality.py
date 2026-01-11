#!/usr/bin/env python3
"""Analyze test quality and identify tests that need to be re-run."""

import pandas as pd
from datetime import datetime
from pathlib import Path

def main():
    df = pd.read_csv('downstream_results.csv')
    
    print("=" * 70)
    print("TEST QUALITY ANALYSIS")
    print("=" * 70)
    
    # The bug was fixed on Jan 10, 2026
    # Tests before that date on BDD10K/IDD-AW (trainID labels) need re-running
    
    # Group tests by quality
    good_threshold = 10.0  # mIoU > 10% is probably valid
    
    good = df[df['mIoU'] > good_threshold]
    bad = df[df['mIoU'] <= good_threshold]
    
    print(f"\nTotal results: {len(df)}")
    print(f"Good tests (mIoU > {good_threshold}%): {len(good)}")
    print(f"Bad tests (mIoU <= {good_threshold}%): {len(bad)}")
    
    print("\n" + "=" * 70)
    print("GOOD TESTS BY DATASET")
    print("=" * 70)
    for ds in sorted(df['dataset'].unique()):
        ds_good = good[good['dataset'] == ds]
        ds_total = df[df['dataset'] == ds]
        print(f"  {ds:25s}: {len(ds_good):3d} / {len(ds_total):3d} good")
    
    print("\n" + "=" * 70)
    print("BAD TESTS BREAKDOWN")
    print("=" * 70)
    for ds in sorted(bad['dataset'].unique()):
        ds_bad = bad[bad['dataset'] == ds]
        print(f"\n{ds}: {len(ds_bad)} bad tests")
        for strat in sorted(ds_bad['strategy'].unique()):
            count = len(ds_bad[ds_bad['strategy'] == strat])
            print(f"    {strat}: {count}")
    
    print("\n" + "=" * 70)
    print("TESTS NEEDING RE-RUN (by result_dir)")
    print("=" * 70)
    
    # Check result directories that have test_results_detailed (not fixed)
    needs_retest = bad[~bad['result_dir'].str.contains('_fixed', na=False)]
    print(f"\nTotal tests needing re-run: {len(needs_retest)}")
    
    # Group by strategy and dataset
    retest_summary = needs_retest.groupby(['dataset', 'strategy']).size().reset_index(name='count')
    print("\nBy dataset and strategy:")
    for ds in sorted(retest_summary['dataset'].unique()):
        ds_data = retest_summary[retest_summary['dataset'] == ds]
        print(f"\n  {ds}:")
        for _, row in ds_data.iterrows():
            print(f"    {row['strategy']:35s}: {row['count']}")
    
    # Generate list of directories needing re-test
    print("\n" + "=" * 70)
    print("DIRECTORIES NEEDING RE-TEST")
    print("=" * 70)
    for _, row in needs_retest.iterrows():
        base_dir = Path(row['result_dir']).parent.parent
        print(f"{base_dir}")

if __name__ == '__main__':
    main()
