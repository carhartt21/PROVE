#!/usr/bin/env python3
"""
Run missing MapillaryVistas domain adaptation tests.

This script runs the domain adaptation tests for MapillaryVistas models
that were missed due to SOURCE_DATASETS being incorrectly configured.

Usage:
    # Dry run
    python scripts/run_missing_mv_domain_adaptation.py --dry-run
    
    # Run tests locally (one at a time)
    python scripts/run_missing_mv_domain_adaptation.py
    
    # Run with limit
    python scripts/run_missing_mv_domain_adaptation.py --limit 10
"""

import json
import subprocess
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run missing MV domain adaptation tests')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be run')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of tests to run')
    args = parser.parse_args()
    
    # Load missing tests
    missing_file = PROJECT_ROOT / 'missing_mv_tests.json'
    if not missing_file.exists():
        print("Error: missing_mv_tests.json not found. Run the analysis first.")
        return 1
    
    with open(missing_file) as f:
        missing = json.load(f)
    
    print(f"Found {len(missing)} missing MapillaryVistas domain adaptation tests")
    
    if args.limit > 0:
        missing = missing[:args.limit]
        print(f"Limited to {len(missing)} tests")
    
    print()
    
    script_path = PROJECT_ROOT / 'scripts' / 'run_domain_adaptation_tests.py'
    
    for i, test in enumerate(missing, 1):
        strategy = test['strategy']
        model = test['model']
        
        cmd = [
            sys.executable,
            str(script_path),
            '--source-dataset', 'mapillaryvistas',
            '--model', model,
            '--strategy', strategy,
        ]
        
        print(f"[{i}/{len(missing)}] {strategy}/mapillaryvistas/{model}")
        
        if args.dry_run:
            print(f"  Would run: {' '.join(cmd)}")
        else:
            print(f"  Running...")
            result = subprocess.run(cmd, cwd=PROJECT_ROOT)
            if result.returncode != 0:
                print(f"  WARNING: Test failed with return code {result.returncode}")
        
        print()
    
    print("Done!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
