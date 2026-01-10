import os
from pathlib import Path
from collections import defaultdict

WEIGHTS_ROOT = Path('/scratch/aaa_exchange/AWARE/WEIGHTS')

def summarize_detailed_results():
    strategies = sorted([d.name for d in WEIGHTS_ROOT.iterdir() if d.is_dir()])
    
    print(f"{'Strategy':<35} | {'BDD10k':<10} | {'IDD-AW':<10} | {'Mapillary':<10} | {'Outside15k':<10}")
    print("-" * 90)
    
    for strategy in strategies:
        strat_path = WEIGHTS_ROOT / strategy
        counts = defaultdict(int)
        datasets = ['bdd10k', 'idd-aw', 'mapillaryvistas', 'outside15k']
        
        for dataset in datasets:
            # Look for dataset_cd or dataset
            dataset_dirs = list(strat_path.glob(f"{dataset}*"))
            for ddir in dataset_dirs:
                # Check for detailed results in any model subdirectory
                detailed = list(ddir.glob("*/test_results_detailed"))
                if detailed:
                    counts[dataset] = len(detailed)
        
        results = []
        for dataset in datasets:
            count = counts[dataset]
            results.append(f"{count if count > 0 else '-':<10}")
            
        print(f"{strategy:<35} | {' | '.join(results)}")

if __name__ == '__main__':
    summarize_detailed_results()
