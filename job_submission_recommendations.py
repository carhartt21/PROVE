#!/usr/bin/env python3
"""
Generate prioritized job submission recommendations.

Based on the sanity check analysis, this script recommends which jobs 
to submit first for maximum value.
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict

def main():
    df = pd.read_csv('${HOME}/repositories/PROVE/downstream_results.csv')
    
    # Filter to bad tests (need re-running)
    bad = df[df['mIoU'] <= 10.0].copy()
    
    print("=" * 90)
    print("PRIORITIZED JOB SUBMISSION RECOMMENDATIONS")
    print("=" * 90)
    
    # Priority 1: BDD10K_CD - Most reliable dataset, uses Cityscapes trainIDs
    print("\n" + "=" * 90)
    print("PRIORITY 1: BDD10K_CD (44 jobs)")
    print("=" * 90)
    print("""
Reason: 
- BDD10K uses Cityscapes trainIDs natively - most reliable after fix
- 1,857 test images across 7 weather domains
- Results directly comparable to valid BDD10K tests
- Quick validation: baseline should achieve ~44% mIoU (like bdd10k proper)

Command:
  cd ${HOME}/repositories/PROVE
  bash scripts/retest_jobs/submit_bdd10k_cd_retests.sh
  
Expected time: ~2-3 hours total (44 jobs × ~3-4 min each with parallelism)
""")
    
    bdd_cd = bad[bad['dataset'] == 'bdd10k_cd']
    print(f"Jobs by strategy:")
    for strat in sorted(bdd_cd['strategy'].unique()):
        count = len(bdd_cd[bdd_cd['strategy'] == strat])
        print(f"  {strat:35s}: {count} jobs")
    
    # Priority 2: IDD-AW_CD - Fix the label corruption issue
    print("\n" + "=" * 90)
    print("PRIORITY 2: IDD-AW_CD (11 jobs)")
    print("=" * 90)
    print("""
Reason:
- IDD-AW labels were converted to Cityscapes trainIDs
- 2,928 test images
- Need to verify label fix worked correctly
- Some tests with corrupted labels need to be excluded

Note: Exclude results from *_corrupted_labels_backup directories!

Command:
  bash scripts/retest_jobs/submit_idd-aw_cd_retests.sh
  
Expected time: ~1-2 hours (11 jobs)
""")
    
    iddaw_cd = bad[bad['dataset'] == 'idd-aw_cd']
    print(f"Jobs by strategy:")
    for strat in sorted(iddaw_cd['strategy'].unique()):
        count = len(iddaw_cd[iddaw_cd['strategy'] == strat])
        print(f"  {strat:35s}: {count} jobs")
    
    # Priority 3: MapillaryVistas_CD - Now with fixed label conversion
    print("\n" + "=" * 90)
    print("PRIORITY 3: MapillaryVistas_CD (55 jobs)")
    print("=" * 90)
    print("""
Reason:
- MapillaryVistas RGB labels now converted to Cityscapes trainIDs
- Previously had 66 native classes vs 19 predicted classes (mismatch)
- Now should show reasonable results (~30-40% mIoU expected)
- Large dataset with diverse scenes

Command:
  bash scripts/retest_jobs/submit_mapillaryvistas_cd_retests.sh
  
Expected time: ~3-4 hours (55 jobs)
""")
    
    mapillary_cd = bad[bad['dataset'] == 'mapillaryvistas_cd']
    print(f"Jobs by strategy:")
    for strat in sorted(mapillary_cd['strategy'].unique())[:10]:
        count = len(mapillary_cd[mapillary_cd['strategy'] == strat])
        print(f"  {strat:35s}: {count} jobs")
    print(f"  ... and {len(mapillary_cd['strategy'].unique()) - 10} more strategies")
    
    # Priority 4: OUTSIDE15k_CD - Similar to MapillaryVistas
    print("\n" + "=" * 90)
    print("PRIORITY 4: OUTSIDE15k_CD (60 jobs)")
    print("=" * 90)
    print("""
Reason:
- OUTSIDE15k native labels (24 classes) now converted to Cityscapes trainIDs
- 2,505 test images
- Good for cross-dataset generalization testing

Command:
  bash scripts/retest_jobs/submit_outside15k_cd_retests.sh
  
Expected time: ~3-4 hours (60 jobs)
""")
    
    outside_cd = bad[bad['dataset'] == 'outside15k_cd']
    print(f"Jobs by strategy:")
    for strat in sorted(outside_cd['strategy'].unique())[:10]:
        count = len(outside_cd[outside_cd['strategy'] == strat])
        print(f"  {strat:35s}: {count} jobs")
    print(f"  ... and {len(outside_cd['strategy'].unique()) - 10} more strategies")
    
    # Summary
    print("\n" + "=" * 90)
    print("RECOMMENDED SUBMISSION ORDER")
    print("=" * 90)
    print("""
For fastest validation:
1. Submit BDD10K_CD first (44 jobs) - ~2-3 hours
   - Wait for completion and verify baseline achieves ~44% mIoU
   - This confirms the label processing fix works

2. If successful, submit IDD-AW_CD (11 jobs) - ~1-2 hours
   - Verify IDD-AW label conversion works

3. Then submit MapillaryVistas_CD (55 jobs) - ~3-4 hours
   - Verify RGB→native→trainID conversion works

4. Finally submit OUTSIDE15k_CD (60 jobs) - ~3-4 hours
   - Verify native→trainID conversion works

Total: 170 jobs, ~10-12 hours if run sequentially, ~4-5 hours with parallelism

ONE-SHOT SUBMISSION (if confident):
  bash scripts/retest_jobs/submit_all_retests.sh
  
This submits all 170 jobs at once with 1-second delays between submissions.
""")
    
    # Quick validation script
    print("\n" + "=" * 90)
    print("QUICK VALIDATION (run one test manually)")
    print("=" * 90)
    print("""
To verify the fix before submitting all jobs:

cd ${HOME}/repositories/PROVE
python fine_grained_test.py \\
    --config ${AWARE_DATA_ROOT}/WEIGHTS/baseline/bdd10k_cd/deeplabv3plus_r50/training_config.py \\
    --checkpoint ${AWARE_DATA_ROOT}/WEIGHTS/baseline/bdd10k_cd/deeplabv3plus_r50/iter_80000.pth \\
    --output-dir ${AWARE_DATA_ROOT}/WEIGHTS/baseline/bdd10k_cd/deeplabv3plus_r50/test_results_detailed_quick_check \\
    --dataset BDD10k \\
    --data-root ${AWARE_DATA_ROOT}/FINAL_SPLITS

Expected result: ~44% mIoU on overall, ~40% on clear_day domain
If you see ~0.6% mIoU, the fix didn't work!
""")


if __name__ == '__main__':
    main()
