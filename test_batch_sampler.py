#!/usr/bin/env python3
"""
Test script to verify MixedBatchSampler correctly handles ratio=0.0 (100% generated)
"""

import sys
sys.path.insert(0, '/home/mima2416/repositories/PROVE')

from mixed_dataloader import BatchSplitSampler

def test_batch_composition():
    """Test batch composition for different ratios"""
    
    test_cases = [
        (0.0, 8, "100% generated (0 real + 8 gen)"),
        (0.25, 8, "25% real (2 real + 6 gen)"),
        (0.5, 8, "50% real (4 real + 4 gen)"),
        (1.0, 8, "100% real (8 real + 0 gen)"),
    ]
    
    real_size = 100
    gen_size = 500
    
    print("="*70)
    print("BATCH COMPOSITION TEST")
    print("="*70)
    
    for ratio, batch_size, description in test_cases:
        print(f"\n{description}")
        print(f"Ratio: {ratio}, Batch Size: {batch_size}")
        print("-"*70)
        
        sampler = BatchSplitSampler(
            real_dataset_size=real_size,
            generated_dataset_size=gen_size,
            batch_size=batch_size,
            real_gen_ratio=ratio,
            shuffle=False,
            drop_last=False,
        )
        
        # Check first 5 batches
        for i, batch in enumerate(sampler):
            if i >= 5:
                break
            
            real_count = sum(1 for source, _ in batch if source == 'real')
            gen_count = sum(1 for source, _ in batch if source == 'generated')
            
            status = "✅" if (real_count == sampler.real_per_batch and 
                           gen_count == sampler.gen_per_batch) else "❌"
            
            print(f"  Batch {i+1}: {real_count} real + {gen_count} gen = {len(batch)} total {status}")
            
            if i == 0:  # Show expected for first batch
                print(f"    Expected: {sampler.real_per_batch} real + {sampler.gen_per_batch} gen")
        
        print(f"\nTotal batches generated: {len(sampler.batches)}")

if __name__ == "__main__":
    test_batch_composition()
