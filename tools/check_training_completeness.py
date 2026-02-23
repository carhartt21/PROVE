#!/usr/bin/env python3
"""Analyze training completeness across all strategies."""
import os
import pandas as pd

WEIGHTS_DIR = '${AWARE_DATA_ROOT}/WEIGHTS'
GENERATED_IMAGES_DIR = '${AWARE_DATA_ROOT}/GENERATED_IMAGES'

DATASETS = ['bdd10k', 'idd-aw', 'mapillaryvistas']
MODELS = ['deeplabv3plus_r50', 'deeplabv3plus_r50_clear_day', 'pspnet_r50', 
          'pspnet_r50_clear_day', 'segformer_mit-b5', 'segformer_mit-b5_clear_day']

strategies = [d for d in os.listdir(WEIGHTS_DIR) if os.path.isdir(os.path.join(WEIGHTS_DIR, d))]
strategies = [s for s in strategies if s != 'baseline']

complete = []
incomplete = []

for strategy in strategies:
    strategy_path = os.path.join(WEIGHTS_DIR, strategy)
    trained_count = 0
    missing_info = {}
    
    for dataset in DATASETS:
        dataset_path = os.path.join(strategy_path, dataset.lower())
        missing_info[dataset] = {'trained': 0, 'missing': []}
        
        for model in MODELS:
            model_path = os.path.join(dataset_path, model)
            checkpoint_path = os.path.join(model_path, 'iter_80000.pth')
            
            if os.path.exists(checkpoint_path):
                trained_count += 1
                missing_info[dataset]['trained'] += 1
            else:
                missing_info[dataset]['missing'].append(model)
    
    data_available = {}
    if strategy.startswith('gen_'):
        gen_name = strategy[4:]
        manifest_path = os.path.join(GENERATED_IMAGES_DIR, gen_name, 'manifest.csv')
        
        if os.path.exists(manifest_path):
            try:
                df = pd.read_csv(manifest_path)
                for dataset in DATASETS:
                    if 'dataset' in df.columns:
                        count = len(df[df['dataset'].str.lower() == dataset.lower()])
                    else:
                        count = len(df[df['original_path'].str.contains(dataset, case=False, na=False)])
                    data_available[dataset] = count
            except:
                pass
    
    if trained_count == 18:
        complete.append(strategy)
    else:
        incomplete.append({
            'strategy': strategy, 
            'trained': trained_count, 
            'missing_info': missing_info, 
            'data_available': data_available
        })

print('COMPLETE STRATEGIES (18/18):')
print('-' * 40)
for s in complete:
    print(f'  ok {s}')
print(f'Total: {len(complete)}')

print('')
print('INCOMPLETE STRATEGIES:')
print('-' * 40)
for item in incomplete:
    s = item['strategy']
    t = item['trained']
    print(f'\n{s} ({t}/18):')
    for dataset in DATASETS:
        info = item['missing_info'][dataset]
        missing = info['missing']
        if missing:
            data_count = item['data_available'].get(dataset, '?')
            if isinstance(data_count, int):
                data_str = f'[{data_count} images]' if data_count >= 1000 else f'[{data_count} - INSUFFICIENT]'
            else:
                data_str = '[?]'
            print(f'  {dataset}: {info["trained"]}/6 trained, {len(missing)} missing {data_str}')

print(f'\nTotal incomplete: {len(incomplete)}')
