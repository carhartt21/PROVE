# t-SNE Domain Gap Visualization

Analyzes domain gap reduction between clear-day and adverse weather conditions using t-SNE embeddings of semantic segmentation features.

## Features

- **Feature extraction** from DeepLabv3+, PSPNet, SegFormer decoder layers
- **Pixel subsampling** (50k-100k) for computational efficiency  
- **3-panel visualization**: baseline vs augmented + silhouette comparison
- **Silhouette score quantification** (-1 to +1, lower = better domain invariance)

## Usage

```bash
python tools/tsne_domain_gap.py \
    --checkpoint-baseline ./checkpoints/baseline.pth \
    --checkpoint-augmented ./checkpoints/augmented.pth \
    --data-root ${AWARE_DATA_ROOT}/FINAL_SPLITS/ACDC \
    --model-type deeplabv3plus \
    --output ./tsne_output
```

## Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--checkpoint-baseline` | Baseline model checkpoint (required) | - |
| `--checkpoint-augmented` | Augmented model checkpoint (required) | - |
| `--data-root` | Dataset root path (required) | - |
| `--model-type` | `deeplabv3plus`, `pspnet`, `segformer` | `deeplabv3plus` |
| `--num-samples` | Pixels to sample for t-SNE | 75000 |
| `--output` | Output directory | `./tsne_output` |
| `--split` | Dataset split (`train`, `val`, `test`) | `val` |
| `--max-images-per-domain` | Max images per weather domain | 50 |

## Output

- `tsne_domain_gap_{model}.png` - 3-panel visualization
- `tsne_results_{model}.json` - Numerical results

## Interpretation

| Silhouette Score | Interpretation |
|-----------------|----------------|
| 0.4 - 0.6 | High domain gap (baseline) |
| 0.1 - 0.3 | Reduced gap (augmented) ✓ |
| < 0 | Overlapping domains (ideal) |
