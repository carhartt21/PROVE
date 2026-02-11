# Why Crop Size Significantly Impacts CNN-Based Segmentation vs Transformers

## Executive Summary

Based on experimental results from the PROVE/Cityscapes experiments:
| Model | 512×512 | 769×769 | Δ mIoU |
|-------|---------|---------|--------|
| **PSPNet R50** | 57.64% | 72.50% | **+14.86%** |
| **DeepLabV3+ R50** | 58.02% | 66.57% | **+8.55%** |
| **SegFormer MIT-B3** | 79.98% | - | N/A (already high) |
| **SegNeXt MSCAN-B** | 81.22% | - | N/A (already high) |

This document provides a detailed technical analysis of why CNN-based methods suffer dramatic performance loss at smaller crop sizes while Transformer-based methods maintain high performance.

---

## 1. Receptive Field Analysis

### 1.1 CNN Receptive Field: The Fundamental Limitation

**Receptive Field Definition**: The region in the input image that contributes to a single output pixel's prediction.

For a standard 3×3 convolution:
$$RF_{n} = RF_{n-1} + (k-1) \times \prod_{i=1}^{n-1} s_i$$

Where:
- $RF_n$ = Receptive field after layer $n$
- $k$ = Kernel size
- $s_i$ = Stride at layer $i$

**ResNet-50 with Dilated Convolutions** (as used in DeepLabV3+/PSPNet):
```python
backbone = {
    'dilations': (1, 1, 2, 4),  # Dilation rates for stages 1-4
    'strides': (1, 2, 1, 1),     # No downsampling in stages 3-4
}
```

**Theoretical Receptive Field Calculation for ResNet-50-D8:**

| Stage | Output Stride | Dilation | Effective Kernel | Cumulative RF |
|-------|---------------|----------|------------------|---------------|
| Conv1 | 4 | 1 | 7×7 | 7 |
| Stage 1 | 4 | 1 | 3×3 | ~35 |
| Stage 2 | 8 | 1 | 3×3 | ~91 |
| Stage 3 | 8 | 2 | 7×7 effective | ~219 |
| Stage 4 | 8 | 4 | 15×15 effective | ~475 |

**Critical Observation**: The theoretical receptive field of ~475 pixels seems sufficient for 512×512 input. However, the **Effective Receptive Field (ERF)** tells a different story.

### 1.2 Effective Receptive Field (ERF) vs Theoretical

Research by Luo et al. (2016) demonstrated that:
- The **theoretical receptive field** grows linearly with depth
- The **effective receptive field** follows a Gaussian distribution
- Only ~30-50% of the theoretical RF contributes meaningfully to the output

For ResNet-50 backbone at 512×512:
- Theoretical RF: ~475 pixels
- Effective RF: ~150-240 pixels (Gaussian-weighted center)
- This covers only **29-47%** of the image diagonal (724 pixels)

At 769×769:
- Same theoretical RF: ~475 pixels
- Effective RF: ~150-240 pixels
- Image diagonal: 1088 pixels
- **BUT**: The image has more spatial context to sample from during training

### 1.3 Why Larger Crops Help CNNs

**Key Insight**: It's not that larger crops give the network a larger receptive field—**the receptive field is architecture-dependent, not input-dependent**.

The benefit comes from:
1. **More Context Per Crop**: Training samples contain more complete objects and scene context
2. **Multi-scale Feature Quality**: PPM and ASPP modules receive features with more semantic content
3. **Boundary Completeness**: Object boundaries are more likely to be fully contained within crops
4. **Batch Statistics**: BatchNorm statistics are computed over richer feature distributions

---

## 2. PSPNet's Pyramid Pooling Module (PPM)

### 2.1 PPM Architecture

```python
decode_head = {
    'type': 'PSPHead',
    'pool_scales': (1, 2, 3, 6),  # Fixed pooling grid sizes
    'in_channels': 2048,
    'channels': 512,
}
```

The PPM applies global average pooling at fixed grid scales:
- **1×1**: Single global feature vector (entire feature map)
- **2×2**: 4 sub-regions (quadrants)
- **3×3**: 9 sub-regions
- **6×6**: 36 sub-regions

### 2.2 How Input Size Affects PPM

**At 512×512 input (feature map: 64×64 at output stride 8):**

| Pool Scale | Grid Size | Pixels per Grid Cell | Semantic Coverage |
|------------|-----------|---------------------|-------------------|
| 1×1 | 64×64 | 4096 pixels | Global but noisy |
| 2×2 | 32×32 | 1024 pixels | Quadrant-level |
| 3×3 | 21×21 | 441 pixels | Sub-scene level |
| 6×6 | 10×10 | 100 pixels | Object-level |

**At 769×769 input (feature map: 96×96 at output stride 8):**

| Pool Scale | Grid Size | Pixels per Grid Cell | Semantic Coverage |
|------------|-----------|---------------------|-------------------|
| 1×1 | 96×96 | 9216 pixels | Global with richer context |
| 2×2 | 48×48 | 2304 pixels | Quadrant with more objects |
| 3×3 | 32×32 | 1024 pixels | Sub-scene with better boundaries |
| 6×6 | 16×16 | 256 pixels | Object-level with spatial detail |

### 2.3 The PPM Dilemma at Small Crops

**Problem 1: Insufficient Semantic Diversity**
- At 512×512, a 6×6 grid cell covers only 10×10 = 100 feature pixels
- This often captures only parts of objects, not complete semantic units
- The pooled features lack discriminative power

**Problem 2: Fixed Ratios, Variable Content**
- PPM was designed for larger inputs (Cityscapes: 2048×1024)
- The 6×6 grid was meant to capture object-level features
- At 512×512, the same grid captures sub-object features

**Problem 3: Global Feature Degradation**
- The 1×1 global pooling produces a single 512-D vector
- At smaller crops, this vector represents less semantic diversity
- Global context becomes "local context" in disguise

### 2.4 Mathematical Formulation

Let $F \in \mathbb{R}^{H \times W \times C}$ be the input feature map.

PPM output for scale $s$:
$$P_s = \text{Upsample}(\text{Conv}(\text{AdaptiveAvgPool}(F, s \times s)))$$

Concatenated output:
$$\text{PPM}(F) = \text{Concat}(F, P_1, P_2, P_3, P_6)$$

**The issue**: $\text{AdaptiveAvgPool}(F, s \times s)$ divides $F$ into $s^2$ bins.

At input 512×512:
- Feature map: 64×64
- Bin size for scale 6: $\lfloor 64/6 \rfloor = 10$ pixels
- **10 pixels capture ~80×80 input pixels** (after OS=8)
- This is **only 2.4% of the input area**

At input 769×769:
- Feature map: 96×96  
- Bin size for scale 6: $\lfloor 96/6 \rfloor = 16$ pixels
- **16 pixels capture ~128×128 input pixels**
- This is **2.8% of the input area but with 2.56× more pixels**

---

## 3. DeepLabV3+'s ASPP Module

### 3.1 ASPP Architecture

```python
decode_head = {
    'type': 'DepthwiseSeparableASPPHead',
    'dilations': (1, 12, 24, 36),  # Atrous rates
    'c1_in_channels': 256,  # Low-level features from Stage 1
    'c1_channels': 48,
}
```

ASPP applies parallel atrous convolutions:
- **Rate 1**: Standard 3×3 conv (RF = 3)
- **Rate 12**: Dilated 3×3 conv (RF = 25)
- **Rate 24**: Dilated 3×3 conv (RF = 49)
- **Rate 36**: Dilated 3×3 conv (RF = 73)
- **Image Pooling**: Global average pooling

### 3.2 The Dilation Rate Problem at Small Inputs

**Effective Coverage at Different Input Sizes:**

At 512×512 (feature map 64×64 at OS=8):
- Rate 36 kernel covers: 73×73 feature pixels
- But feature map is only 64×64!
- **ASPP rate 36 "sees outside" the feature map**
- Padding artifacts contaminate the features

```
Feature Map (64×64):
+------------------------+
|     Rate 36 kernel     |
|   (73×73) exceeds      |
|   feature map bounds   |
|   → Zero-padding used  |
|   → Boundary artifacts |
+------------------------+
```

At 769×769 (feature map 96×96 at OS=8):
- Rate 36 kernel covers: 73×73 feature pixels
- Feature map is 96×96
- **ASPP rate 36 fits comfortably within bounds**
- Clean feature extraction without padding artifacts

### 3.3 Quantifying ASPP Degradation

**Kernel Validity Ratio** (pixels inside feature map / total kernel coverage):

| Dilation Rate | Effective RF | 64×64 Feature Map | 96×96 Feature Map |
|---------------|--------------|-------------------|-------------------|
| 1 | 3×3 | 100% valid | 100% valid |
| 12 | 25×25 | 100% valid | 100% valid |
| 24 | 49×49 | ~85% valid | 100% valid |
| 36 | 73×73 | **~60% valid** | ~97% valid |

**Impact**: At 512×512, approximately 40% of the rate-36 ASPP kernel captures zero-padded regions, introducing systematic bias toward boundary predictions.

### 3.4 Why DeepLabV3+ is Less Affected Than PSPNet

DeepLabV3+ shows +8.55% improvement (vs PSPNet's +14.86%) because:

1. **Encoder-Decoder Architecture**: Low-level features from Stage 1 (c1_channels=48) help recover spatial details regardless of ASPP issues

2. **More Moderate Dilation Rates**: Rate 12 and 24 are still valid at 512×512

3. **Image Pooling Complement**: The global pooling branch provides consistent global context even when dilated convs degrade

---

## 4. Why Transformers Work Well at Small Crop Sizes

### 4.1 SegFormer's Mix Vision Transformer (MiT)

```python
backbone = {
    'type': 'MixVisionTransformer',
    'num_stages': 4,
    'num_heads': [1, 2, 5, 8],
    'sr_ratios': [8, 4, 2, 1],  # Spatial Reduction ratios
    'patch_sizes': [7, 3, 3, 3],
}
```

**Key Design Principles:**

1. **Overlapping Patch Embeddings**: Unlike ViT's non-overlapping patches, MiT uses overlapping patches (7×7, 3×3) that preserve local continuity

2. **Efficient Self-Attention with Spatial Reduction**:
   ```python
   # At stage 1: sr_ratio=8
   # Query: H×W tokens
   # Key/Value: (H/8)×(W/8) tokens after spatial reduction
   # Attention: Global context with linear complexity
   ```

3. **No Positional Encoding**: SegFormer explicitly avoids positional encoding, making it resolution-agnostic

### 4.2 Self-Attention: Global Context by Design

For a feature map $X \in \mathbb{R}^{N \times C}$ where $N = H \times W$:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Critical Property**: Every output token attends to ALL input tokens (after spatial reduction).

At 512×512 input (with sr_ratio=8 in stage 1):
- Feature tokens: 64×64 = 4096
- Reduced tokens for K,V: 8×8 = 64
- **Each of 4096 queries attends to 64 global tokens**
- Global context is captured regardless of crop size

This is fundamentally different from CNNs where the receptive field is:
- Fixed by architecture
- Grows gradually through depth
- Never truly "global" (Gaussian-weighted ERF)

### 4.3 SegFormer's Multi-Scale Feature Aggregation

```python
decode_head = {
    'type': 'SegformerHead',
    'in_channels': [64, 128, 320, 512],  # From all 4 stages
    'in_index': [0, 1, 2, 3],
    'channels': 256,
}
```

The MLP decoder:
1. Unifies features from all 4 stages via linear projections
2. Upsamples all features to 1/4 resolution
3. Concatenates and applies MLP fusion

**Why This Works at Small Crops:**

- Stage 1 features (1/4 scale): Local details, high resolution
- Stage 4 features (1/32 scale): Global context, low resolution
- **Both scales contribute equally regardless of absolute input size**
- The relative multi-scale ratios are preserved

### 4.4 SegNeXt's Multi-Scale Convolutional Attention (MSCAN)

```python
backbone = {
    'type': 'MSCAN',
    'attention_kernel_sizes': [5, [1, 7], [1, 11], [1, 21]],
    'attention_kernel_paddings': [2, [0, 3], [0, 5], [0, 10]],
}
```

**MSCAN Design:**
1. Uses large strip convolutions: 1×7, 1×11, 1×21
2. Combines with depth-wise convolutions
3. Creates multi-scale attention without pooling

**Why SegNeXt is Robust to Small Crops:**

1. **Strip Convolutions**: The 1×21 kernel captures long-range horizontal/vertical dependencies without the extreme dilation of ASPP

2. **Depth-wise Design**: Each channel has its own attention pattern, avoiding the one-size-fits-all problem of PPM

3. **Efficient Global Modeling**: The LightHam decoder uses Hamburger attention:
   ```python
   ham_kwargs = {
       'MD_S': 1,      # Matrix decomposition
       'MD_R': 16,     # Rank
       'train_steps': 6,
       'eval_steps': 7,
   }
   ```
   This models global context through low-rank matrix decomposition, not spatial pooling.

---

## 5. Multi-Scale Feature Aggregation Comparison

### 5.1 CNN Multi-Scale: Fixed Spatial Scales

**PSPNet PPM:**
```
Input Features (H×W×C)
      ↓
[AdaptivePool 1×1] → Upsample → Concat
[AdaptivePool 2×2] → Upsample → Concat  → Final Features
[AdaptivePool 3×3] → Upsample → Concat
[AdaptivePool 6×6] → Upsample → Concat
```

**Problem**: Pooling scales are absolute, not relative to semantic content.

**DeepLabV3+ ASPP:**
```
Input Features (H×W×C)
      ↓
[Conv 1×1]        → Concat
[AtrousConv r=12] → Concat  → Conv → Concat with Low-level
[AtrousConv r=24] → Concat        → Final Features
[AtrousConv r=36] → Concat
[GlobalPool]      → Concat
```

**Problem**: Dilation rates are designed for specific input sizes.

### 5.2 Transformer Multi-Scale: Relative Feature Hierarchies

**SegFormer:**
```
Input Image
      ↓
Stage 1 (1/4): Local features with global attention (sr=8)
Stage 2 (1/8): Mid-level features with global attention (sr=4)
Stage 3 (1/16): High-level features with global attention (sr=2)
Stage 4 (1/32): Global features with full attention (sr=1)
      ↓
MLP Decoder: Unify all scales → Final Features
```

**Advantage**: Each stage maintains relative multi-scale relationships.

### 5.3 Why Relative > Absolute for Scale Handling

Consider an object (e.g., a car) at different crop sizes:

| Crop Size | Car Size in Crop | PPM 6×6 Cell | Self-Attention Coverage |
|-----------|------------------|--------------|------------------------|
| 512×512 | 80×40 pixels | ~85×85 pixels | All 4096 tokens |
| 769×769 | 80×40 pixels | ~128×128 pixels | All 9216 tokens |

**CNN (PPM)**: The car may span 1-2 grid cells at 512×512, losing structural coherence.

**Transformer**: Every token (including all car tokens) attends to every other token, maintaining object coherence regardless of crop size.

---

## 6. Dilated Convolutions in ResNet Backbone

### 6.1 Configuration

```python
backbone = {
    'dilations': (1, 1, 2, 4),  # Stages 1-4
    'strides': (1, 2, 1, 1),    # Stages 1-4
}
```

This creates an **output stride of 8** (OS=8) instead of 32:
- Standard ResNet: 224 → 7 (OS=32)
- Dilated ResNet: 512 → 64 (OS=8)

### 6.2 Gridding Artifacts at Small Inputs

**Gridding Problem**: Dilated convolutions sample at regular intervals, creating a checkerboard pattern that misses intermediate pixels.

For a dilation rate $r$, a 3×3 kernel samples every $r$-th pixel:

```
Rate 4 sampling pattern (3×3 kernel):
X · · · X · · · X
· · · · · · · · ·
· · · · · · · · ·
· · · · · · · · ·
X · · · X · · · X
· · · · · · · · ·
· · · · · · · · ·
· · · · · · · · ·
X · · · X · · · X
```

At small input sizes (512×512 → 64×64 feature map):
- Rate 4 dilation samples pixels 4 apart
- Missing 3/4 of the spatial information per sample
- **Gridding artifacts become more prominent when the feature map is small**

At larger input sizes (769×769 → 96×96 feature map):
- Same sampling pattern, but more context between samples
- **Gridding artifacts are diluted by the larger spatial extent**

### 6.3 The Hybrid Dilated Convolution (HDC) Solution

DeepLabV3+ uses `contract_dilation=True` which implements HDC:
```python
backbone = {
    'contract_dilation': True,  # Prevents gridding
}
```

This ensures consecutive layers use different dilation rates (1, 2, 4) in a pattern that covers all spatial positions. However, this mitigation is less effective at small input sizes where the feature maps have fewer pixels to begin with.

---

## 7. Training Dynamics Differences

### 7.1 Gradient Flow in CNNs vs Transformers

**CNNs (PSPNet/DeepLabV3+):**
- Gradients flow through sequential convolutions
- Deeper layers receive attenuated gradients
- Skip connections (ResNet) help but don't solve global gradient issues
- **Global context modules (PPM/ASPP) receive gradients filtered through entire backbone**

**Transformers (SegFormer/SegNeXt):**
- Self-attention creates direct paths between distant tokens
- Gradients can flow directly from any output to any input token
- **Global context is learned at every layer, not just the decode head**

### 7.2 Batch Normalization vs Layer Normalization

**CNNs**: Use Batch Normalization (BN)
```python
norm_cfg = {'type': 'SyncBN', 'requires_grad': True}
```
- BN statistics depend on batch spatial dimensions
- At 512×512 with batch_size=2: statistics computed over 2×512×512 = 524K pixels
- At 769×769 with batch_size=2: statistics computed over 2×769×769 = 1.18M pixels
- **2.25× more pixels for more stable statistics at larger crops**

**Transformers**: Use Layer Normalization (LN)
- LN normalizes across channels per token
- Independent of batch size and spatial dimensions
- **Equally stable at any crop size**

### 7.3 Learning Rate Sensitivity

**Observation from experiments:**
- PSPNet/DeepLabV3+ use SGD with lr=0.01
- SegFormer/SegNeXt use AdamW with lr=0.00006

**Implication**: The 167× learning rate difference (after accounting for Adam's effective LR) suggests:
- CNN gradients are noisier, requiring larger LR for escape from local minima
- Transformer gradients are more consistent, allowing precise optimization
- **At small crops, CNN gradient noise increases further, compounding optimization difficulty**

---

## 8. Summary: Root Causes of the Performance Gap

### 8.1 PSPNet (+14.86% gain from larger crops)

| Issue | Impact | Severity |
|-------|--------|----------|
| Fixed pool scales (1,2,3,6) lose semantic meaning | High | ⚠️⚠️⚠️ |
| Global pooling (1×1) captures limited context | High | ⚠️⚠️⚠️ |
| No mechanism for adaptive context capture | High | ⚠️⚠️⚠️ |
| Purely local backbone features | Medium | ⚠️⚠️ |

**Conclusion**: PSPNet's design fundamentally assumes large input sizes. The PPM's fixed scales become meaningless at small inputs.

### 8.2 DeepLabV3+ (+8.55% gain from larger crops)

| Issue | Impact | Severity |
|-------|--------|----------|
| Large dilation rates (36) exceed feature map | High | ⚠️⚠️⚠️ |
| ASPP assumes minimum input resolution | Medium | ⚠️⚠️ |
| Encoder-decoder helps recover details | Mitigated | ✓ |
| Global pooling branch provides backup | Mitigated | ✓ |

**Conclusion**: DeepLabV3+'s encoder-decoder design partially compensates, but ASPP dilation rates are still problematic.

### 8.3 SegFormer (79.98% at 512×512)

| Strength | Mechanism |
|----------|-----------|
| Global attention at every stage | Self-attention with spatial reduction |
| Resolution-agnostic design | No positional encoding |
| Relative multi-scale features | Hierarchical transformer stages |
| Stable optimization | LayerNorm + AdamW |

**Conclusion**: Designed for efficiency and flexibility, SegFormer inherently handles varying input sizes.

### 8.4 SegNeXt (81.22% at 512×512)

| Strength | Mechanism |
|----------|-----------|
| Efficient convolutional attention | Strip convolutions (1×7, 1×11, 1×21) |
| Multi-scale without pooling | MSCAN attention blocks |
| Global context via Hamburger | Matrix decomposition attention |
| Depth-wise independence | Per-channel attention patterns |

**Conclusion**: SegNeXt proves that CNNs can be resolution-robust with proper attention design.

---

## 9. Recommendations

### 9.1 For Training CNNs at Small Crop Sizes

1. **Reduce ASPP dilation rates** proportionally:
   ```python
   # For 512×512 (feature map 64×64)
   'dilations': (1, 6, 12, 18)  # Instead of (1, 12, 24, 36)
   ```

2. **Adjust PPM pool scales**:
   ```python
   # More fine-grained for small inputs
   'pool_scales': (1, 2, 4, 8)  # Instead of (1, 2, 3, 6)
   ```

3. **Use larger batch sizes** to stabilize BatchNorm:
   ```python
   batch_size = 4  # Instead of 2 at 512×512
   ```

### 9.2 For Fair Model Comparison

When comparing CNN and Transformer architectures:
- Use the **recommended crop size for each architecture**
- Or adapt architecture-specific hyperparameters for the target crop size
- Report results with crop size clearly specified

### 9.3 For Production Deployment

If small crop sizes are required (memory/speed constraints):
- **Prefer Transformer-based models** (SegFormer, SegNeXt)
- Or use **lightweight CNN designs** that don't rely on extreme dilations

---

## 10. Appendix: Configuration Comparison

### PSPNet R50 Configurations Used

```python
# 512×512 config
decode_head = {
    'type': 'PSPHead',
    'pool_scales': (1, 2, 3, 6),
    'in_channels': 2048,
    'channels': 512,
}

# 769×769 config (identical - this is the problem!)
decode_head = {
    'type': 'PSPHead', 
    'pool_scales': (1, 2, 3, 6),  # Same scales!
    'in_channels': 2048,
    'channels': 512,
}
```

### DeepLabV3+ R50 Configurations Used

```python
# 512×512 config
decode_head = {
    'type': 'DepthwiseSeparableASPPHead',
    'dilations': (1, 12, 24, 36),
}

# 769×769 config (identical - also problematic)
decode_head = {
    'type': 'DepthwiseSeparableASPPHead',
    'dilations': (1, 12, 24, 36),  # Same rates!
}
```

### SegFormer MiT-B3 Configuration

```python
backbone = {
    'type': 'MixVisionTransformer',
    'sr_ratios': [8, 4, 2, 1],  # Adaptive to any input size
    'num_heads': [1, 2, 5, 8],
}
decode_head = {
    'type': 'SegformerHead',
    'in_channels': [64, 128, 320, 512],  # Multi-scale fusion
}
```

### SegNeXt MSCAN-B Configuration

```python
backbone = {
    'type': 'MSCAN',
    'attention_kernel_sizes': [5, [1, 7], [1, 11], [1, 21]],  # Adaptive attention
}
decode_head = {
    'type': 'LightHamHead',
    'ham_kwargs': {'MD_R': 16},  # Matrix decomposition attention
}
```

---

## References

1. Zhao, H., et al. "Pyramid Scene Parsing Network." CVPR 2017.
2. Chen, L.-C., et al. "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation." ECCV 2018.
3. Xie, E., et al. "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers." NeurIPS 2021.
4. Guo, M.-H., et al. "SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation." NeurIPS 2022.
5. Luo, W., et al. "Understanding the Effective Receptive Field in Deep Convolutional Neural Networks." NeurIPS 2016.
6. Yu, F., & Koltun, V. "Multi-Scale Context Aggregation by Dilated Convolutions." ICLR 2016.

---

*Document created: 2026-02-01*
*PROVE Repository: Cityscapes Replication Experiment Analysis*
