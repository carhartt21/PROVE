#!/usr/bin/env python3
"""Check ACDC label format and conversion."""

import numpy as np
from PIL import Image
from pathlib import Path
import sys
# Add project root to path dynamically
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))
from custom_transforms import CITYSCAPES_ID_TO_TRAINID

label_path = Path('${AWARE_DATA_ROOT}/FINAL_SPLITS/test/labels/ACDC')

# Create LUT
lut = np.full(256, 255, dtype=np.uint8)
for label_id, train_id in CITYSCAPES_ID_TO_TRAINID.items():
    if 0 <= label_id < 256:
        lut[label_id] = train_id

print("=" * 70)
print("ACDC LABEL CONVERSION CHECK")
print("=" * 70)

for domain in ['clear_day', 'foggy', 'snowy', 'rainy', 'night']:
    domain_label = label_path / domain
    if domain_label.exists():
        files = list(domain_label.glob('*.png'))[:2]
        for f in files:
            label_raw = np.array(Image.open(f))
            label_conv = lut[label_raw]
            raw_uniq = sorted(np.unique(label_raw))[:12]
            conv_uniq = sorted(np.unique(label_conv))[:12]
            
            # Count non-ignore pixels
            n_valid_raw = np.sum((label_raw != 0) & (label_raw != 255))
            n_valid_conv = np.sum(label_conv != 255)
            
            print(f"\n{domain} - {f.name[:40]}:")
            print(f"  Raw unique:       {raw_uniq}")
            print(f"  Converted unique: {conv_uniq}")
            print(f"  Valid pixels: raw={n_valid_raw}, conv={n_valid_conv}")
