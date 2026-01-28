"""Quick comparison of model inference time: 19 vs 66 classes"""
import torch
import torch.nn as nn
import time

# Simulate DeepLabV3+ forward pass with different output channels
def measure_inference_time(out_channels, batch_size=10, iterations=50):
    device = torch.device('cuda')
    
    # Simple simulated decoder (convolution to out_channels)
    model = nn.Sequential(
        nn.Conv2d(2048, 512, 3, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.Conv2d(512, 256, 3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Conv2d(256, out_channels, 1),  # Final classification
    ).to(device).eval()
    
    x = torch.randn(batch_size, 2048, 64, 64).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(x)
    torch.cuda.synchronize()
    
    # Time
    start = time.time()
    for _ in range(iterations):
        with torch.no_grad():
            y = model(x)
            _ = y.argmax(dim=1)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    return elapsed / iterations * 1000  # ms per batch

if __name__ == "__main__":
    print("Testing inference time with different output channels...")
    print("=" * 60)
    
    for out_ch in [19, 66]:
        time_ms = measure_inference_time(out_ch)
        print(f"Output channels: {out_ch:3d} -> {time_ms:.2f} ms/batch")
    
    print("=" * 60)
    print("This tests only the decoder part, not full model")
