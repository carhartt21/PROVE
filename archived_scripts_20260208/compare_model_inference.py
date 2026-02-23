"""Direct comparison: BDD10k model vs MapillaryVistas model inference time"""
import torch
import time
import sys
sys.path.insert(0, '${HOME}/repositories/PROVE')

from mmengine.config import Config
from mmseg.registry import MODELS
from mmengine.model.utils import revert_sync_batchnorm
import mmseg.models  # Register all mmseg models including SegDataPreProcessor
from mmseg.utils import register_all_modules
register_all_modules(init_default_scope=True)

def test_model_speed(config_path, checkpoint_path, name, batch_size=10, iterations=20):
    """Test model inference speed."""
    device = torch.device('cuda')
    
    # Load model
    print(f"\nLoading {name} model...")
    cfg = Config.fromfile(config_path)
    model = MODELS.build(cfg.model)
    model = revert_sync_batchnorm(model)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    x = torch.randn(batch_size, 3, 512, 512).to(device)
    batch_img_metas = [{'ori_shape': (512, 512), 'img_shape': (512, 512), 
                        'pad_shape': (512, 512), 'scale_factor': (1.0, 1.0)}] * batch_size
    
    # Warmup
    print("Warming up...")
    for _ in range(5):
        with torch.no_grad():
            _ = model.inference(x, batch_img_metas)
    torch.cuda.synchronize()
    
    # Time
    print("Timing...")
    times = []
    for i in range(iterations):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            result = model.inference(x, batch_img_metas)
            if isinstance(result, torch.Tensor):
                _ = result.argmax(dim=1)
        torch.cuda.synchronize()
        times.append(time.time() - start)
    
    avg_time = sum(times) / len(times)
    return avg_time * 1000  # ms

if __name__ == "__main__":
    models = [
        {
            'name': 'BDD10k (19 classes)',
            'config': '${AWARE_DATA_ROOT}/WEIGHTS/baseline/bdd10k/deeplabv3plus_r50/training_config.py',
            'checkpoint': '${AWARE_DATA_ROOT}/WEIGHTS/baseline/bdd10k/deeplabv3plus_r50/iter_80000.pth',
        },
        {
            'name': 'MapillaryVistas (66 classes)',
            'config': '${AWARE_DATA_ROOT}/WEIGHTS/baseline/mapillaryvistas/deeplabv3plus_r50/training_config.py',
            'checkpoint': '${AWARE_DATA_ROOT}/WEIGHTS/baseline/mapillaryvistas/deeplabv3plus_r50/iter_80000.pth',
        },
    ]
    
    print("=" * 60)
    print("MODEL INFERENCE TIME COMPARISON")
    print("=" * 60)
    
    results = []
    for model_info in models:
        time_ms = test_model_speed(
            model_info['config'],
            model_info['checkpoint'],
            model_info['name']
        )
        results.append((model_info['name'], time_ms))
        print(f"\n{model_info['name']}: {time_ms:.2f} ms/batch (10 images)")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, time_ms in results:
        print(f"  {name}: {time_ms:.2f} ms/batch = {time_ms/10:.2f} ms/image")
    
    if len(results) == 2:
        ratio = results[1][1] / results[0][1]
        print(f"\n  Slowdown factor: {ratio:.2f}x")
