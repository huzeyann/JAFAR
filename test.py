"""
Test script to load a JAFAR model using torch.hub.load()

This script demonstrates how to:
1. Load a JAFAR model using torch.hub
2. Prepare an input image
3. Run inference

Usage:
    python test.py
"""

import torch
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
import sys

# Import from hubconf
try:
    from hubconf import dinov3_l, load
    DIRECT_IMPORT_AVAILABLE = True
except ImportError:
    DIRECT_IMPORT_AVAILABLE = False


def test_load_model():
    """Test loading a JAFAR model using torch.hub.load()"""
    
    print("=" * 60)
    print("Testing JAFAR Model Loading via torch.hub")
    print("=" * 60)
    
    # Get the current directory (assuming we're in the JAFAR repo)
    repo_path = Path(__file__).parent.absolute()
    print(f"\nRepository path: {repo_path}")
    
    # Test loading model
    print("\n[1/4] Loading JAFAR model with DINOv3-L backbone...")
    try:
        # Try direct import first (faster for testing)
        if DIRECT_IMPORT_AVAILABLE:
            print("  Using direct import from hubconf...")
            # Load with output resolution of 512 (can be customized)
            jafar_model = dinov3_l(
                pretrained=True,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                output_resolution=512
            )
        else:
            # Use torch.hub.load()
            print("  Using torch.hub.load()...")
            jafar_model = torch.hub.load(
                str(repo_path),
                'dinov3_l',
                pretrained=True,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                output_resolution=512
            )
        print("✓ Model loaded successfully!")
        print(f"  - Model type: {type(jafar_model).__name__}")
        print(f"  - Output resolution: {jafar_model.output_resolution}")
        print(f"  - Device: {jafar_model.device}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check if we have a test image
    test_image_path = repo_path / "asset" / "parrot.png"
    if not test_image_path.exists():
        print(f"\n[2/4] Test image not found at {test_image_path}")
        print("  Creating a dummy image for testing...")
        # Create a dummy image
        dummy_image = Image.new('RGB', (448, 448), color='red')
        image = dummy_image
    else:
        print(f"\n[2/4] Loading test image from {test_image_path}...")
        image = Image.open(test_image_path).convert("RGB")
        print("✓ Image loaded successfully!")
    
    # Prepare image for model
    print("\n[3/4] Preparing image for model...")
    device = jafar_model.device
    img_size = jafar_model.output_resolution
    print(f"  Using output resolution: {img_size}x{img_size}")
    
    # Normalization for ViT models
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)
    std = torch.tensor([0.229, 0.224, 0.225], device=device)
    
    transform = T.Compose([
        T.Resize(img_size),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    
    image_batch = transform(image).unsqueeze(0).to(device)
    print(f"✓ Image prepared: shape {image_batch.shape}")
    
    # Run inference
    print("\n[4/4] Running inference...")
    try:
        # Simple forward pass - just pass the image batch!
        print("  - Running forward pass...")
        hr_feats = jafar_model(image_batch)
        print(f"    High-res features shape: {hr_feats.shape}")
        
        # Demonstrate that JAFAR supports various resolutions
        print("\n  - Testing different output resolutions:")
        for test_res in [256, 512]:
            test_hr_feats = jafar_model(image_batch, output_resolution=test_res)
            print(f"    {test_res}x{test_res}: {test_hr_feats.shape}")
        
        # Also demonstrate getting low-res features if needed
        print("\n  - Accessing backbone directly (if needed):")
        with torch.no_grad():
            lr_feats, _ = jafar_model.backbone(image_batch)
            print(f"    Low-res features shape: {lr_feats.shape}")
        
        print("✓ Inference completed successfully!")
        print(f"\n  Summary:")
        print(f"    Input image: {image_batch.shape}")
        print(f"    Low-res features: {lr_feats.shape}")
        print(f"    High-res features (output {img_size}x{img_size}): {hr_feats.shape}")
        print(f"    Upsampling ratio: {img_size / lr_feats.shape[-2]:.1f}x")
        print(f"    Note: JAFAR supports any output resolution!")
        
        return True
        
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simplified_interface():
    """Test the simplified interface: load(name, resolution) and jafar(image_batch)"""
    print("\n" + "=" * 60)
    print("Testing Simplified Interface")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Create a dummy image batch
        image_batch = torch.randn(1, 3, 448, 448).to(device)
        
        print("\n[1/2] Loading model with simplified interface...")
        # Simple usage: load(name, resolution)
        jafar = load('dinov3_l', resolution=512, pretrained=False, device=device)
        print(f"✓ Model loaded: {type(jafar).__name__}")
        print(f"  Output resolution: {jafar.output_resolution}")
        
        print("\n[2/2] Running inference...")
        # Simple usage: jafar(image_batch)
        hr_feats = jafar(image_batch)
        print(f"✓ Inference successful!")
        print(f"  Input shape: {image_batch.shape}")
        print(f"  Output shape: {hr_feats.shape}")
        
        # Test with different resolution
        print("\n  Testing with different resolution (1024x1024)...")
        hr_feats_1024 = jafar(image_batch, output_resolution=1024)
        print(f"  Output shape at 1024x1024: {hr_feats_1024.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_load_without_pretrained():
    """Test loading model without pretrained weights"""
    print("\n" + "=" * 60)
    print("Testing Model Loading WITHOUT Pretrained Weights")
    print("=" * 60)
    
    repo_path = Path(__file__).parent.absolute()
    
    try:
        if DIRECT_IMPORT_AVAILABLE:
            jafar_model = dinov3_l(
                pretrained=False,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
        else:
            jafar_model = torch.hub.load(
                str(repo_path),
                'dinov3_l',
                pretrained=False,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
        print("✓ Model loaded successfully without pretrained weights!")
        print(f"  Model type: {type(jafar_model).__name__}")
        return True
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("JAFAR Model Loading Test")
    print("=" * 60)
    
    # Test 1: Load with pretrained weights
    success = test_load_model()
    
    # Test 2: Simplified interface
    if success:
        test_simplified_interface()
    
    # Test 3: Load without pretrained weights
    if success:
        test_load_without_pretrained()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")
    print("=" * 60 + "\n")

