"""
Test script to visualize low-resolution features using mspace_color from ncut-pytorch

This script demonstrates how to:
1. Load a JAFAR model and extract low-resolution features
2. Visualize features using normalized cuts mspace_color (ncut-pytorch)

Usage:
    python test_ncut_lr_mspace.py
"""

import torch
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# Import from hubconf
try:
    from hubconf import dinov3_l, load
    DIRECT_IMPORT_AVAILABLE = True
except ImportError:
    DIRECT_IMPORT_AVAILABLE = False

# Try to import ncut-pytorch
NCUT_AVAILABLE = False
NCut = None
mspace_color = None
try:
    from ncut_pytorch import Ncut
    from ncut_pytorch.color import mspace_color
    NCut = Ncut
    NCUT_AVAILABLE = True
except ImportError:
    print("Warning: ncut-pytorch not found. Install with: pip install ncut-pytorch")
    NCUT_AVAILABLE = False


def get_ncut_eigenvectors(features, n_eig=20):
    """
    Get NCut eigenvectors from features.
    
    Args:
        features: Feature tensor of shape (B, C, H, W) or (C, H, W)
        n_eig: Number of eigenvectors to compute
    
    Returns:
        eigvecs: Eigenvectors tensor of shape (H*W, n_eig)
        H, W: Spatial dimensions
    """
    if not NCUT_AVAILABLE or NCut is None:
        return None, None, None
    
    # Ensure features are batched
    if features.dim() == 3:
        features = features.unsqueeze(0)
    
    # Get single batch
    feats = features[0]  # (C, H, W)
    C, H, W = feats.shape
    
    # Reshape features to (H*W, C) for ncut
    # ncut-pytorch expects features in shape (N, C) where N is number of samples
    feats_flat = feats.permute(1, 2, 0).reshape(H * W, C).detach().cpu()
    
    # Normalize features (optional but can help)
    feats_flat = (feats_flat - feats_flat.mean(dim=0)) / (feats_flat.std(dim=0) + 1e-8)
    
    try:
        # Get eigenvectors using Ncut
        ncut_model = NCut(n_eig=n_eig)
        eigvecs = ncut_model.fit_transform(feats_flat)  # Returns (H*W, n_eig)
        return eigvecs, H, W
    except Exception as e:
        print(f"Error computing NCut eigenvectors: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def visualize_features_with_mspace_color(features, n_eig=20, title="NCut M-Space Color", ax=None):
    """
    Visualize features using mspace_color from ncut-pytorch.
    
    Args:
        features: Feature tensor of shape (B, C, H, W) or (C, H, W)
        n_eig: Number of eigenvectors to compute
        title: Title for the plot
        ax: Matplotlib axis to plot on (optional)
    """
    if not NCUT_AVAILABLE or mspace_color is None:
        print("Skipping M-space color visualization: ncut-pytorch not available")
        return None
    
    eigvecs, H, W = get_ncut_eigenvectors(features, n_eig=n_eig)
    if eigvecs is None:
        return None
    
    try:
        # Use mspace_color to get RGB visualization
        rgb_mspace = mspace_color(eigvecs, progress_bar=True)  # Returns RGB image
        
        # Reshape to spatial dimensions
        rgb_mspace_spatial = rgb_mspace.reshape(H, W, 3).numpy()
        
        # Normalize to [0, 1] for display
        rgb_mspace_spatial = (rgb_mspace_spatial - rgb_mspace_spatial.min()) / (rgb_mspace_spatial.max() - rgb_mspace_spatial.min() + 1e-8)
        
    except Exception as e:
        print(f"Error applying mspace_color: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Visualize
    if ax is None:
        plt.figure(figsize=(8, 8))
        plt.imshow(rgb_mspace_spatial)
        plt.title(title)
        plt.axis('off')
    else:
        ax.imshow(rgb_mspace_spatial)
        ax.set_title(title)
        ax.axis('off')
    
    return rgb_mspace_spatial


def test_feature_visualization():
    """Test feature visualization using mspace_color on low-resolution features"""
    
    print("=" * 60)
    print("Testing Low-Resolution Feature Visualization with M-Space Color")
    print("=" * 60)
    
    # Get the current directory
    repo_path = Path(__file__).parent.absolute()
    
    # Load model
    print("\n[1/3] Loading JAFAR model...")
    try:
        if DIRECT_IMPORT_AVAILABLE:
            print("  Using direct import from hubconf...")
            jafar_model = dinov3_l(
                pretrained=True,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                output_resolution=512
            )
        else:
            print("  Using torch.hub.load()...")
            jafar_model = torch.hub.load(
                str(repo_path),
                'dinov3_l',
                pretrained=True,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                output_resolution=512
            )
        print("✓ Model loaded successfully!")
        device = jafar_model.device
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Load test image
    print("\n[2/3] Loading test image...")
    test_image_path = repo_path / "asset" / "parrot.png"
    if not test_image_path.exists():
        print(f"  Test image not found at {test_image_path}")
        print("  Creating a dummy image for testing...")
        image = Image.new('RGB', (448, 448), color='red')
    else:
        print(f"  Loading image from {test_image_path}...")
        image = Image.open(test_image_path).convert("RGB")
    print("✓ Image loaded successfully!")
    
    # Prepare image for model
    print("\n[3/3] Preparing image and extracting low-resolution features...")
    img_size = jafar_model.output_resolution
    
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
    print(f"  Input image shape: {image_batch.shape}")
    
    # Extract low-resolution features
    with torch.no_grad():
        lr_feats, _ = jafar_model.backbone(image_batch)
        print(f"  Low-res features shape: {lr_feats.shape}")
    
    print("✓ Features extracted successfully!")
    
    # Visualize features
    print("\n[4/4] Visualizing features with M-Space Color...")
    
    # Create output directory
    output_dir = repo_path / "visualizations"
    output_dir.mkdir(exist_ok=True)
    
    # Prepare original image for display (unnormalized)
    img_display = image_batch[0].cpu().clone()
    for t, m, s in zip(img_display, mean.cpu(), std.cpu()):
        t.mul_(s).add_(m)
    img_display = img_display.clamp(0, 1).permute(1, 2, 0).numpy()
    
    # Create figure with subplots: Original image and M-Space visualization
    fig = plt.figure(figsize=(16, 8))
    
    # Original image
    ax1 = plt.subplot(1, 2, 1)
    plt.imshow(img_display)
    plt.title("Original Image")
    plt.axis('off')
    
    # M-Space color visualization
    ax2 = plt.subplot(1, 2, 2)
    if NCUT_AVAILABLE:
        mspace_lr = visualize_features_with_mspace_color(lr_feats[0], n_eig=20, title="Low-Res M-Space Color", ax=ax2)
    else:
        ax2.text(0.5, 0.5, "NCut not available", ha='center', va='center', transform=ax2.transAxes)
        ax2.axis('off')
        mspace_lr = None
    
    # Save the combined visualization
    save_path = output_dir / "lr_mspace_visualization.png"
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"✓ Visualization saved to {save_path}")
    plt.close()
    
    # Also save individual M-space visualization
    if NCUT_AVAILABLE and mspace_lr is not None:
        plt.figure(figsize=(8, 8))
        plt.imshow(mspace_lr)
        plt.title("Low-Res Features (M-Space Color)")
        plt.axis('off')
        plt.savefig(output_dir / "mspace_color_lowres.png", bbox_inches='tight', dpi=150)
        plt.close()
        print("✓ M-space color visualization saved")
    
    print("\n✓ Visualization completed!")
    print(f"\n  Summary:")
    print(f"    Input image: {image_batch.shape}")
    print(f"    Low-res features: {lr_feats.shape}")
    print(f"    Visualizations saved to: {output_dir}")
    
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("JAFAR Low-Resolution M-Space Color Visualization Test")
    print("=" * 60)
    
    success = test_feature_visualization()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ Test completed successfully!")
    else:
        print("✗ Test failed!")
    print("=" * 60 + "\n")

