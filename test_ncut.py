"""
Test script to visualize features using PCA and ncut-pytorch

This script demonstrates how to:
1. Load a JAFAR model and extract features
2. Visualize features using PCA
3. Visualize features using normalized cuts (ncut-pytorch)

Usage:
    python test_ncut.py
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

# Import visualization utilities
from utils.visualization import pca

# Try to import ncut-pytorch
NCUT_AVAILABLE = False
NCut = None
umap_color = None
tsne_color = None
mspace_color = None
try:
    from ncut_pytorch import Ncut
    from ncut_pytorch.color import umap_color, tsne_color, mspace_color
    from ncut_pytorch.utils.math import cosine_affinity, rbf_affinity
    NCut = Ncut
    NCUT_AVAILABLE = True
except ImportError:
    print("Warning: ncut-pytorch not found. Install with: pip install ncut-pytorch")
    NCUT_AVAILABLE = False


def visualize_features_with_pca(features, title="PCA Visualization", ax=None):
    """
    Visualize features using PCA.
    
    Args:
        features: Feature tensor of shape (B, C, H, W) or (C, H, W)
        title: Title for the plot
        ax: Matplotlib axis to plot on (optional)
    """
    # Ensure features are batched
    if features.dim() == 3:
        features = features.unsqueeze(0)
    
    # Use existing PCA function
    reduced_feats, _ = pca([features], dim=3, use_torch_pca=True)
    pca_vis = reduced_feats[0].squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    
    # Normalize to [0, 1] for display
    pca_vis = (pca_vis - pca_vis.min()) / (pca_vis.max() - pca_vis.min() + 1e-8)
    
    if ax is None:
        plt.figure(figsize=(8, 8))
        plt.imshow(pca_vis)
        plt.title(title)
        plt.axis('off')
    else:
        ax.imshow(pca_vis)
        ax.set_title(title)
        ax.axis('off')
    
    return pca_vis


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
        ncut_model = NCut(n_eig=n_eig, affinity_fn=rbf_affinity)
        eigvecs = ncut_model.fit_transform(feats_flat)  # Returns (H*W, n_eig)
        return eigvecs, H, W
    except Exception as e:
        print(f"Error computing NCut eigenvectors: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def visualize_features_with_umap_color(features, n_eig=20, title="NCut UMAP Color", ax=None):
    """
    Visualize features using umap_color from ncut-pytorch.
    
    Args:
        features: Feature tensor of shape (B, C, H, W) or (C, H, W)
        n_eig: Number of eigenvectors to compute
        title: Title for the plot
        ax: Matplotlib axis to plot on (optional)
    """
    if not NCUT_AVAILABLE or umap_color is None:
        print("Skipping UMAP color visualization: ncut-pytorch not available")
        return None
    
    eigvecs, H, W = get_ncut_eigenvectors(features, n_eig=n_eig)
    if eigvecs is None:
        return None
    
    try:
        # Use umap_color to get RGB visualization
        rgb_umap = umap_color(eigvecs)  # Returns RGB image
        
        # Reshape to spatial dimensions
        rgb_umap_spatial = rgb_umap.reshape(H, W, 3).numpy()
        
        # Normalize to [0, 1] for display
        rgb_umap_spatial = (rgb_umap_spatial - rgb_umap_spatial.min()) / (rgb_umap_spatial.max() - rgb_umap_spatial.min() + 1e-8)
        
    except Exception as e:
        print(f"Error applying umap_color: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Visualize
    if ax is None:
        plt.figure(figsize=(8, 8))
        plt.imshow(rgb_umap_spatial)
        plt.title(title)
        plt.axis('off')
    else:
        ax.imshow(rgb_umap_spatial)
        ax.set_title(title)
        ax.axis('off')
    
    return rgb_umap_spatial


def visualize_features_with_tsne_color(features, n_eig=20, title="NCut t-SNE Color", ax=None):
    """
    Visualize features using tsne_color from ncut-pytorch.
    
    Args:
        features: Feature tensor of shape (B, C, H, W) or (C, H, W)
        n_eig: Number of eigenvectors to compute
        title: Title for the plot
        ax: Matplotlib axis to plot on (optional)
    """
    if not NCUT_AVAILABLE or tsne_color is None:
        print("Skipping t-SNE color visualization: ncut-pytorch not available")
        return None
    
    eigvecs, H, W = get_ncut_eigenvectors(features, n_eig=n_eig)
    if eigvecs is None:
        return None
    
    try:
        # Use tsne_color to get RGB visualization
        rgb_tsne = tsne_color(eigvecs)  # Returns RGB image
        
        # Reshape to spatial dimensions
        rgb_tsne_spatial = rgb_tsne.reshape(H, W, 3).numpy()
        
        # Normalize to [0, 1] for display
        rgb_tsne_spatial = (rgb_tsne_spatial - rgb_tsne_spatial.min()) / (rgb_tsne_spatial.max() - rgb_tsne_spatial.min() + 1e-8)
        
    except Exception as e:
        print(f"Error applying tsne_color: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Visualize
    if ax is None:
        plt.figure(figsize=(8, 8))
        plt.imshow(rgb_tsne_spatial)
        plt.title(title)
        plt.axis('off')
    else:
        ax.imshow(rgb_tsne_spatial)
        ax.set_title(title)
        ax.axis('off')
    
    return rgb_tsne_spatial


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
    """Test feature visualization using PCA and ncut"""
    
    print("=" * 60)
    print("Testing Feature Visualization with PCA and NCut")
    print("=" * 60)
    
    # Get the current directory
    repo_path = Path(__file__).parent.absolute()
    
    # Load model
    print("\n[1/4] Loading JAFAR model...")
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
    print("\n[2/4] Loading test image...")
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
    print("\n[3/4] Preparing image and extracting features...")
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
    
    # Extract features
    with torch.no_grad():
        # Get high-resolution features
        hr_feats = jafar_model(image_batch)
        print(f"  High-res features shape: {hr_feats.shape}")
        
        # Also get low-res features for comparison
        lr_feats, _ = jafar_model.backbone(image_batch)
        print(f"  Low-res features shape: {lr_feats.shape}")
    
    print("✓ Features extracted successfully!")
    
    # Visualize features
    print("\n[4/4] Visualizing features...")
    
    # Create output directory
    output_dir = repo_path / "visualizations"
    output_dir.mkdir(exist_ok=True)
    
    # Prepare original image for display (unnormalized)
    img_display = image_batch[0].cpu().clone()
    for t, m, s in zip(img_display, mean.cpu(), std.cpu()):
        t.mul_(s).add_(m)
    img_display = img_display.clamp(0, 1).permute(1, 2, 0).numpy()
    
    # Initialize visualization variables
    umap_lr = None
    umap_hr = None
    tsne_lr = None
    tsne_hr = None
    mspace_lr = None
    mspace_hr = None
    
    # Create figure with subplots
    # Layout: 2 rows x 5 columns (Original, PCA, UMAP, t-SNE, M-Space for low-res and high-res)
    fig = plt.figure(figsize=(25, 10))
    
    # Row 1: Low-res features visualizations
    ax1 = plt.subplot(2, 5, 1)
    plt.imshow(img_display)
    plt.title("Original Image")
    plt.axis('off')
    
    ax2 = plt.subplot(2, 5, 2)
    pca_lr = visualize_features_with_pca(lr_feats[0], "Low-Res PCA", ax=ax2)
    
    ax3 = plt.subplot(2, 5, 3)
    if NCUT_AVAILABLE:
        umap_lr = visualize_features_with_umap_color(lr_feats[0], n_eig=20, title="Low-Res UMAP Color", ax=ax3)
    else:
        ax3.text(0.5, 0.5, "NCut not available", ha='center', va='center', transform=ax3.transAxes)
        ax3.axis('off')
    
    ax4 = plt.subplot(2, 5, 4)
    if NCUT_AVAILABLE:
        tsne_lr = visualize_features_with_tsne_color(lr_feats[0], n_eig=20, title="Low-Res t-SNE Color", ax=ax4)
    else:
        ax4.text(0.5, 0.5, "NCut not available", ha='center', va='center', transform=ax4.transAxes)
        ax4.axis('off')
    
    ax5 = plt.subplot(2, 5, 5)
    if NCUT_AVAILABLE:
        mspace_lr = visualize_features_with_mspace_color(lr_feats[0], n_eig=20, title="Low-Res M-Space Color", ax=ax5)
    else:
        ax5.text(0.5, 0.5, "NCut not available", ha='center', va='center', transform=ax5.transAxes)
        ax5.axis('off')
    
    # Row 2: High-res features visualizations
    ax6 = plt.subplot(2, 5, 6)
    plt.imshow(img_display)
    plt.title("Original Image")
    plt.axis('off')
    
    ax7 = plt.subplot(2, 5, 7)
    pca_hr = visualize_features_with_pca(hr_feats[0], "High-Res PCA", ax=ax7)
    
    ax8 = plt.subplot(2, 5, 8)
    if NCUT_AVAILABLE:
        umap_hr = visualize_features_with_umap_color(hr_feats[0], n_eig=20, title="High-Res UMAP Color", ax=ax8)
    else:
        ax8.text(0.5, 0.5, "NCut not available", ha='center', va='center', transform=ax8.transAxes)
        ax8.axis('off')
    
    ax9 = plt.subplot(2, 5, 9)
    if NCUT_AVAILABLE:
        tsne_hr = visualize_features_with_tsne_color(hr_feats[0], n_eig=20, title="High-Res t-SNE Color", ax=ax9)
    else:
        ax9.text(0.5, 0.5, "NCut not available", ha='center', va='center', transform=ax9.transAxes)
        ax9.axis('off')
    
    ax10 = plt.subplot(2, 5, 10)
    if NCUT_AVAILABLE:
        mspace_hr = visualize_features_with_mspace_color(hr_feats[0], n_eig=20, title="High-Res M-Space Color", ax=ax10)
    else:
        ax10.text(0.5, 0.5, "NCut not available", ha='center', va='center', transform=ax10.transAxes)
        ax10.axis('off')
    
    # Save the combined visualization
    save_path = output_dir / "feature_visualization.png"
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"✓ Visualization saved to {save_path}")
    plt.close()
    
    # Also save individual visualizations
    print("\n  Saving individual visualizations...")
    
    # PCA visualizations
    plt.figure(figsize=(8, 8))
    plt.imshow(pca_lr)
    plt.title("Low-Res Features (PCA)")
    plt.axis('off')
    plt.savefig(output_dir / "pca_lowres.png", bbox_inches='tight', dpi=150)
    plt.close()
    
    plt.figure(figsize=(8, 8))
    plt.imshow(pca_hr)
    plt.title("High-Res Features (PCA)")
    plt.axis('off')
    plt.savefig(output_dir / "pca_highres.png", bbox_inches='tight', dpi=150)
    plt.close()
    
    # UMAP color visualizations
    if NCUT_AVAILABLE and umap_lr is not None:
        plt.figure(figsize=(8, 8))
        plt.imshow(umap_lr)
        plt.title("Low-Res Features (UMAP Color)")
        plt.axis('off')
        plt.savefig(output_dir / "umap_color_lowres.png", bbox_inches='tight', dpi=150)
        plt.close()
        
        plt.figure(figsize=(8, 8))
        plt.imshow(umap_hr)
        plt.title("High-Res Features (UMAP Color)")
        plt.axis('off')
        plt.savefig(output_dir / "umap_color_highres.png", bbox_inches='tight', dpi=150)
        plt.close()
        print("✓ UMAP color visualizations saved")
    
    # t-SNE color visualizations
    if NCUT_AVAILABLE and tsne_lr is not None:
        plt.figure(figsize=(8, 8))
        plt.imshow(tsne_lr)
        plt.title("Low-Res Features (t-SNE Color)")
        plt.axis('off')
        plt.savefig(output_dir / "tsne_color_lowres.png", bbox_inches='tight', dpi=150)
        plt.close()
        
        plt.figure(figsize=(8, 8))
        plt.imshow(tsne_hr)
        plt.title("High-Res Features (t-SNE Color)")
        plt.axis('off')
        plt.savefig(output_dir / "tsne_color_highres.png", bbox_inches='tight', dpi=150)
        plt.close()
        print("✓ t-SNE color visualizations saved")
    
    # M-space color visualizations
    if NCUT_AVAILABLE and mspace_lr is not None:
        plt.figure(figsize=(8, 8))
        plt.imshow(mspace_lr)
        plt.title("Low-Res Features (M-Space Color)")
        plt.axis('off')
        plt.savefig(output_dir / "mspace_color_lowres.png", bbox_inches='tight', dpi=150)
        plt.close()
        
        plt.figure(figsize=(8, 8))
        plt.imshow(mspace_hr)
        plt.title("High-Res Features (M-Space Color)")
        plt.axis('off')
        plt.savefig(output_dir / "mspace_color_highres.png", bbox_inches='tight', dpi=150)
        plt.close()
        print("✓ M-space color visualizations saved")
    
    print("\n✓ All visualizations completed!")
    print(f"\n  Summary:")
    print(f"    Input image: {image_batch.shape}")
    print(f"    Low-res features: {lr_feats.shape}")
    print(f"    High-res features: {hr_feats.shape}")
    print(f"    Visualizations saved to: {output_dir}")
    
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("JAFAR Feature Visualization Test")
    print("=" * 60)
    
    success = test_feature_visualization()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ Test completed successfully!")
    else:
        print("✗ Test failed!")
    print("=" * 60 + "\n")

