"""
PyTorch Hub configuration for JAFAR models.

This file allows loading JAFAR models using torch.hub.load().

Example usage:
    import torch
    
    # Load JAFAR with DINOv2-S backbone (output resolution: 512)
    jafar, backbone = torch.hub.load('path/to/JAFAR', 'jafar_dinov2_s', pretrained=True, output_resolution=512)
    
    # Use the model - JAFAR supports any output resolution
    with torch.no_grad():
        lr_feats, _ = backbone(image_batch)
        # Use output resolution
        hr_feats = jafar(image_batch, lr_feats, (jafar.output_resolution, jafar.output_resolution))
        # Or specify any custom resolution
        hr_feats_1024 = jafar(image_batch, lr_feats, (1024, 1024))
"""

import os
import torch
from torch import nn
from pathlib import Path
import urllib.request

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Import required modules
from src.upsampler import JAFAR
from src.backbone.vit_wrapper import PretrainedViTWrapper
from src.backbone.radio import RadioWrapper
from hydra_plugins.resolvers import get_feature


# Mapping from backbone names to their download URLs
WEIGHTS_URLS = {
    'vit_base_patch16_224': 'https://github.com/PaulCouairon/JAFAR/releases/download/Weights/vit_base_patch16_224.pth',
    'vit_base_patch16_224.dino': 'https://github.com/PaulCouairon/JAFAR/releases/download/Weights/vit_base_patch16_224.dino.pth',
    'vit_small_patch14_dinov2.lvd142m': 'https://github.com/PaulCouairon/JAFAR/releases/download/Weights/vit_small_patch14_dinov2.lvd142m.pth',
    'vit_base_patch14_dinov2.lvd142m': 'https://github.com/PaulCouairon/JAFAR/releases/download/Weights/vit_base_patch14_dinov2.lvd142m.pth',
    'vit_small_patch14_reg4_dinov2': 'https://github.com/PaulCouairon/JAFAR/releases/download/Weights/vit_small_patch14_reg4_dinov2.pth',
    'vit_small_patch16_dinov3.lvd1689m': 'https://github.com/PaulCouairon/JAFAR/releases/download/Weights/vit_small_patch16_dinov3.lvd1689m.pth',
    'vit_small_plus_patch16_dinov3.lvd1689m': 'https://github.com/PaulCouairon/JAFAR/releases/download/Weights/vit_small_plus_patch16_dinov3.lvd1689m.pth',
    'vit_base_patch16_dinov3.lvd1689m': 'https://github.com/PaulCouairon/JAFAR/releases/download/Weights/vit_base_patch16_dinov3.lvd1689m.pth',
    'vit_large_patch16_dinov3.lvd1689m': 'https://github.com/PaulCouairon/JAFAR/releases/download/Weights/vit_large_patch16_dinov3.lvd1689m.pth',
    'vit_base_patch16_clip_384': 'https://github.com/PaulCouairon/JAFAR/releases/download/Weights/vit_base_patch16_clip_384.pth',
    'vit_base_patch16_siglip_512.v2_webli': 'https://github.com/PaulCouairon/JAFAR/releases/download/Weights/vit_base_patch16_siglip_512.v2_webli.pth',
    'radio_v2.5-b': 'https://github.com/PaulCouairon/JAFAR/releases/download/Weights/radio_v2.5-b.pth',
    'radio_v2.5-l': 'https://github.com/PaulCouairon/JAFAR/releases/download/Weights/radio_v2.5-l.pth',
    'radio_v2.5-h': 'https://github.com/PaulCouairon/JAFAR/releases/download/Weights/radio_v2.5-h.pth',
}


def _get_cache_dir():
    """Get the cache directory for JAFAR weights."""
    cache_dir = Path.home() / '.cache' / 'torch' / 'hub' / 'checkpoints' / 'jafar'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _download_file(url, destination, progress=True):
    """
    Download a file from URL to destination with progress bar.
    
    Args:
        url: URL to download from
        destination: Path where file should be saved
        progress: Whether to show progress bar
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    if progress and HAS_TQDM:
        def _progress_hook(count, block_size, total_size):
            if not hasattr(_progress_hook, 'pbar'):
                _progress_hook.pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc=destination.name)
            _progress_hook.pbar.update(block_size)
        
        try:
            urllib.request.urlretrieve(url, destination, _progress_hook)
            if hasattr(_progress_hook, 'pbar'):
                _progress_hook.pbar.close()
        except Exception as e:
            if hasattr(_progress_hook, 'pbar'):
                _progress_hook.pbar.close()
            raise e
    else:
        urllib.request.urlretrieve(url, destination)
    
    return destination


def _get_weights_path(backbone_name, weights_path=None):
    """
    Get the path to weights file, downloading if necessary.
    
    Args:
        backbone_name: Name of the backbone model
        weights_path: Optional custom path to weights file
    
    Returns:
        Path to weights file
    """
    # If custom path provided, use it
    if weights_path is not None:
        weights_path = Path(weights_path)
        if weights_path.exists():
            return weights_path
        else:
            raise FileNotFoundError(f"Weights not found at {weights_path}")
    
    # Use cache directory
    cache_dir = _get_cache_dir()
    weights_file = cache_dir / f"{backbone_name}.pth"
    
    # If weights exist in cache, return them
    if weights_file.exists():
        return weights_file
    
    # Download weights if URL is available
    if backbone_name in WEIGHTS_URLS:
        url = WEIGHTS_URLS[backbone_name]
        print(f"Downloading weights for {backbone_name} from {url}...")
        try:
            _download_file(url, weights_file, progress=True)
            print(f"Weights saved to {weights_file}")
            return weights_file
        except Exception as e:
            import warnings
            warnings.warn(
                f"Failed to download weights for {backbone_name}: {e}. "
                f"Please download manually from: {url}"
            )
            raise FileNotFoundError(f"Weights not found and download failed: {weights_file}")
    else:
        raise ValueError(
            f"No weights URL available for backbone '{backbone_name}'. "
            f"Available backbones: {list(WEIGHTS_URLS.keys())}"
        )


def _load_jafar_with_backbone(backbone_name, pretrained=True, device='cuda', weights_path=None, output_resolution=512):
    """
    Helper function to load JAFAR model with a given backbone.
    
    Args:
        backbone_name: Name of the backbone model
        pretrained: Whether to load pre-trained weights
        device: Device to load the model on
        weights_path: Optional path to weights file. If None, uses default path.
        output_resolution: Output resolution for upsampling (default: 512)
    
    Returns:
        Tuple of (jafar_model, backbone_model)
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Determine if it's a Radio backbone
    is_radio = 'radio' in backbone_name.lower()
    
    # Instantiate backbone
    if is_radio:
        backbone = RadioWrapper(name=backbone_name, device=device)
    else:
        backbone = PretrainedViTWrapper(name=backbone_name)
        backbone = backbone.to(device)
    
    # Get feature dimension
    feature_dim = get_feature(backbone_name)
    
    # Instantiate JAFAR model
    jafar = JAFAR(
        input_dim=3,
        qk_dim=128,
        v_dim=feature_dim,
        feature_dim=feature_dim,
        kernel_size=1,
        num_heads=4,
    )
    jafar = jafar.to(device)
    jafar.eval()
    
    # Store output resolution as an attribute for convenience
    jafar.output_resolution = output_resolution
    
    # Load pre-trained weights if requested
    if pretrained:
        try:
            weights_path = _get_weights_path(backbone_name, weights_path)
            checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
            if 'jafar' in checkpoint:
                jafar.load_state_dict(checkpoint['jafar'], strict=False)
            else:
                jafar.load_state_dict(checkpoint, strict=False)
        except (FileNotFoundError, ValueError) as e:
            import warnings
            warnings.warn(
                f"Could not load pre-trained weights: {e}. "
                f"Loading model without pre-trained weights."
            )
    
    return jafar, backbone


# ============ Simplified Interface ============

class JAFARModel(nn.Module):
    """
    Wrapper class for JAFAR that simplifies usage.
    
    Usage:
        from hubconf import dinov3_l
        jafar = dinov3_l(resolution=512)
        hr_feats = jafar(image_batch)
    """
    
    def __init__(self, jafar_model, backbone, output_resolution=512):
        """
        Initialize JAFAR wrapper.
        
        Args:
            jafar_model: The JAFAR upsampler model
            backbone: The backbone model for feature extraction
            output_resolution: Default output resolution for upsampling
        """
        super().__init__()
        self.jafar = jafar_model
        self.backbone = backbone
        self.output_resolution = output_resolution
        self.device = next(jafar_model.parameters()).device
        
        # Set models to eval mode
        self.jafar.eval()
        self.backbone.eval()
    
    def __call__(self, image_batch, output_resolution=None):
        """
        Forward pass: extract features and upsample.
        
        Args:
            image_batch: Input image tensor of shape (B, C, H, W)
            output_resolution: Output resolution (H, W) or int for square. 
                              If None, uses self.output_resolution
        
        Returns:
            High-resolution features tensor
        """
        if output_resolution is None:
            output_resolution = self.output_resolution
        
        # Convert int to tuple if needed
        if isinstance(output_resolution, int):
            output_resolution = (output_resolution, output_resolution)
        
        with torch.no_grad():
            # Extract low-resolution features from backbone
            lr_feats, _ = self.backbone(image_batch)
            
            # Upsample features using JAFAR
            hr_feats = self.jafar(image_batch, lr_feats, output_resolution)
        
        return hr_feats
    
    def to(self, device):
        """Move model to device."""
        self.jafar = self.jafar.to(device)
        self.backbone = self.backbone.to(device)
        self.device = device
        return self
    
    def eval(self):
        """Set models to evaluation mode."""
        self.jafar.eval()
        self.backbone.eval()
        return self
    
    def train(self):
        """Set models to training mode."""
        self.jafar.train()
        self.backbone.train()
        return self


def dinov2_s(pretrained=True, device='cuda', weights_path=None, output_resolution=512):
    """JAFAR model with DINOv2-S (ViT-S/14) backbone."""
    return load('dinov2_s', resolution=output_resolution, pretrained=pretrained, device=device, weights_path=weights_path)


def dinov2_b(pretrained=True, device='cuda', weights_path=None, output_resolution=512):
    """JAFAR model with DINOv2-B (ViT-B/14) backbone."""
    return load('dinov2_b', resolution=output_resolution, pretrained=pretrained, device=device, weights_path=weights_path)


def dinov2_reg4_s(pretrained=True, device='cuda', weights_path=None, output_resolution=512):
    """JAFAR model with DINOv2-Reg4-S (ViT-S-Reg4/14) backbone."""
    return load('dinov2_reg4_s', resolution=output_resolution, pretrained=pretrained, device=device, weights_path=weights_path)


def dinov3_s(pretrained=True, device='cuda', weights_path=None, output_resolution=512):
    """JAFAR model with DINOv3-S (ViT-S/16) backbone."""
    return load('dinov3_s', resolution=output_resolution, pretrained=pretrained, device=device, weights_path=weights_path)


def dinov3_s_plus(pretrained=True, device='cuda', weights_path=None, output_resolution=512):
    """JAFAR model with DINOv3-S+ (ViT-S+/16) backbone."""
    return load('dinov3_s_plus', resolution=output_resolution, pretrained=pretrained, device=device, weights_path=weights_path)


def dinov3_b(pretrained=True, device='cuda', weights_path=None, output_resolution=512):
    """JAFAR model with DINOv3-B (ViT-B/16) backbone."""
    return load('dinov3_b', resolution=output_resolution, pretrained=pretrained, device=device, weights_path=weights_path)


def dinov3_l(pretrained=True, device='cuda', weights_path=None, output_resolution=512):
    """JAFAR model with DINOv3-L (ViT-L/16) backbone."""
    return load('dinov3_l', resolution=output_resolution, pretrained=pretrained, device=device, weights_path=weights_path)


def dino_b(pretrained=True, device='cuda', weights_path=None, output_resolution=512):
    """JAFAR model with DINO-B (ViT-B/16) backbone."""
    return load('dino_b', resolution=output_resolution, pretrained=pretrained, device=device, weights_path=weights_path)


def clip_b(pretrained=True, device='cuda', weights_path=None, output_resolution=512):
    """JAFAR model with CLIP-B (ViT-B/16) backbone."""
    return load('clip_b', resolution=output_resolution, pretrained=pretrained, device=device, weights_path=weights_path)


def siglip2_b(pretrained=True, device='cuda', weights_path=None, output_resolution=512):
    """JAFAR model with SigLIP2-B (ViT-B/16) backbone."""
    return load('siglip2_b', resolution=output_resolution, pretrained=pretrained, device=device, weights_path=weights_path)


def vit_b(pretrained=True, device='cuda', weights_path=None, output_resolution=512):
    """JAFAR model with ViT-B/16 backbone."""
    return load('vit_b', resolution=output_resolution, pretrained=pretrained, device=device, weights_path=weights_path)


def radio_b(pretrained=True, device='cuda', weights_path=None, output_resolution=512):
    """JAFAR model with RADIO v2.5-B backbone."""
    return load('radio_b', resolution=output_resolution, pretrained=pretrained, device=device, weights_path=weights_path)


def radio_l(pretrained=True, device='cuda', weights_path=None, output_resolution=512):
    """JAFAR model with RADIO v2.5-L backbone."""
    return load('radio_l', resolution=output_resolution, pretrained=pretrained, device=device, weights_path=weights_path)


def radio_h(pretrained=True, device='cuda', weights_path=None, output_resolution=512):
    """JAFAR model with RADIO v2.5-H backbone."""
    return load('radio_h', resolution=output_resolution, pretrained=pretrained, device=device, weights_path=weights_path)


def jafar(backbone_name, pretrained=True, device='cuda', weights_path=None, output_resolution=512):
    """
    Generic JAFAR model loader for any supported backbone.
    
    Args:
        backbone_name: Name of the backbone model (e.g., 'vit_small_patch14_dinov2.lvd142m')
        pretrained: If True, loads pre-trained weights
        device: Device to load the model on ('cuda' or 'cpu')
        weights_path: Optional path to weights file
        output_resolution: Output resolution for upsampling (default: 512)
    
    Returns:
        Tuple of (jafar_model, backbone_model)
    
    Example:
        jafar, backbone = torch.hub.load('path/to/JAFAR', 'jafar', 
                                         backbone_name='vit_small_patch14_dinov2.lvd142m',
                                         output_resolution=512)
    """
    return _load_jafar_with_backbone(
        backbone_name,
        pretrained=pretrained,
        device=device,
        weights_path=weights_path,
        output_resolution=output_resolution
    )


def load(name, resolution=512, pretrained=True, device=None, weights_path=None):
    """
    Load a JAFAR model with simplified interface.
    
    Args:
        name: Model name (e.g., 'dinov3_l', 'dinov2_s', 'dinov3_b', etc.)
              or full backbone name (e.g., 'vit_large_patch16_dinov3.lvd1689m')
        resolution: Output resolution (default: 512)
        pretrained: Whether to load pretrained weights (default: True)
        device: Device to load on ('cuda' or 'cpu'). If None, auto-detects.
        weights_path: Optional path to weights file
    
    Returns:
        JAFARModel instance
    
    Examples:
        from hubconf import load
        jafar = load('dinov3_l', resolution=512)
        hr_feats = jafar(image_batch)
        
        # Using full backbone name
        jafar = load('vit_small_patch14_dinov2.lvd142m', resolution=1024)
        hr_feats = jafar(image_batch)
        
        # Custom resolution per call
        hr_feats_512 = jafar(image_batch, output_resolution=512)
        hr_feats_1024 = jafar(image_batch, output_resolution=1024)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    name_mapping = {
        'dinov2_s': 'vit_small_patch14_dinov2.lvd142m',
        'dinov2_b': 'vit_base_patch14_dinov2.lvd142m',
        'dinov2_reg4_s': 'vit_small_patch14_reg4_dinov2',
        'dinov3_s': 'vit_small_patch16_dinov3.lvd1689m',
        'dinov3_s_plus': 'vit_small_plus_patch16_dinov3.lvd1689m',
        'dinov3_b': 'vit_base_patch16_dinov3.lvd1689m',
        'dinov3_l': 'vit_large_patch16_dinov3.lvd1689m',
        'dino_b': 'vit_base_patch16_224.dino',
        'clip_b': 'vit_base_patch16_clip_384',
        'siglip2_b': 'vit_base_patch16_siglip_512.v2_webli',
        'vit_b': 'vit_base_patch16_224',
        'radio_b': 'radio_v2.5-b',
        'radio_l': 'radio_v2.5-l',
        'radio_h': 'radio_v2.5-h',
    }
    
    backbone_name = name_mapping.get(name.lower(), name)
    jafar_model, backbone = jafar(
        backbone_name=backbone_name,
        pretrained=pretrained,
        device=device,
        weights_path=weights_path,
        output_resolution=resolution
    )
    
    return JAFARModel(jafar_model, backbone, output_resolution=resolution)

