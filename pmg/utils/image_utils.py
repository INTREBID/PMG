"""
Image processing utilities for PMG
"""
import numpy as np
from PIL import Image
import torch
from torchvision import transforms


def resize_rgb(img_path: str, size: int = 512) -> np.ndarray:
    """
    Load and resize an RGB image from path
    
    Args:
        img_path: Path to image file
        size: Target size for both dimensions
        
    Returns:
        Resized image as numpy array (H, W, 3)
    """
    try:
        im = Image.open(img_path).convert('RGB')
        im = im.resize((size, size), Image.BICUBIC)
        arr = np.array(im, dtype=np.uint8)
        return np.ascontiguousarray(arr)
    except Exception as e:
        print(f"Warning: Failed to load image {img_path}: {e}")
        # Return random image as fallback
        return np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)


def image_to_tensor(image: Image.Image, size: int = 512, device: str = 'cuda') -> torch.Tensor:
    """
    Convert PIL Image to PyTorch tensor
    
    Args:
        image: PIL Image
        size: Target size
        device: Device to place tensor on
        
    Returns:
        Tensor in [0, 1] range with shape (1, 3, H, W)
    """
    image = image.resize((size, size), Image.BICUBIC)
    transform = transforms.Compose([
        transforms.ToTensor()  # Converts to [0, 1]
    ])
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """
    Convert PyTorch tensor to PIL Image
    
    Args:
        tensor: Tensor with shape (C, H, W) or (1, C, H, W) in [0, 1] or [-1, 1]
        
    Returns:
        PIL Image
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Normalize to [0, 1]
    if tensor.min() < 0:
        tensor = (tensor + 1) / 2
    tensor = tensor.clamp(0, 1)
    
    # Convert to numpy
    arr = tensor.cpu().numpy().transpose(1, 2, 0)
    arr = (arr * 255).astype(np.uint8)
    
    return Image.fromarray(arr)


def preprocess_image_for_sd(image: Image.Image, size: int = 512) -> torch.Tensor:
    """
    Preprocess image for Stable Diffusion VAE
    
    Args:
        image: PIL Image
        size: Target size
        
    Returns:
        Tensor with shape (1, 3, H, W) in [-1, 1] range
    """
    transform = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])
    tensor = transform(image).unsqueeze(0)
    return tensor


def save_image_grid(images: list, output_path: str, nrow: int = 4):
    """
    Save a grid of images
    
    Args:
        images: List of PIL Images or numpy arrays
        output_path: Path to save the grid
        nrow: Number of images per row
    """
    from torchvision.utils import make_grid, save_image as tv_save_image
    
    # Convert all images to tensors
    tensors = []
    for img in images:
        if isinstance(img, Image.Image):
            tensor = transforms.ToTensor()(img)
        elif isinstance(img, np.ndarray):
            tensor = torch.from_numpy(img).permute(2, 0, 1) / 255.0
        elif isinstance(img, torch.Tensor):
            tensor = img
        else:
            continue
        tensors.append(tensor)
    
    # Create grid
    grid = make_grid(tensors, nrow=nrow, padding=2, normalize=False)
    tv_save_image(grid, output_path)

