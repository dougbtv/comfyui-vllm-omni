"""
Image conversion utilities for ComfyUI-vLLM-Omni.

Converts base64-encoded images from vLLM-Omni API to ComfyUI-compatible tensors.
"""

import base64
from io import BytesIO
from typing import Union

import torch
import numpy as np
from PIL import Image


def base64_to_image_tensor(base64_str: str, mode: str = "RGBA") -> torch.Tensor:
    """
    Convert base64-encoded image to ComfyUI image tensor.

    Args:
        base64_str: Base64-encoded image string
        mode: PIL image mode (default RGBA for transparency support)

    Returns:
        torch.Tensor with shape (1, H, W, C) in float32 [0, 1] range

    Raises:
        ValueError: If base64 string is invalid or image cannot be decoded
    """
    try:
        # Decode base64 to bytes
        image_bytes = base64.b64decode(base64_str)
    except Exception as e:
        raise ValueError(f"Invalid base64 string: {e}")

    # Create BytesIO object for PIL
    image_bytesio = BytesIO(image_bytes)

    # Open with PIL and convert to desired mode
    try:
        pil_image = Image.open(image_bytesio)
        pil_image = pil_image.convert(mode)
    except Exception as e:
        raise ValueError(f"Failed to open image: {e}")

    # Convert to numpy array and normalize to [0, 1]
    # This follows the pattern from ComfyUI's nodes_openai.py (lines 93-95)
    image_array = np.asarray(pil_image).astype(np.float32) / 255.0

    # Convert to torch tensor with batch dimension (1, H, W, C)
    # ComfyUI expects shape: (batch, height, width, channels)
    image_tensor = torch.from_numpy(image_array).unsqueeze(0)

    return image_tensor


def validate_image_tensor(tensor: torch.Tensor) -> None:
    """
    Validate that tensor matches ComfyUI image format.

    Expected format: (B, H, W, C) with float32 dtype in [0, 1] range

    Args:
        tensor: Tensor to validate

    Raises:
        ValueError: If tensor format is invalid
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Expected torch.Tensor")

    if tensor.ndim != 4:
        raise ValueError(f"Expected 4D tensor (B,H,W,C), got {tensor.ndim}D")

    if tensor.dtype != torch.float32:
        raise ValueError(f"Expected float32 dtype, got {tensor.dtype}")

    if tensor.min() < 0.0 or tensor.max() > 1.0:
        raise ValueError(
            f"Tensor values must be in [0,1], got [{tensor.min():.3f}, {tensor.max():.3f}]"
        )
