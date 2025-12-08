"""
ComfyUI node for vLLM-Omni text-to-image generation.

This module provides a custom ComfyUI node that enables image generation
using vLLM-Omni's diffusion backend via HTTP API.
"""

import torch
from typing import Tuple, Optional

from .vllm_api import VLLMOmniClient
from .utils import base64_to_image_tensor, image_tensor_to_png_bytes


class VLLMTextToImage:
    """
    ComfyUI node for generating images using vLLM-Omni's diffusion backend.

    Connects to a running vLLM-Omni server and generates images from text prompts.
    The server should be running with the image generation endpoint:
        python -m vllm_omni.entrypoints.openai.serving_image --model Qwen/Qwen-Image

    Node Category: image/generation/vllm-omni
    """

    @classmethod
    def INPUT_TYPES(cls):
        """
        Define input parameters for the node.

        Returns ComfyUI-compatible input type dictionary with required and optional fields.
        """
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text description of the image to generate",
                    },
                ),
            },
            "optional": {
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Negative prompt to guide what NOT to generate",
                    },
                ),
                "width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 256,
                        "max": 2048,
                        "step": 64,
                        "tooltip": "Image width in pixels",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 256,
                        "max": 2048,
                        "step": 64,
                        "tooltip": "Image height in pixels",
                    },
                ),
                "num_inference_steps": (
                    "INT",
                    {
                        "default": 50,
                        "min": 1,
                        "max": 200,
                        "step": 1,
                        "tooltip": "Number of denoising steps (higher = better quality, slower)",
                    },
                ),
                "guidance_scale": (
                    "FLOAT",
                    {
                        "default": 4.0,
                        "min": 0.0,
                        "max": 20.0,
                        "step": 0.5,
                        "tooltip": "Classifier-free guidance scale (higher = more prompt adherence)",
                    },
                ),
                "n": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 10,
                        "step": 1,
                        "tooltip": "Number of images to generate",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2**31 - 1,
                        "step": 1,
                        "tooltip": "Random seed for reproducibility (0 = random)",
                    },
                ),
                "server_base_url": (
                    "STRING",
                    {
                        "default": "http://localhost:8000",
                        "tooltip": "Base URL of vLLM-Omni server (e.g., http://10.14.216.12:8000)",
                    },
                ),
                "endpoint_path": (
                    "STRING",
                    {
                        "default": "/v1/images/generations",
                        "tooltip": "API endpoint path (usually /v1/images/generations)",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate"
    CATEGORY = "image/generation/vllm-omni"
    DESCRIPTION = "Generate images using vLLM-Omni's diffusion models (Qwen-Image)"

    async def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        n: int = 1,
        seed: int = 0,
        server_base_url: str = "http://localhost:8000",
        endpoint_path: str = "/v1/images/generations",
    ) -> Tuple[torch.Tensor]:
        """
        Main execution method - generates images via vLLM-Omni API.

        Modern ComfyUI supports async node functions natively, so this method
        is async and will be awaited by ComfyUI's execution system.

        Model Awareness:
        This method implements basic model awareness by querying the server's
        /health endpoint to detect which model is running and applying appropriate
        defaults:

        - Qwen/Qwen-Image: 50 steps, guidance 4.0
        - Z-Image-Turbo: 9 steps, guidance 0.0

        Model defaults are ONLY applied if the parameter matches the UI default.
        If you manually adjust any parameter, your value is always respected.

        If /health endpoint is unavailable, falls back to Qwen-Image defaults.

        Args:
            prompt: Text prompt for image generation
            negative_prompt: Negative prompt (optional)
            width: Image width in pixels
            height: Image height in pixels
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale
            n: Number of images to generate
            seed: Random seed
            server_base_url: Base URL of vLLM-Omni server (e.g., http://localhost:8000)
            endpoint_path: API endpoint path (default: /v1/images/generations)

        Returns:
            Tuple containing a single tensor with shape (n, height, width, 4)

        Raises:
            ValueError: If prompt is empty or invalid
            RuntimeError: If generation fails
        """
        # Validate prompt
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        # Validate and construct full endpoint URL
        if endpoint_path and not endpoint_path.startswith('/'):
            endpoint_path = '/' + endpoint_path

        full_endpoint_url = server_base_url.rstrip('/') + endpoint_path

        # Create API client
        client = VLLMOmniClient(full_endpoint_url)

        # Query model-specific defaults
        model_defaults = await client.get_model_defaults()

        # Display detected model information
        model_name = model_defaults["model_name"]
        if "unknown" not in model_name.lower() and "fallback" not in model_name.lower():
            print(f"\n{'='*60}")
            print(f"vLLM-Omni Server Connected")
            print(f"{'='*60}")
            print(f"Model: {model_name}")
            print(f"Default Steps: {model_defaults['num_inference_steps']}")
            print(f"Guidance Scale: {model_defaults['guidance_scale']}")
            print(f"{'='*60}\n")
        elif "fallback" in model_name.lower():
            print(f"\n⚠️  vLLM-Omni: Using fallback defaults (server /health unavailable)")
            print(f"   Using Qwen-Image defaults: 50 steps, guidance 4.0\n")

        # Apply model defaults if parameters match UI defaults
        # UI defaults: num_inference_steps=50, guidance_scale=4.0
        adjusted_steps = num_inference_steps
        adjusted_guidance = guidance_scale

        if num_inference_steps == 50:  # Matches UI default
            adjusted_steps = model_defaults["num_inference_steps"]
            if adjusted_steps != 50:
                print(f"vLLM-Omni: Using model default steps: {adjusted_steps} "
                      f"(model: {model_defaults['model_name']})")

        if guidance_scale == 4.0:  # Matches UI default
            adjusted_guidance = model_defaults["guidance_scale"]
            if adjusted_guidance != 4.0:
                print(f"vLLM-Omni: Using model default guidance: {adjusted_guidance} "
                      f"(model: {model_defaults['model_name']})")

        try:
            # Generate images via API with adjusted params
            response_data = await client.generate_images(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=adjusted_steps,      # Use adjusted value
                guidance_scale=adjusted_guidance,         # Use adjusted value
                n=n,
                seed=seed,
            )

            # Extract and convert images from response
            # Response format: {"created": ..., "data": [{"b64_json": "..."}, ...]}
            image_tensors = []
            for img_data in response_data["data"]:
                base64_str = img_data["b64_json"]
                tensor = base64_to_image_tensor(base64_str)
                image_tensors.append(tensor)

            # Concatenate all images into batch tensor
            # Individual tensors are (1, H, W, C), concatenate along batch dim
            batch_tensor = torch.cat(image_tensors, dim=0)

            # Return as tuple for ComfyUI (expects tuple of outputs)
            return (batch_tensor,)

        except Exception as e:
            # Re-raise with context for better error messages
            raise RuntimeError(f"vLLM-Omni generation failed: {str(e)}")


class VLLMImageEdit:
    """
    ComfyUI node for editing images using vLLM-Omni's image editing backend.

    Connects to a running vLLM-Omni server and edits images based on text prompts.
    The server should be running with the image editing endpoint:
        python -m vllm_omni.entrypoints.openai.serving_image --model Qwen/Qwen-Image-Edit

    Node Category: image/editing/vllm-omni
    """

    @classmethod
    def INPUT_TYPES(cls):
        """
        Define input parameters for the node.

        Returns ComfyUI-compatible input type dictionary with required and optional fields.
        """
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "Input image to edit",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text instruction describing the edit to perform",
                    },
                ),
            },
            "optional": {
                "mask": (
                    "IMAGE",
                    {
                        "tooltip": "Optional mask for inpainting (white areas will be edited). Currently not supported by server but included for future compatibility.",
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Negative prompt to guide what NOT to generate",
                    },
                ),
                "width": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2048,
                        "step": 64,
                        "tooltip": "Output image width in pixels (0 = auto-calculate from input aspect ratio)",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2048,
                        "step": 64,
                        "tooltip": "Output image height in pixels (0 = auto-calculate from input aspect ratio)",
                    },
                ),
                "num_inference_steps": (
                    "INT",
                    {
                        "default": 50,
                        "min": 1,
                        "max": 200,
                        "step": 1,
                        "tooltip": "Number of denoising steps (higher = better quality, slower)",
                    },
                ),
                "guidance_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 20.0,
                        "step": 0.1,
                        "tooltip": "Standard CFG scale",
                    },
                ),
                "true_cfg_scale": (
                    "FLOAT",
                    {
                        "default": 4.0,
                        "min": 0.0,
                        "max": 20.0,
                        "step": 0.1,
                        "tooltip": "True CFG scale for advanced control",
                    },
                ),
                "n": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 10,
                        "step": 1,
                        "tooltip": "Number of edited variations to generate",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2**31 - 1,
                        "step": 1,
                        "tooltip": "Random seed for reproducibility (0 = random)",
                    },
                ),
                "server_base_url": (
                    "STRING",
                    {
                        "default": "http://localhost:8000",
                        "tooltip": "Base URL of vLLM-Omni server (e.g., http://10.14.216.12:8000)",
                    },
                ),
                "endpoint_path": (
                    "STRING",
                    {
                        "default": "/v1/images/edits",
                        "tooltip": "API endpoint path (usually /v1/images/edits)",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("edited_image",)
    FUNCTION = "edit"
    CATEGORY = "image/editing/vllm-omni"
    DESCRIPTION = "Edit images using vLLM-Omni's image editing models (Qwen-Image-Edit)"

    async def edit(
        self,
        image: torch.Tensor,
        prompt: str,
        mask: Optional[torch.Tensor] = None,
        negative_prompt: str = "",
        width: int = 0,
        height: int = 0,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        true_cfg_scale: float = 4.0,
        n: int = 1,
        seed: int = 0,
        server_base_url: str = "http://localhost:8000",
        endpoint_path: str = "/v1/images/edits",
    ) -> Tuple[torch.Tensor]:
        """
        Main execution method - edits images via vLLM-Omni API.

        Modern ComfyUI supports async node functions natively, so this method
        is async and will be awaited by ComfyUI's execution system.

        Args:
            image: Input image tensor to edit (B, H, W, C)
            prompt: Text instruction for editing
            mask: Optional mask tensor (B, H, W, C) for inpainting
            negative_prompt: Negative prompt (optional)
            width: Output width (0 = auto-calculate from input)
            height: Output height (0 = auto-calculate from input)
            num_inference_steps: Number of denoising steps
            guidance_scale: Standard CFG scale
            true_cfg_scale: True CFG scale
            n: Number of variations to generate
            seed: Random seed
            server_base_url: Base URL of vLLM-Omni server
            endpoint_path: API endpoint path

        Returns:
            Tuple containing a single tensor with shape (n, height, width, channels)

        Raises:
            ValueError: If prompt is empty, size parameters invalid, or image conversion fails
            RuntimeError: If editing fails
        """
        # Validate prompt
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        # Validate and construct full endpoint URL
        if endpoint_path and not endpoint_path.startswith('/'):
            endpoint_path = '/' + endpoint_path

        full_endpoint_url = server_base_url.rstrip('/') + endpoint_path

        # Validate width/height combination
        # Both must be 0 (auto) or both must be > 0 (explicit size)
        size_param = None
        if width > 0 and height > 0:
            # Both specified - use exact size
            size_param = f"{width}x{height}"
        elif width > 0 or height > 0:
            # Mixed (one zero, one non-zero) - error
            raise ValueError(
                "Width and height must both be 0 (auto) or both be non-zero. "
                f"Got width={width}, height={height}"
            )
        # else: both zero - size_param stays None, server auto-calculates

        # Convert input image tensor to PNG BytesIO
        try:
            image_bytes = image_tensor_to_png_bytes(image, filename="image.png")
        except Exception as e:
            raise ValueError(f"Failed to convert input image to PNG: {e}")

        # Convert mask if provided
        mask_bytes = None
        if mask is not None:
            try:
                mask_bytes = image_tensor_to_png_bytes(mask, filename="mask.png")
            except Exception as e:
                raise ValueError(f"Failed to convert mask to PNG: {e}")

        # Create API client
        client = VLLMOmniClient(full_endpoint_url)

        try:
            # Edit images via API
            response_data = await client.edit_image(
                image_bytes=image_bytes,
                prompt=prompt,
                mask_bytes=mask_bytes,
                negative_prompt=negative_prompt,
                size=size_param,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                true_cfg_scale=true_cfg_scale,
                n=n,
                seed=seed,
            )

            # Extract and convert images from response
            # Response format: {"created": ..., "data": [{"b64_json": "..."}, ...]}
            image_tensors = []
            for img_data in response_data["data"]:
                base64_str = img_data["b64_json"]
                tensor = base64_to_image_tensor(base64_str)
                image_tensors.append(tensor)

            # Concatenate all images into batch tensor
            # Individual tensors are (1, H, W, C), concatenate along batch dim
            batch_tensor = torch.cat(image_tensors, dim=0)

            # Return as tuple for ComfyUI (expects tuple of outputs)
            return (batch_tensor,)

        except Exception as e:
            # Re-raise with context for better error messages
            raise RuntimeError(f"vLLM-Omni image editing failed: {str(e)}")
