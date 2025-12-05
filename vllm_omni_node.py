"""
ComfyUI node for vLLM-Omni text-to-image generation.

This module provides a custom ComfyUI node that enables image generation
using vLLM-Omni's diffusion backend via HTTP API.
"""

import asyncio
import torch
from typing import Tuple

from .vllm_api import VLLMOmniClient
from .utils import base64_to_image_tensor


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
                "server_url": (
                    "STRING",
                    {
                        "default": "http://localhost:8000/v1/images/generations",
                        "tooltip": "vLLM-Omni server endpoint URL",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate"
    CATEGORY = "image/generation/vllm-omni"
    DESCRIPTION = "Generate images using vLLM-Omni's diffusion models (Qwen-Image)"

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        n: int = 1,
        seed: int = 0,
        server_url: str = "http://localhost:8000/v1/images/generations",
    ) -> Tuple[torch.Tensor]:
        """
        Main execution method - generates images via vLLM-Omni API.

        This is a synchronous wrapper around the async implementation to maintain
        ComfyUI compatibility.

        Args:
            prompt: Text prompt for image generation
            negative_prompt: Negative prompt (optional)
            width: Image width in pixels
            height: Image height in pixels
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale
            n: Number of images to generate
            seed: Random seed
            server_url: Full URL to vLLM-Omni endpoint

        Returns:
            Tuple containing a single tensor with shape (n, height, width, 4)

        Raises:
            ValueError: If prompt is empty or invalid
            RuntimeError: If generation fails
        """
        # Create and run event loop for async execution
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                self._generate_async(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    n=n,
                    seed=seed,
                    server_url=server_url,
                )
            )
            return result
        finally:
            loop.close()

    async def _generate_async(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        n: int,
        seed: int,
        server_url: str,
    ) -> Tuple[torch.Tensor]:
        """
        Async implementation of image generation.

        Steps:
        1. Validate prompt
        2. Create API client and call vLLM-Omni
        3. Parse response data[].b64_json fields
        4. Convert each base64 string to tensor
        5. Concatenate into batch tensor
        6. Return as tuple for ComfyUI

        Args:
            All parameters from generate() method

        Returns:
            Tuple containing batch tensor

        Raises:
            ValueError: If prompt is empty
            RuntimeError: If API call fails or response is invalid
        """
        # Validate prompt
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        # Create API client
        client = VLLMOmniClient(server_url)

        try:
            # Generate images via API
            response_data = await client.generate_images(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
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
