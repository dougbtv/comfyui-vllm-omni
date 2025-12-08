"""
ComfyUI-vLLM-Omni custom node package.

Provides text-to-image generation capabilities using vLLM-Omni's diffusion backend.

To use this node:
1. Start a vLLM-Omni image generation server:
   python -m vllm_omni.entrypoints.openai.serving_image --model Qwen/Qwen-Image

2. In ComfyUI, add the "vLLM-Omni Text-to-Image" node from the
   "image/generation/vllm-omni" category

3. Connect it to other nodes (e.g., SaveImage) and generate images!
"""

from .vllm_omni_node import VLLMTextToImage, VLLMImageEdit

# ComfyUI requires these two dictionaries for node registration
NODE_CLASS_MAPPINGS = {
    "VLLMTextToImage": VLLMTextToImage,
    "VLLMImageEdit": VLLMImageEdit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VLLMTextToImage": "vLLM-Omni Text-to-Image",
    "VLLMImageEdit": "vLLM-Omni Image Edit",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
