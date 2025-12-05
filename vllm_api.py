"""
HTTP client for vLLM-Omni text-to-image API.

Implements OpenAI DALL-E compatible API calls to vLLM-Omni server.
"""

import aiohttp
from typing import Dict, Any


class VLLMOmniClient:
    """
    Async HTTP client for vLLM-Omni text-to-image API.

    The client communicates with a vLLM-Omni server running the image generation
    endpoint (typically at http://localhost:8000/v1/images/generations).

    API format is OpenAI DALL-E compatible.
    """

    def __init__(self, base_url: str, timeout: float = 300.0):
        """
        Initialize client.

        Args:
            base_url: Full URL to vLLM-Omni endpoint
            timeout: Request timeout in seconds (default 5 minutes for slow diffusion)
        """
        self.base_url = base_url
        self.timeout = aiohttp.ClientTimeout(total=timeout)

    async def generate_images(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        n: int = 1,
        seed: int = 0,
    ) -> Dict[str, Any]:
        """
        Generate images via vLLM-Omni API.

        This method accepts ComfyUI-style parameters (width, height) and converts
        them to OpenAI DALL-E format (size string) for the API request.

        Args:
            prompt: Text prompt for image generation
            negative_prompt: Negative prompt (optional)
            width: Image width in pixels
            height: Image height in pixels
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale (classifier-free guidance)
            n: Number of images to generate
            seed: Random seed (0 for random)

        Returns:
            Dict containing API response with 'data' array and 'created' timestamp

        Raises:
            RuntimeError: On network errors or connection failures
            ValueError: On invalid API response or non-200 status codes
        """
        # Convert width/height to OpenAI size format
        size = f"{width}x{height}"

        # Build OpenAI DALL-E compatible request
        request_data = {
            "prompt": prompt,
            "n": n,
            "size": size,
            "response_format": "b64_json",
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        }

        # Only include seed if non-zero (0 means random)
        if seed != 0:
            request_data["seed"] = seed

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            try:
                async with session.post(
                    self.base_url,
                    json=request_data,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    # Check status code
                    if response.status != 200:
                        error_text = await response.text()
                        raise ValueError(
                            f"vLLM-Omni API returned status {response.status}: {error_text}"
                        )

                    # Parse JSON response
                    try:
                        data = await response.json()
                    except aiohttp.ContentTypeError as e:
                        raise ValueError(f"Invalid JSON response from vLLM-Omni: {e}")

                    # Validate response structure (OpenAI DALL-E format)
                    if "data" not in data:
                        raise ValueError(
                            "API response missing 'data' field - expected OpenAI DALL-E format"
                        )

                    if not data["data"]:
                        raise ValueError("API returned empty data array")

                    # Validate each image has b64_json field
                    for idx, img in enumerate(data["data"]):
                        if "b64_json" not in img:
                            raise ValueError(f"Image {idx} missing 'b64_json' field")

                    return data

            except aiohttp.ClientError as e:
                raise RuntimeError(
                    f"Network error connecting to vLLM-Omni at {self.base_url}: {e}"
                )
