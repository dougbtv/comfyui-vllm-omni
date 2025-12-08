"""
HTTP client for vLLM-Omni text-to-image API.

Implements OpenAI DALL-E compatible API calls to vLLM-Omni server.
"""

import aiohttp
from typing import Dict, Any, Optional


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

        # Cache for health endpoint data (model-aware defaults)
        self._health_cache: Optional[Dict[str, Any]] = None
        self._health_checked: bool = False

    async def get_health(self) -> Optional[Dict[str, Any]]:
        """
        Query /health endpoint to detect model and profile.

        Returns model info with profile (default_steps, max_steps) or None if unavailable.
        Results are cached to avoid repeated calls.
        """
        if self._health_checked:
            return self._health_cache

        self._health_checked = True

        # Parse base_url to construct health URL
        try:
            from urllib.parse import urlparse, urlunparse
            parsed = urlparse(self.base_url)
            health_url = urlunparse((parsed.scheme, parsed.netloc, '/health', '', '', ''))
        except Exception as e:
            print(f"Warning: Failed to parse base_url for health check: {e}")
            return None

        # Query /health with 5-second timeout
        health_timeout = aiohttp.ClientTimeout(total=5.0)

        async with aiohttp.ClientSession(timeout=health_timeout) as session:
            try:
                async with session.get(health_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        self._health_cache = data
                        return data
                    else:
                        print(f"Warning: /health returned status {response.status}")
                        return None
            except Exception as e:
                # Fail silently - server may not support /health
                print(f"Warning: Health check failed (server may not support /health): {e}")
                return None

    async def get_model_defaults(self) -> Dict[str, Any]:
        """
        Get model-specific parameter defaults from health endpoint.

        Returns:
            Dict with keys:
                - num_inference_steps: int
                - guidance_scale: float
                - model_name: str

        Fallback (if /health unavailable):
            Returns Qwen-Image defaults: steps=50, guidance=4.0
        """
        health_info = await self.get_health()

        if health_info is None:
            # Fallback to Qwen-Image defaults
            return {
                "num_inference_steps": 50,
                "guidance_scale": 4.0,
                "model_name": "unknown (fallback)",
            }

        profile = health_info.get("profile", {})
        model_name = health_info.get("model", "unknown")
        default_steps = profile.get("default_steps", 50)

        # Z-Image-Turbo forces guidance_scale to 0.0
        if "z-image-turbo" in model_name.lower():
            guidance_scale = 0.0
        else:
            guidance_scale = 4.0

        return {
            "num_inference_steps": default_steps,
            "guidance_scale": guidance_scale,
            "model_name": model_name,
        }

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
