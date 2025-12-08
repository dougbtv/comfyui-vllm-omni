"""
HTTP client for vLLM-Omni text-to-image API.

Implements OpenAI DALL-E compatible API calls to vLLM-Omni server.
"""

import aiohttp
from io import BytesIO
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

    async def edit_image(
        self,
        image_bytes: BytesIO,
        prompt: str,
        mask_bytes: Optional[BytesIO] = None,
        negative_prompt: str = "",
        size: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        true_cfg_scale: float = 4.0,
        n: int = 1,
        seed: int = 0,
    ) -> Dict[str, Any]:
        """
        Edit images via vLLM-Omni API using multipart form-data.

        This method sends a POST request to the /v1/images/edits endpoint
        with the input image and editing parameters as multipart/form-data.

        Args:
            image_bytes: Input image as PNG BytesIO (must have .name attribute)
            prompt: Text instruction for editing the image
            mask_bytes: Optional mask image as PNG BytesIO (must have .name attribute)
                       Currently not supported by server but accepted for future compatibility
            negative_prompt: Negative prompt to guide what NOT to generate (optional)
            size: Output size as "WIDTHxHEIGHT" string (e.g., "1024x768")
                 If None, server auto-calculates size from input aspect ratio
            num_inference_steps: Number of denoising steps (default: 50)
            guidance_scale: Standard CFG scale (default: 1.0)
            true_cfg_scale: True CFG scale for advanced control (default: 4.0)
            n: Number of edited variations to generate (1-10, default: 1)
            seed: Random seed for reproducibility (0 for random)

        Returns:
            Dict containing API response with 'data' array and 'created' timestamp
            Response format matches generate_images() (OpenAI DALL-E compatible)

        Raises:
            RuntimeError: On network errors or connection failures
            ValueError: On invalid API response or non-200 status codes
        """
        # Build multipart form-data
        form = aiohttp.FormData()

        # Add required image file
        form.add_field(
            'image',
            image_bytes,
            filename='image.png',
            content_type='image/png'
        )

        # Add required text prompt
        form.add_field('prompt', prompt)

        # Add optional fields
        if negative_prompt:
            form.add_field('negative_prompt', negative_prompt)

        if size is not None:
            form.add_field('size', size)

        # Add numeric parameters as strings (form fields are text)
        form.add_field('num_inference_steps', str(num_inference_steps))
        form.add_field('guidance_scale', str(guidance_scale))
        form.add_field('true_cfg_scale', str(true_cfg_scale))
        form.add_field('n', str(n))

        # Only include seed if non-zero (0 means random)
        if seed != 0:
            form.add_field('seed', str(seed))

        # Add optional mask file
        if mask_bytes is not None:
            form.add_field(
                'mask',
                mask_bytes,
                filename='mask.png',
                content_type='image/png'
            )

        # Send request with multipart form-data
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            try:
                async with session.post(self.base_url, data=form) as response:
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

                    # Validate response structure (same as generate_images)
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
