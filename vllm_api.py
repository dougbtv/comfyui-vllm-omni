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

    async def generate_images(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = -1,
        guidance_scale: float = -1.0,
        true_cfg_scale: float = -1.0,
        n: int = 1,
        seed: int = 0,
        vae_use_slicing: bool = False,
        vae_use_tiling: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate images via vLLM-Omni API.

        This method accepts ComfyUI-style parameters (width, height) and converts
        them to OpenAI DALL-E format (size string) for the API request.

        Parameters set to sentinel values (-1 for int, -1.0 for float) will be
        omitted from the API request, allowing the server to use its own defaults.

        Args:
            prompt: Text prompt for image generation
            negative_prompt: Negative prompt (optional, omitted if empty)
            width: Image width in pixels
            height: Image height in pixels
            num_inference_steps: Number of denoising steps (-1 = server default)
            guidance_scale: CFG scale (-1.0 = server default)
            true_cfg_scale: True CFG scale for advanced control (-1.0 = server default)
            n: Number of images to generate
            seed: Random seed (0 = random)
            vae_use_slicing: Enable VAE slicing for memory optimization
            vae_use_tiling: Enable VAE tiling for large images

        Returns:
            Dict containing API response with 'data' array and 'created' timestamp

        Raises:
            RuntimeError: On network errors or connection failures
            ValueError: On invalid API response or non-200 status codes
        """
        # Convert width/height to OpenAI size format
        size = f"{width}x{height}"

        # Build base request (always included)
        request_data = {
            "prompt": prompt,
            "n": n,
            "size": size,
            "response_format": "b64_json",
        }

        # Only add optional parameters if not at sentinel value or non-empty
        if negative_prompt:
            request_data["negative_prompt"] = negative_prompt

        if num_inference_steps != -1:
            request_data["num_inference_steps"] = num_inference_steps

        if guidance_scale != -1.0:
            request_data["guidance_scale"] = guidance_scale

        if true_cfg_scale != -1.0:
            request_data["true_cfg_scale"] = true_cfg_scale

        # VAE params always sent (booleans, no sentinel concept)
        request_data["vae_use_slicing"] = vae_use_slicing
        request_data["vae_use_tiling"] = vae_use_tiling

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
