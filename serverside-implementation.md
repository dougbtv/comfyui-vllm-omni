vLLM-Omni Text-to-Image API Implementation Plan

 Overview

 Create an OpenAI DALL-E compatible text-to-image API for vllm-omni, enabling ComfyUI (and other clients) to generate images via HTTP. This is a standalone server separate from the existing LLM server,
 wrapping the existing OmniDiffusion backend.

 Architecture

 ┌────────────────────────────────┐
 │  HTTP Client (ComfyUI, curl)  │
 └───────────┬────────────────────┘
             │ POST /v1/images/generations
 ┌───────────▼────────────────────┐
 │  FastAPI Server                │
 │  - Parse OpenAI request        │
 │  - Map to OmniDiffusionRequest │
 │  - Encode PIL → base64         │
 └───────────┬────────────────────┘
             │
 ┌───────────▼────────────────────┐
 │  Omni.generate()               │
 │  (existing diffusion engine)   │
 └───────────┬────────────────────┘
             │
             └→ List[PIL.Image]

 Files to Create

 1. Protocol Definitions

 Path: vllm_omni/entrypoints/openai/protocol/images.py

 Purpose: Pydantic models for OpenAI-compatible request/response

 Key Classes:
 - ImageGenerationRequest: OpenAI standard fields (prompt, n, size, response_format) + vllm-omni extensions (negative_prompt, num_inference_steps, true_cfg_scale, seed)
 - ImageGenerationResponse: Contains created timestamp and data array of images
 - ImageData: Single image with b64_json field
 - ImageSize: Enum for supported sizes (256x256, 512x512, 1024x1024, etc.)
 - ResponseFormat: Enum (b64_json, url - only b64_json supported in PoC)

 Validation:
 - n: 1-10 images
 - num_inference_steps: 1-200
 - guidance_scale/true_cfg_scale: 0.0-20.0
 - Required: prompt

 2. FastAPI Server

 Path: vllm_omni/entrypoints/openai/image_server.py

 Purpose: Main server application with endpoint implementation

 Key Components:

 # Global state (simple PoC approach)
 omni_instance: Optional[Omni] = None
 model_name: str = None

 @asynccontextmanager
 async def lifespan(app):
     """Load model on startup, cleanup on shutdown"""
     global omni_instance
     omni_instance = Omni(
         model=model_name,
         vae_use_slicing=True,
         vae_use_tiling=True,
     )
     yield
     omni_instance = None

 @app.post("/v1/images/generations")
 async def create_image(request: ImageGenerationRequest):
     """Generate images from text prompt"""
     # 1. Parse size: "1024x1024" → (1024, 1024)
     # 2. Create generator if seed provided
     # 3. Call omni.generate(prompt, **params)
     # 4. Encode PIL images to base64 PNG
     # 5. Return ImageGenerationResponse

 Helper Functions:
 - parse_size(size_str: str) -> tuple[int, int]: Parse "1024x1024" to (width, height)
 - encode_image_base64(image: PIL.Image) -> str: PIL → base64 PNG string

 Error Handling:
 - 400 Bad Request: Invalid size, unsupported response_format, validation errors
 - 422 Unprocessable Entity: Pydantic validation failures
 - 500 Internal Server Error: Generation failures, CUDA OOM
 - 503 Service Unavailable: Model not loaded

 3. CLI Entry Point

 Path: vllm_omni/entrypoints/openai/serving_image.py

 Purpose: Command-line launcher for the server

 Usage:
 python -m vllm_omni.entrypoints.openai.serving_image \
   --model Qwen/Qwen-Image \
   --host 0.0.0.0 \
   --port 8000

 Arguments:
 - --model: Model name/path (default: Qwen/Qwen-Image)
 - --host: Bind address (default: 0.0.0.0)
 - --port: Port number (default: 8000)
 - --log-level: debug/info/warning/error (default: info)
 - --reload: Enable auto-reload for development

 Implementation: Simple argparse + uvicorn.run()

 4. Example Client

 Path: examples/api_server/image_generation_client.py

 Purpose: Demonstrate API usage with Python requests library

 Features:
 - Send POST request with prompt and parameters
 - Decode base64 response to PIL Image
 - Save to file
 - Support for multiple images (batch generation)

 Example Usage:
 python examples/api_server/image_generation_client.py \
   --prompt "a dragon in the sky" \
   --output dragon.png \
   --seed 42 \
   --steps 50

 5. Tests

 Path: tests/entrypoints/openai/test_image_server.py

 Test Categories:

 Unit Tests (no GPU required):
 - test_parse_size_valid(): Valid size strings
 - test_parse_size_invalid(): Invalid formats
 - test_encode_image_base64(): PIL → base64 → PIL roundtrip

 Integration Tests (mocked):
 - test_health_endpoint(): Health check works
 - test_generate_single_image(): Basic generation flow
 - test_generate_multiple_images(): Batch (n > 1)
 - test_with_negative_prompt(): Negative prompt parameter
 - test_with_seed(): Seed for reproducibility
 - test_invalid_size(): Error handling
 - test_missing_prompt(): Validation errors
 - test_url_response_format_not_supported(): Unsupported format

 Mocking Strategy: Mock Omni.generate() to return fake PIL images

 Parameter Mapping

 OpenAI → vllm-omni

 | OpenAI Field        | vllm-omni Field                                | Mapping                             |
 |---------------------|------------------------------------------------|-------------------------------------|
 | prompt              | prompt                                         | Direct                              |
 | n                   | num_images_per_prompt + num_outputs_per_prompt | Both set to n                       |
 | size                | height + width                                 | Parse "1024x1024" → (1024, 1024)    |
 | response_format     | N/A                                            | Only accept "b64_json"              |
 | model               | N/A                                            | Fixed at server startup             |
 | Extensions          |                                                |                                     |
 | negative_prompt     | negative_prompt                                | Direct                              |
 | num_inference_steps | num_inference_steps                            | Direct (default: 50)                |
 | true_cfg_scale      | true_cfg_scale                                 | Direct (default: 4.0)               |
 | seed                | generator                                      | torch.Generator().manual_seed(seed) |

 Ignored Parameters (PoC)

 - OpenAI quality, style: Not applicable to Qwen-Image
 - OpenAI user: Could add for logging later

 Implementation Sequence

 Phase 1: Protocol (Day 1)

 1. Create protocol/images.py with Pydantic models
 2. Add validation constraints (ranges, enums)
 3. Write unit tests for protocol

 Phase 2: Server Core (Day 2)

 1. Create image_server.py with FastAPI app
 2. Implement lifespan (model loading)
 3. Add /health endpoint
 4. Add /v1/images/generations endpoint
 5. Implement parameter mapping

 Phase 3: Image Processing (Day 3)

 1. Implement parse_size() function
 2. Implement encode_image_base64() function
 3. Add generator creation logic
 4. Test with mocked Omni instance

 Phase 4: CLI & Integration (Day 4)

 1. Create serving_image.py CLI entry point
 2. Test with real model (GPU required)
 3. Verify end-to-end generation
 4. Debug any parameter issues

 Phase 5: Testing & Examples (Day 5)

 1. Write comprehensive test suite
 2. Create example client script
 3. Test different parameter combinations
 4. Add error scenario tests

 Key Implementation Details

 Model Loading (Lifespan Context)

 @asynccontextmanager
 async def lifespan(app: FastAPI):
     global omni_instance
     logger.info(f"Loading model: {model_name}")
     omni_instance = Omni(
         model=model_name,
         vae_use_slicing=True,  # Memory optimization
         vae_use_tiling=True,   # Memory optimization
     )
     logger.info("Model loaded")
     yield
     logger.info("Shutting down")
     omni_instance = None

 Generation Call

 # Parse size
 width, height = parse_size(request.size)

 # Create generator if seed provided
 generator = None
 if request.seed is not None:
     from vllm_omni.utils.platform_utils import detect_device_type
     device = detect_device_type()
     generator = torch.Generator(device=device).manual_seed(request.seed)

 # Generate images
 images = omni_instance.generate(
     prompt=request.prompt,
     negative_prompt=request.negative_prompt,
     height=height,
     width=width,
     num_images_per_prompt=request.n,
     num_outputs_per_prompt=request.n,
     num_inference_steps=request.num_inference_steps,
     true_cfg_scale=request.true_cfg_scale,
     guidance_scale=request.guidance_scale,
     generator=generator,
 )

 Base64 Encoding

 def encode_image_base64(image: PIL.Image) -> str:
     """Encode PIL Image to base64 PNG string"""
     buffer = io.BytesIO()
     image.save(buffer, format="PNG")
     buffer.seek(0)
     return base64.b64encode(buffer.read()).decode('utf-8')

 Reference Files

 Read these for context (don't modify):
 - vllm_omni/entrypoints/omni.py: Shows Omni class usage pattern
 - vllm_omni/diffusion/omni_diffusion.py: Backend generate() method
 - vllm_omni/diffusion/request.py: Valid parameters for OmniDiffusionRequest
 - examples/offline_inference/qwen_image/text_to_image.py: Usage example

 Testing Strategy

 Manual Testing (Requires GPU)

 # Start server
 python -m vllm_omni.entrypoints.openai.serving_image \
   --model Qwen/Qwen-Image \
   --port 8000

 # Test with curl
 curl -X POST http://localhost:8000/v1/images/generations \
   -H "Content-Type: application/json" \
   -d '{
     "prompt": "a cat on a laptop",
     "size": "1024x1024",
     "seed": 42
   }' | jq -r '.data[0].b64_json' | base64 -d > output.png

 # Test with Python client
 python examples/api_server/image_generation_client.py \
   --prompt "a dragon" \
   --output dragon.png

 Automated Testing

 # Unit tests (no GPU)
 pytest tests/entrypoints/openai/test_image_server.py -k unit

 # Integration tests (mocked)
 pytest tests/entrypoints/openai/test_image_server.py

 Documentation to Add

 README Section

 Add to main README.md:

 ## Image Generation API

 vLLM-Omni provides an OpenAI DALL-E compatible API for text-to-image generation.

 ### Quick Start

 # Start the server
 python -m vllm_omni.entrypoints.openai.serving_image \
   --model Qwen/Qwen-Image

 # Generate an image
 curl -X POST http://localhost:8000/v1/images/generations \
   -H "Content-Type: application/json" \
   -d '{"prompt": "a cat", "size": "1024x1024"}'

 API Documentation

 Create docs/image_api.md with:
 - Endpoint specification
 - Parameter descriptions
 - Request/response examples
 - Error codes
 - Troubleshooting

 Dependencies

 All required packages already exist in pyproject.toml:
 - fastapi - Web framework
 - uvicorn - ASGI server
 - pydantic - Validation
 - torch - For Generator
 - pillow - Image handling
 - pytest - Testing

 No new dependencies needed!

 Success Criteria

 - Server starts and loads Qwen-Image model
 - /v1/images/generations endpoint accepts OpenAI-formatted requests
 - Images generated successfully and returned as base64
 - Seed parameter produces reproducible results
 - Multiple images (n > 1) work correctly
 - Negative prompts are respected
 - All tests pass
 - Example client works end-to-end
 - Error handling provides clear messages

 Known Limitations (Acceptable for PoC)

 1. Synchronous only: Blocks during generation (OK for single-user PoC)
 2. Single model: Cannot switch models without restart
 3. b64_json only: URL format not implemented (requires hosting)
 4. No batching: One request at a time
 5. No authentication: Open access (fine for local/dev)

 Future Enhancements (Out of Scope)

 - Async request handling for concurrency
 - Multiple model support
 - URL response format with image hosting
 - Request queuing and load balancing
 - Integration with main vllm serve command
 - Authentication and rate limiting
 - Prometheus metrics endpoint

 Notes for Implementation Agent

 1. Start simple: Get basic endpoint working first, then add features
 2. Use existing patterns: FastAPI app structure follows vllm patterns
 3. Test incrementally: Unit tests → mocked integration → real model
 4. Reference offline example: text_to_image.py shows correct parameter usage
 5. Error messages matter: Users need clear feedback for debugging
 6. Logging is critical: Log requests, errors, and generation times

 Contact Points with ComfyUI Integration

 The ComfyUI custom node (handled by separate agent) will:
 - POST to http://localhost:8000/v1/images/generations
 - Send OpenAI-compatible JSON request
 - Receive base64-encoded PNG in response
 - Decode and convert to ComfyUI tensor format

 This backend API is the server side of that integration.
