# ComfyUI-vLLM-Omni

> **⚠️ UPDATES COMING SOON! **
> A [DALL-E compatible API was added to vllm-omni](https://github.com/vllm-project/vllm-omni/pull/292)! An official implementation that tracks the official release, coming soon!
> ...and maybe experimental endpoints as well.

Until then...

> **⚠️ EXPERIMENTAL PROOF-OF-CONCEPT**
> This is an experimental integration that relies on experimental features in vLLM-Omni. The image generation API in vLLM-Omni is currently under development and may change. Use this for testing and development purposes only.

Custom ComfyUI nodes that enable **text-to-image generation** and **image editing** using **vLLM-Omni's diffusion backend**. This integration allows you to use vLLM-Omni's experimental image capabilities (Qwen-Image, Qwen-Image-Edit) directly within ComfyUI workflows.

![Example Generation](docs/images/example-steampunk-skier.png)
*Example: "telemark skier in the Adirondacks, 1880s clothing, steampunk goggles, action shot, powder skiing, a portrait by Nick Alm"*

## Features

### Text-to-Image Generation
- **OpenAI DALL-E Compatible API**: Uses standard API format for easy integration
- **Full Parameter Control**: Adjust width, height, steps, guidance scale, seed, and more
- **Negative Prompts**: Guide what NOT to generate
- **Batch Generation**: Generate multiple images in a single request
- **Model-Aware Defaults**: Automatically detects server model and adjusts parameters for optimal results

### Image Editing
- **Edit existing images** with text prompts
- **Auto size calculation** from input image aspect ratio
- **Batch variations**: Generate multiple edited versions
- **Mask support** for future inpainting capabilities
- **Advanced CFG controls** with dual guidance scales

### General
- **Async HTTP**: Non-blocking network calls for better performance
- **ComfyUI Native**: Integrates seamlessly with ComfyUI's node graph system
- **Flexible server configuration**: Split base URL and endpoint path for easier setup

## Model-Aware Defaults

The node automatically detects which diffusion model is running on the vLLM-Omni server and adjusts default parameters for optimal results:

| Model | Default Steps | Guidance Scale | Best For |
|-------|--------------|----------------|----------|
| **Qwen/Qwen-Image** | 50 | 4.0 | High quality, detailed images |
| **Z-Image-Turbo** | 9 | 0.0 | Fast generation, good quality |

**How it works:**
- When you leave parameters at their default values, the node queries `/health` to detect the model
- Model-specific defaults are automatically applied
- If you manually adjust ANY parameter, your value is always used
- Old servers without `/health` fall back to Qwen-Image defaults

**Example:** With Z-Image-Turbo server, leaving defaults will use 9 steps (fast).
With Qwen-Image, defaults remain 50 steps (quality).

No configuration required - it just works!

## Requirements

- **ComfyUI** installed and running
- **Python 3.9+**
- **vLLM-Omni** with **experimental** image generation server support
  - ⚠️ Requires vLLM-Omni with the experimental image generation endpoint
  - The image server implementation is not yet part of the official vLLM-Omni release
  - See `serverside-implementation.md` for details on the server-side requirements
- Dependencies (most already included with ComfyUI):
  - `aiohttp>=3.8.0`
  - `torch>=2.0.0`
  - `pillow>=9.0.0`
  - `numpy>=1.21.0`

## Installation

### Step 1: Install the Custom Node

Clone this repository into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/comfyui-vllm-omni.git
cd comfyui-vllm-omni
pip install -r requirements.txt
```

### Step 2: Start vLLM-Omni Image Server

> **⚠️ EXPERIMENTAL:** The vLLM-Omni image generation server is an experimental feature. The API endpoint may not be available in official vLLM-Omni releases yet. See `serverside-implementation.md` for implementation details.

You need a running vLLM-Omni server with image generation support. Start it with:

```bash
python -m vllm_omni.entrypoints.openai.serving_image \
  --model Qwen/Qwen-Image \
  --port 8000 \
  --host 0.0.0.0
```

**Note:** Adjust the model path and port as needed. The default server URL in the node is `http://localhost:8000/v1/images/generations`.

### Step 3: Restart ComfyUI

Restart ComfyUI to load the new custom node.

## Usage

### Basic Workflow

1. **Add the Node**: In ComfyUI, right-click → Add Node → image/generation/vllm-omni → **vLLM-Omni Text-to-Image**

2. **Configure Parameters**:
   - **prompt** (required): Describe what you want to generate
   - **negative_prompt** (optional): Describe what to avoid
   - **width** / **height**: Image dimensions (default: 1024x1024)
   - **num_inference_steps**: Denoising steps (default: 50)
   - **guidance_scale**: CFG scale (default: 4.0)
   - **n**: Number of images to generate (default: 1)
   - **seed**: Random seed for reproducibility (0 = random)
   - **server_url**: vLLM-Omni endpoint URL

3. **Connect Output**: Connect the IMAGE output to other nodes (e.g., SaveImage, PreviewImage)

4. **Queue Prompt**: Generate your images!

### Example Prompts

```
Positive: "a majestic dragon flying over snow-capped mountains at sunset, highly detailed, 4k"
Negative: "blurry, low quality, distorted, ugly"
```

```
Positive: "a cute robot reading a book in a cozy library, warm lighting, illustration style"
Negative: "dark, scary, realistic"
```

## Example Workflows

Ready-to-use workflow examples are provided in the `examples/` folder. You can drag and drop these JSON files into ComfyUI to get started quickly.

### Available Examples

1. **`vllm-omni basic t2i.json`** - Basic text-to-image generation
   - Simple workflow showing the vLLM-Omni Text-to-Image node
   - Generates: "astronaut riding a horse on the moon"
   - Uses: Qwen-Image or Z-Image-Turbo model

2. **`vllm-omni image edit.json`** - Image editing workflow
   - Load an image and edit it with text prompts
   - Example: "put a tiara on the cat's head"
   - Shows auto size calculation (width=0, height=0)

   ![Image Edit Workflow](docs/images/image-edit-workflow.png)

3. **`vllm-omni text to image plus edit.json`** - Combined workflow
   - Generate image with text-to-image, then edit with image edit node
   - Example: Generate "universe in a bottle" → Edit "add fish swimming"
   - Shows how to chain both nodes together

   ![Text-to-Image + Edit Workflow](docs/images/text-to-image-and-edit-workflow.png)

### Using the Examples

1. Download or clone this repository
2. Open ComfyUI
3. Drag and drop any `.json` file from `examples/` into the ComfyUI window
4. Adjust the `server_base_url` if your vLLM-Omni server is not on localhost:8000
5. Queue the workflow!

## Parameters Reference

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| **prompt** | STRING | "" | - | Text description of image to generate (required) |
| **negative_prompt** | STRING | "" | - | What NOT to generate (optional) |
| **width** | INT | 1024 | 256-2048 | Image width in pixels (step: 64) |
| **height** | INT | 1024 | 256-2048 | Image height in pixels (step: 64) |
| **num_inference_steps** | INT | 50 | 1-200 | Number of denoising steps (higher = better quality, slower) |
| **guidance_scale** | FLOAT | 4.0 | 0.0-20.0 | CFG scale (higher = more prompt adherence) |
| **n** | INT | 1 | 1-10 | Number of images to generate |
| **seed** | INT | 0 | 0-2³¹ | Random seed (0 = random) |
| **server_url** | STRING | http://localhost:8000/... | - | vLLM-Omni API endpoint |

## API Format

This node communicates with vLLM-Omni using the OpenAI DALL-E compatible API format:

### Request

```json
POST /v1/images/generations
{
  "prompt": "a cat on a laptop",
  "n": 1,
  "size": "1024x1024",
  "response_format": "b64_json",
  "negative_prompt": "",
  "num_inference_steps": 50,
  "guidance_scale": 4.0,
  "seed": 42
}
```

### Response

```json
{
  "created": 1234567890,
  "data": [
    {
      "b64_json": "iVBORw0KGgoAAAANSUhEUgAA..."
    }
  ]
}
```

**Note:** The node automatically converts ComfyUI's separate width/height parameters to the OpenAI `size` format ("WIDTHxHEIGHT").

## Troubleshooting

### "Connection refused" or "Network error"

**Problem:** Cannot connect to vLLM-Omni server

**Solutions:**
- Ensure the vLLM-Omni server is running
- Check the server URL and port in the node parameters
- Verify firewall settings allow connections
- Try `curl http://localhost:8000/health` to test server

### "Request timed out"

**Problem:** Generation takes too long (>300s default timeout)

**Solutions:**
- Reduce `num_inference_steps` (try 30-40 instead of 50)
- Reduce image size (try 512x512 instead of 1024x1024)
- Check server GPU resources (might be OOM or slow)

### "Prompt cannot be empty"

**Problem:** No prompt provided

**Solution:** Enter a text prompt in the prompt field

### "API response missing 'data' field"

**Problem:** Server returned unexpected response format

**Solutions:**
- Ensure you're using vLLM-Omni's image server (not text server)
- Check server logs for errors
- Verify server is running the correct endpoint

### Images appear corrupted or wrong colors

**Problem:** Tensor format mismatch

**Solution:** This should not happen with the current implementation, but if it does:
- Check that server is returning valid PNG data
- Verify base64 encoding is correct
- Report as a bug with server/client versions

## Advanced Usage

### Using Different Servers

You can run multiple vLLM-Omni servers with different models and switch between them:

```bash
# Server 1: Qwen-Image on port 8000
python -m vllm_omni.entrypoints.openai.serving_image --model Qwen/Qwen-Image --port 8000

# Server 2: Another model on port 8001
python -m vllm_omni.entrypoints.openai.serving_image --model AnotherModel --port 8001
```

Then in the node, change `server_url` to `http://localhost:8001/v1/images/generations`.

### Reproducible Generation

Set a specific seed value (not 0) to get reproducible results:

```
seed: 42 → Same prompt + seed = same image
seed: 0  → Random seed each time = different images
```

### Batch Generation

Set `n` to generate multiple variations at once. The output will be a batch of images that you can process individually using ComfyUI's batch processing nodes.

## Architecture

```
┌──────────────────┐
│  ComfyUI Node    │
│  (This Package)  │
└────────┬─────────┘
         │ HTTP POST /v1/images/generations
         │ (OpenAI DALL-E format)
┌────────▼─────────┐
│  vLLM-Omni       │
│  Image Server    │
└────────┬─────────┘
         │
┌────────▼─────────┐
│  Omni.generate() │
│  Diffusion Model │
└──────────────────┘
```

**Data Flow:**
1. ComfyUI node collects parameters
2. Converts to OpenAI API format (`size` string, etc.)
3. Sends HTTP POST to vLLM-Omni server
4. Server generates images using diffusion model
5. Returns base64-encoded PNGs
6. Node decodes to PIL → numpy → torch tensor
7. Returns ComfyUI-compatible IMAGE tensor

## File Structure

```
comfyui-vllm-omni/
├── __init__.py              # Node registration
├── vllm_omni_node.py        # Main ComfyUI node class
├── vllm_api.py              # HTTP client for vLLM-Omni API
├── utils.py                 # Image conversion utilities
├── requirements.txt         # Python dependencies
├── pyproject.toml          # Package metadata
└── README.md               # This file
```

## Development

### Running Tests

Currently, testing requires a live vLLM-Omni server. Future versions may include unit tests with mocked API responses.

### Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with a live vLLM-Omni server
5. Submit a pull request

## Known Limitations

This is a **Proof-of-Concept (PoC)** implementation with the following limitations:

1. **⚠️ Experimental vLLM-Omni API**: Relies on experimental image generation features in vLLM-Omni that are under active development and subject to change
2. **Async Generation Only**: Requires modern ComfyUI with async node support
3. **Single Server**: No automatic load balancing or failover
4. **No Progress Bar**: No real-time progress updates during generation
5. **Base64 Only**: No direct file URL support (would require image hosting)
6. **No Authentication**: Assumes open localhost server

These may be addressed in future versions as vLLM-Omni's image generation capabilities mature.

## Future Enhancements

Potential features for future releases:

- Image-to-image generation support
- Inpainting with mask support
- LoRA model selection
- ControlNet integration
- Progress bar during generation
- Connection pooling for better performance
- Model switching without server restart
- Authentication support for remote servers

## License

MIT License - See LICENSE file for details

## Credits

- **vLLM-Omni**: For providing the diffusion backend
- **ComfyUI**: For the excellent node-based UI framework
- **Qwen-Image**: For the powerful diffusion model

## Support

For issues and questions:
- **GitHub Issues**: [Report bugs or request features](https://github.com/yourusername/comfyui-vllm-omni/issues)
- **vLLM-Omni Docs**: [vLLM-Omni documentation](https://github.com/vllm-project/vllm-omni)
- **ComfyUI Docs**: [ComfyUI custom nodes guide](https://github.com/comfyanonymous/ComfyUI)

## Changelog

### v0.1.0 (2024-XX-XX)

- Initial release
- Basic text-to-image generation
- OpenAI DALL-E compatible API
- Negative prompt support
- Batch generation support (n parameter)
- Configurable server URL
- Full parameter control (steps, guidance, size, seed)
