## Project: ComfyUI-vLLM-Omni Text-to-Image Node

### **Overview**

This repository contains a **custom ComfyUI node** that enables ComfyUI to generate images using **vLLM-Omni’s diffusion-capable backend**. The node acts as a bridge between ComfyUI workflows and a running vLLM-Omni server that exposes a text-to-image generation API.

vLLM-Omni recently introduced *experimental diffusion support*, including the ability to generate images using models such as **Qwen-Image**. However, vLLM-Omni currently lacks a native ComfyUI integration. This project provides that integration.

---

### **Purpose**

The purpose of this custom node is to:

1. **Expose vLLM-Omni’s text-to-image generation capabilities inside ComfyUI**, allowing users to include Omni-powered diffusion models in their node graphs.
2. **Provide a simple, lightweight PoC** implementation that sends prompts & parameters to a vLLM-Omni backend via a REST API and converts the returned base64-encoded images into ComfyUI-compatible image tensors.
3. **Serve as a foundation** for future expansion:

   * multi-prompt batching
   * negative prompts
   * multi-image outputs
   * multimodal features (once available in vLLM-Omni API)
   * eventual OpenAI-style image endpoints

This node intentionally keeps all heavy lifting in the vLLM-Omni backend; the node’s responsibility is only to collect parameters, send them to the API server and convert results for ComfyUI.

---

### **Scope of This Spec**

This document provides only **the conceptual purpose and high-level architecture**.
A separate agent-oriented build specification (e.g., `BUILD_INSTRUCTIONS.md`) will include:

* required files
* node class structure
* input/output definitions
* API request structure
* error handling
* installation notes

This keeps CLAUDE.md focused strictly on project definition and intent.

---

### **High-Level Architecture**

```
[ ComfyUI Workflow ]
        ↓
[ Custom Node: VLLMTextToImage ]
        ↓   (HTTP POST /v1/images/generations)
[ vLLM-Omni Server ]
        ↓   (Omni.generate(): diffusion pipeline)
[ Base64-encoded Images ]
        ↓
[ Node decodes → ComfyUI image tensor output ]
```

---

### **Current Status**

* The API used by this node is **experimental** and implemented as part of this project (vLLM-Omni does not ship a text-to-image endpoint yet).
* The node supports:

  * prompt
  * negative prompt
  * height / width
  * num inference steps
  * guidance scale
  * seed
  * number of images
  * custom server URL

Future extensions may add video, audio, or multimodal support once vLLM-Omni exposes corresponding stable APIs.

---

### **Intended Audience**

This repo is designed so that:

* **Automation agents** understand the purpose & constraints of the project.
* **Developers** know where this node fits in the broader ecosystem.
* **Users** can install the node and immediately see how it integrates with ComfyUI and vLLM-Omni.

### **Tools & References**

There's a `./references/` folder here with code references and tools to use.

Included are git clones with: 
* cookiecutter-comfy-extension, to scaffold comfyui custom nodes.
* ComfyUI React Extension Template: A minimal template for creating React/TypeScript frontend extensions for ComfyUI, with complete boilerplate setup.
* The full comfyui codebase



