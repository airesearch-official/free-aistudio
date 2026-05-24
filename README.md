# 🎬 Free AI Studio: LTX-Video 2.3 on Kaggle

[![Kaggle Notebook](https://img.shields.io/badge/Run%20on-Kaggle-blue?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/)
[![stable-diffusion.cpp](https://img.shields.io/badge/Engine-stable--diffusion.cpp-orange?style=for-the-badge)](https://github.com/leejet/stable-diffusion.cpp)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

An optimized, end-to-end production framework to run the **22-Billion parameter LTX-Video 2.3** model on Kaggle's free Tesla T4 GPU tier (15 GB VRAM ceiling). Generates high-quality 5-second videos with synced audio track in under **6 minutes**.

---

## 🗺️ Project Status & Roadmap

This studio is designed to be a unified, future-proof suite for running generative AI on free cloud platforms.

- **`[x]` LTX-Video 2.3 Video Pipeline** (Current Release)
  - Pre-quantized Q3 weights running under VRAM limits.
  - Native spatial latent upscaling (2.0x) pass.
  - Synced audio track generation.
- **`[ ]` Stable Diffusion XL (SDXL) Image Generation** (Upcoming)
  - Fast C++ inference presets for lightweight image generation.
- **`[ ]` FLUX.1 Image Generation** (Upcoming)
  - Quantized Flux GGUF inference setups on T4 GPUs.
- **`[ ]` Unified Multimodal Web UI** (Upcoming)
  - Switch between video, image, and audio-only generation modes from a single frontend room.

---

## 📂 Repository Structure

To support clean execution, the repository separates user-facing notebooks from the core logic:

```
free-aistudio/
├── notebooks/
│   └── ltx2-3-video.ipynb       # 🎬 LTX-Video 2.3 Jupyter Notebook (Import this to Kaggle!)
├── src/                         # 🛠️ Backend helper modules
│   ├── downloader.py            # High-speed model/binary downloader (via aria2c)
│   ├── server.py                # Wrapper to launch the C++ inference server
│   └── ui.py                    # Gradio frontend interface
└── requirements.txt             # 🐍 Python dependencies
```

---

## ⚡ Quick Start: How to Run on Kaggle

Rather than creating code from scratch, you import our pre-configured notebook directly.

### Step 1: Download the Notebook
Save the user-facing notebook file from this repository to your local machine:
- 💾 **[ltx2-3-video.ipynb](notebooks/ltx2-3-video.ipynb)** (Or click download on GitHub).

### Step 2: Upload to Kaggle
1. Go to [Kaggle Notebooks](https://www.kaggle.com/code) and click **New Notebook**.
2. Click **File** → **Upload Notebook** and select the `ltx2-3-video.ipynb` file you just downloaded.
3. In the notebook settings panel (right-hand sidebar):
   - Set **Accelerator** to **GPU T4** (either 1x or 2x T4).
   - Ensure **Internet** is turned **On**.

### Step 3: Run the Cells
Once imported, you only need to run the pre-made cells in sequence. The notebook will automatically:
1. Clone the rest of the repository code.
2. Download the pre-built CUDA C++ binary and model weights.
3. Launch the background server and display your Gradio Web UI link.

---

## 💡 Key Configurations & Optimizations

- **VRAM Saving (VAE Tiling)**: Video VAE decoding is split into tiles (`--vae-tiling`) to prevent Kaggle's T4 GPU from running Out-of-Memory (OOM).
- **Quantization**: Text encoder is loaded as a 2-bit quantized Gemma-3-12B weight (`UD-IQ2_XXS`), and the diffusion model uses `Q3_K_M` GGUF to maintain high-quality outputs within VRAM constraints.
- **Audio Integration**: The pipeline automatically multiplexes spatial audio alongside video streams when sound triggers are described in prompts.

---

## ❤️ Credits
Built by the YouTube community for free AI generation. Engine powered by [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp).
