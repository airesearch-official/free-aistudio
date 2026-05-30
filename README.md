# 🎬 Free AI Studio on Kaggle

[![Kaggle Notebook](https://img.shields.io/badge/Run%20on-Kaggle-blue?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/)
[![stable-diffusion.cpp](https://img.shields.io/badge/Engine-stable--diffusion.cpp-orange?style=for-the-badge)](https://github.com/leejet/stable-diffusion.cpp)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

An optimized, end-to-end production framework to run state-of-the-art AI generation models on Kaggle's free Tesla T4 GPU tier (15 GB VRAM ceiling) utilizing the high-performance [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) engine.

---

## 🗺️ Project Status & Roadmap

This studio is designed to be a unified, future-proof suite for running generative AI on free cloud platforms.

- **`[x]` LTX-Video 2.3 Video Pipeline** (Current Release)
  - Pre-quantized Q3 weights running under VRAM limits.
  - Native spatial latent upscaling (2.0x) pass.
  - Synced audio track generation.
- **`[x]` Z-Image-Turbo Image Pipeline** (Current Release)
  - Ultra-fast image generation utilizing GGUF models.
  - High-speed inference (under 2 seconds per image) using persistent server RAM configurations.
  - Custom resolution presets and LoRA support.
- **`[ ]` More Models & Features Coming Soon!**
  - New generative features and community-requested presets will be added as they arrive.

---

## 📂 Repository Structure

To support clean execution, the repository separates user-facing notebooks from the core logic:

```
free-aistudio/
├── notebooks/
│   ├── ltx2-3-video.ipynb       # 🎬 LTX-Video 2.3 Jupyter Notebook (Video Studio)
│   └── z-image-turbo.ipynb      # 🖼️ Z-Image-Turbo Jupyter Notebook (Image Studio)
├── src/                         # 🛠️ Backend helper modules
│   ├── downloader.py            # High-speed model/binary downloader (via aria2c)
│   ├── server.py                # Wrapper to launch the C++ inference server
│   └── ui.py                    # Gradio frontend interface
└── requirements.txt             # 🐍 Python dependencies
```

---

## ⚡ Quick Start: How to Run on Kaggle

Rather than creating code from scratch, you import one of our pre-configured notebooks directly.

### Step 1: Download your preferred notebook
Save one of the user-facing notebooks from this repository to your local machine:
- 🎬 **[ltx2-3-video.ipynb](notebooks/ltx2-3-video.ipynb)** (For video generation)
- 🖼  **[z-image-turbo.ipynb](notebooks/z-image-turbo.ipynb)** (For image generation)

### Step 2: Upload to Kaggle
1. Go to [Kaggle Notebooks](https://www.kaggle.com/code) and click **New Notebook**.
2. Click **File** → **Upload Notebook** and select the `.ipynb` file you just downloaded.
3. In the notebook settings panel (right-hand sidebar):
   - Set **Accelerator** to **GPU T4** (either 1x or 2x T4).
   - Ensure **Internet** is turned **On**.

### Step 3: Run the Cells
Once imported, you only need to run the pre-made cells in sequence. The notebook will automatically sync the repository code, download the pre-built C++ server binary and optimized model weights, launch the background API inference server, and display your Gradio Web UI link.

---

## 💡 Key Configurations & Optimizations

- **VRAM Saving (VAE Tiling)**: Video VAE decoding is split into tiles (`--vae-tiling`) to prevent Kaggle's T4 GPU from running Out-of-Memory (OOM) during video generation.
- **Model Quantization**: Uses highly-quantized GGUF formats (e.g. Q3/Q4) to fit multiple large model weights simultaneously in memory.
- **Persistent Server Loading**: Starting the server once in the background eliminates reload delays. Subsequent requests generate images/videos instantly.

---

## ❤️ Credits
Built by the YouTube community for free AI generation. Engine powered by [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp).
