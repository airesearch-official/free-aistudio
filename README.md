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
- **`[x]` Lightning.ai High-Quality Studio (Experimental)** (Current Release)
  - Run video generation directly via CLI execution (`sd-cli`) under the hood to bypass server overhead.
  - Optimized GGUF model paths (`LTX-Video-2.3-FP8`) with Q8 diffusion.
  - One-click launch command via `run_lightning.py`.
- **`[ ]` More Models & Features Coming Soon!**
  - New generative features and community-requested presets will be added as they arrive.

---

## 📂 Repository Structure

To support clean execution, the repository separates user-facing entry points from the core logic:

```
free-aistudio/
├── notebooks/
│   ├── ltx2-3-video.ipynb       # 🎬 LTX-Video 2.3 Jupyter Notebook (Kaggle Video Studio)
│   ├── z-image-turbo.ipynb      # 🖼️ Z-Image-Turbo Jupyter Notebook (Kaggle Image Studio)
│   └── lightning-video.ipynb    # 🎬 LTX-Video 2.3 FP8 Notebook (Lightning.ai Studio)
├── src/                         # 🛠️ Backend helper modules
│   ├── downloader.py            # High-speed model/binary downloader
│   ├── server.py                # Wrapper to launch the C++ inference server
│   └── ui.py                    # Gradio frontend interface
├── run_lightning.py             # ⚡ One-click Python startup CLI for Lightning.ai
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

## ⚡ Quick Start: How to Run on Lightning.ai (Experimental)

To run the experimental high-quality FP8 model studio, you need a CUDA-enabled environment.

### Step 1: Create a Studio and Attach a GPU
1. Go to [Lightning.ai](https://lightning.ai/) and log in (or create a free account).
2. Click **Create Studio** in your dashboard.
3. Switch the hardware to an **L4 GPU** (or any CUDA-enabled GPU).
   > [!IMPORTANT]
   > Do **NOT** run this in a CPU-only environment, or the engine will fail to initialize.

### Step 2: Clone the Repository inside the Studio
1. Open a **Terminal** window in the Studio (you can find it in the bottom panel or by selecting "New Terminal").
2. Copy and run the following command to clone this repository into your workspace:
   ```bash
   git clone https://github.com/airesearch-official/free-aistudio.git
   ```
3. Navigate into the cloned folder:
   ```bash
   cd free-aistudio
   ```

### Step 3: Run the Studio
1. Run the script:
   ```bash
   python run_lightning.py
   ```
2. The script will download the prebuilt binary, download the models persistently, and launch the Gradio Web UI with a public `*.gradio.live` link.
3. The UI will execute video generation tasks on-demand via the `sd-cli` binary.

---

## 💡 Key Configurations & Optimizations

- **VRAM Saving (VAE Tiling)**: Video VAE decoding is split into tiles (`--vae-tiling`) to prevent Kaggle's T4 GPU from running Out-of-Memory (OOM) during video generation.
- **Model Quantization**: Uses highly-quantized GGUF formats (e.g. Q3/Q4) to fit multiple large model weights simultaneously in memory.
- **Persistent Server Loading**: Starting the server once in the background eliminates reload delays. Subsequent requests generate images/videos instantly.

---

## ❤️ Credits
Built by the YouTube community for free AI generation. Engine powered by [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp).
