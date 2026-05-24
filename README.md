# 🎬 Free AI Studio: LTX-Video 2.3 on Kaggle

[![Kaggle Notebook](https://img.shields.io/badge/Run%20on-Kaggle-blue?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/)
[![stable-diffusion.cpp](https://img.shields.io/badge/Engine-stable--diffusion.cpp-orange?style=for-the-badge)](https://github.com/leejet/stable-diffusion.cpp)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

An optimized, end-to-end production framework to run the **22-Billion parameter LTX-Video 2.3** model on Kaggle's free Tesla T4 GPU tier (15 GB VRAM ceiling). Generates high-quality 5-second videos with synced audio track in under **6 minutes**.

---

## 📂 Repository Structure

To support clean execution and future expansions, the repository is split into user-facing notebooks and a modular Python backend:

```
free-aistudio/
├── notebooks/
│   └── ltx2-3-video.ipynb       # 🎬 Main Kaggle Jupyter Notebook
├── src/                         # 🛠️ Backend helper modules
│   ├── downloader.py            # High-speed model/binary downlader (via aria2c)
│   ├── server.py                # Wrapper to launch the C++ inference server
│   └── ui.py                    # Gradio frontend interface
└── requirements.txt             # 🐍 Python dependencies
```

---

## ⚡ Quick Start: How to Run on Kaggle

Follow these simple steps to run the pipeline on Kaggle:

### 1. Set Up Your Kaggle Environment
1. Go to [Kaggle](https://www.kaggle.com/) and create a **New Notebook**.
2. In the right panel, set the **Accelerator** to **GPU T4** (Internet must be **Enabled**).

### 2. Run the Initialization Cell
Copy and paste this code into your first notebook cell to clone the repository and install requirements:

```python
import os

# 1. Sync the codebase
if not os.path.exists('/kaggle/working/free-aistudio'):
    print("📦 Repository not found. Cloning new codebase...")
    !git clone https://github.com/your-username/free-aistudio.git /kaggle/working/free-aistudio
else:
    print("🔄 Repository exists. Syncing latest updates...")
    !git -C /kaggle/working/free-aistudio pull

# 2. Install dependencies
!pip install -q -r /kaggle/working/free-aistudio/requirements.txt

# 3. Add to Python Path
import sys
sys.path.append('/kaggle/working/free-aistudio')
```

### 3. Restore the Engine & Weights
Execute the downloader module to grab the pre-compiled C++ server binary and LTX weights (~20 GB total) in a few minutes:

```python
from src.downloader import restore_binary, download_models

# Restore C++ engine from GitHub Releases
restore_binary(repo="your-username/free-aistudio", tag="v1.0.0")

# Download LTX-Video weights
download_models(preset="LTX-Video-2.3-Q3")
```

### 4. Launch the C++ Inference Server
Load the weights into VRAM. The background server takes about 90 seconds to fully warm up:

```python
from src.server import start_server, tail_logs

# Launch background server
server_process = start_server(load_audio_vae=True)

# Show active startup log status
print(tail_logs())
```

### 5. Launch the Web UI
Run the Gradio block and click the public shareable link:

```python
from src.ui import launch

# Launches the interactive video generation web room
launch()
```

---

## 💡 Key Configurations & Optimizations

- **VRAM Saving (VAE Tiling)**: Video VAE decoding is split into tiles (`--vae-tiling`) to prevent Kaggle's T4 GPU from running Out-of-Memory (OOM).
- **Quantization**: Text encoder is loaded as a 2-bit quantized Gemma-3-12B weight (`UD-IQ2_XXS`), and the diffusion model uses `Q3_K_M` GGUF to maintain high-quality outputs within VRAM constraints.
- **Audio Integration**: The pipeline automatically multiplexes spatial audio alongside video streams when sound triggers are described in prompts.

---

## 🚀 Adding Future Models

This workspace layout makes adding image generation models (like SDXL or Flux) easy:
1. Register new Hugging Face weights under `MODEL_PRESETS` in [`src/downloader.py`](src/downloader.py).
2. Configure parameters for running those models in [`src/server.py`](src/server.py).
3. Create a dedicated notebook in [`notebooks/`](notebooks/) calling the new configurations.

---

## ❤️ Credits
Built by the YouTube community for free AI generation. Engine powered by [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp).
