import os
import sys
import subprocess

# Auto-install requirements if any are missing before importing local modules
try:
    import gradio
    import requests
except ImportError:
    print("📥 Installing python requirements (gradio, requests)...")
    req_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", req_path])

# Ensure import paths resolve correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.downloader import restore_lightning_binary, download_models
from src.ui import build_app, get_working_dir

def main():
    print("🚀 Starting AI Studio on Lightning.ai (CLI Mode)...")
    
    # Setup paths for Lightning.ai persistent workspace storage
    models_base = "/teamspace/studios/this_studio/models"
    bin_dir = "/teamspace/studios/this_studio/sd_bin"
    
    # 1. Restore the C++ compilation binary if missing
    restore_lightning_binary(
        repo="airesearch-official/free-aistudio",
        tag="v1.0.0",
        target_dir=bin_dir,
    )
        
    # 2. Download LTX-Video FP8 weights (downloader skips already completed downloads)
    download_models(preset="LTX-Video-2.3-FP8", models_base=models_base)
    
    # 3. Launch Gradio Web Interface
    print("💻 Starting Gradio Interface...")
    app = build_app()
    
    # Generates a shareable URL (share=True) for public web access
    app.launch(share=True, inline=False, allowed_paths=[get_working_dir()])

if __name__ == "__main__":
    main()
