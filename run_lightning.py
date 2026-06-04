import os
import sys
import subprocess
import signal
import time

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

from src.downloader import restore_binary, download_models, build_binary_from_source, LIGHTNING_SDC_TAG
from src.server import start_server
from src.ui import build_app, get_working_dir

def detect_cuda_vram_gb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        values = [int(line.strip()) for line in out.splitlines() if line.strip()]
        if values:
            return max(values) / 1024
    except Exception:
        pass
    return 0

def main():
    print("🚀 Starting AI Studio on Lightning.ai...")
    
    # Setup paths for Lightning.ai persistent workspace storage
    models_base = "/teamspace/studios/this_studio/models"
    bin_dir = "/teamspace/studios/this_studio/sd_bin"
    bin_path = os.path.join(bin_dir, "bin/sd-server")
    log_path = "/teamspace/studios/this_studio/server.log"
    full_gpu_env = os.environ.get("FREE_AISTUDIO_LIGHTNING_FULL_GPU", "").lower()
    force_cpu_offload_env = os.environ.get("FREE_AISTUDIO_LIGHTNING_CPU_OFFLOAD", "").lower()
    detected_vram_gb = detect_cuda_vram_gb()
    full_gpu = (
        full_gpu_env in ("1", "true", "yes", "on")
        or (not full_gpu_env and force_cpu_offload_env not in ("1", "true", "yes", "on") and detected_vram_gb >= 35)
    )
    load_audio_vae = os.environ.get("FREE_AISTUDIO_DISABLE_AUDIO_VAE", "").lower() not in ("1", "true", "yes", "on")
    diffusion_fa = os.environ.get("FREE_AISTUDIO_LIGHTNING_DIFFUSION_FA", "").lower() in ("1", "true", "yes", "on")
    wait_timeout = int(os.environ.get("FREE_AISTUDIO_SERVER_WAIT_TIMEOUT", "300"))
    use_upstream_build = os.environ.get("FREE_AISTUDIO_LIGHTNING_USE_RELEASE_BINARY", "").lower() not in ("1", "true", "yes", "on")
    force_rebuild = os.environ.get("FREE_AISTUDIO_LIGHTNING_FORCE_REBUILD", "").lower() in ("1", "true", "yes", "on")
    sdc_tag = os.environ.get("FREE_AISTUDIO_LIGHTNING_SDC_TAG", LIGHTNING_SDC_TAG)
    
    # 1. Restore the C++ compilation binary if missing
    if use_upstream_build:
        build_binary_from_source(target_dir=bin_dir, tag=sdc_tag, force=force_rebuild)
    elif not os.path.exists(bin_path):
        restore_binary(repo="airesearch-official/free-aistudio", tag="v1.0.0", target_dir=bin_dir)
        
    # 2. Download LTX-Video FP8 weights (downloader skips already completed downloads)
    download_models(preset="LTX-Video-2.3-FP8", models_base=models_base)
    
    # 3. Start the background stable-diffusion.cpp inference server
    if full_gpu:
        print(f"Running Lightning GGUF Q8 preset on GPU. Detected VRAM: {detected_vram_gb:.1f}GB.")
    else:
        print(f"Running Lightning GGUF Q8 preset with CPU offload. Detected VRAM: {detected_vram_gb:.1f}GB.")
    if diffusion_fa:
        print("Diffusion flash-attention is enabled for Lightning.")
    else:
        print("Diffusion flash-attention is disabled for Lightning to avoid CUDA kernel failures during video generation.")

    server_process = start_server(
        preset="LTX-Video-2.3-FP8",
        bin_path=bin_path,
        models_base=models_base,
        log_path=log_path,
        load_audio_vae=load_audio_vae,
        offload_to_cpu=not full_gpu,
        wait_timeout=wait_timeout,
        fail_on_timeout=True,
        diffusion_fa=diffusion_fa
    )
    
    # Handler for clean exit (preventing GPU process leaks)
    def handle_shutdown(signum, frame):
        print("\n🛑 Shutting down server and UI processes...")
        try:
            server_process.terminate()
            server_process.wait(timeout=5)
            print("✅ Background engine server stopped successfully.")
        except Exception as e:
            print(f"⚠️ Warning during engine shutdown: {e}")
        sys.exit(0)
        
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    # 4. Launch Gradio Web Interface
    print("💻 Starting Gradio Interface...")
    app = build_app()
    
    # Generates a shareable URL (share=True) for public web access
    app.launch(share=True, inline=False, allowed_paths=[get_working_dir()])
    
    # Keep the main thread alive and monitor the backend engine process
    try:
        while True:
            if server_process.poll() is not None:
                print("❌ C++ engine server has crashed or stopped! Check logs:")
                with open(log_path, "r") as f:
                    print("".join(f.readlines()[-30:]))
                break
            time.sleep(5)
    except KeyboardInterrupt:
        handle_shutdown(None, None)

if __name__ == "__main__":
    main()
