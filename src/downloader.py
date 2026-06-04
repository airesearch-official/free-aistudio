import os
import shutil
import urllib.request
import tarfile
import subprocess
import glob

# Configuration for GitHub Releases binary
DEFAULT_REPO = "airesearch-official/free-aistudio"
DEFAULT_TAG = "v1.0.0"
BINARY_FILENAME = "sd_cpp_cuda_built.tar.gz"

# Model presets containing component downloads
MODEL_PRESETS = {
    "LTX-Video-2.3-Q3": {
        "diffusion_models": [
            "https://huggingface.co/unsloth/LTX-2.3-GGUF/resolve/main/distilled-1.1/ltx-2.3-22b-distilled-1.1-Q3_K_M.gguf"
        ],
        "text_encoders": [
            "https://huggingface.co/unsloth/gemma-3-12b-it-GGUF/resolve/main/gemma-3-12b-it-UD-IQ2_XXS.gguf",
            "https://huggingface.co/unsloth/LTX-2.3-GGUF/resolve/main/text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors"
        ],
        "vae": [
            "https://huggingface.co/unsloth/LTX-2.3-GGUF/resolve/main/vae/ltx-2.3-22b-distilled_video_vae.safetensors",
            "https://huggingface.co/unsloth/LTX-2.3-GGUF/resolve/main/vae/ltx-2.3-22b-distilled_audio_vae.safetensors"
        ],
        "latent_upscale_models": [
            "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
        ]
    },
    "LTX-Video-2.3-FP8": {
        "diffusion_models": [
            "https://huggingface.co/unsloth/LTX-2.3-GGUF/resolve/main/distilled-1.1/ltx-2.3-22b-distilled-1.1-Q8_0.gguf"
        ],
        "text_encoders": [
            "https://huggingface.co/GitMylo/LTX-2-comfy_gemma_fp8_e4m3fn/resolve/main/gemma_3_12B_it_fp8_e4m3fn.safetensors",
            "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/text_encoders/ltx-2.3_text_projection_bf16.safetensors"
        ],
        "vae": [
            "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_video_vae_bf16.safetensors",
            "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_audio_vae_bf16.safetensors",
            "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/taeltx2_3.safetensors"
        ],
        "latent_upscale_models": [
            "https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-spatial-upscaler-x2-1.0.safetensors"
        ]
    },
    "Z-Image-Turbo-Q4": {
        "diffusion_models": [
            "https://huggingface.co/unsloth/Z-Image-Turbo-GGUF/resolve/main/z-image-turbo-Q4_0.gguf"
        ],
        "text_encoders": [
            "https://huggingface.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
        ],
        "vae": [
            "https://huggingface.co/airesearch-official/z-image-turbo-vae/resolve/main/ae.safetensors"
        ]
    }
}

def restore_binary(repo=DEFAULT_REPO, tag=DEFAULT_TAG, target_dir="/tmp/sd_bin"):
    """Downloads the pre-compiled stable-diffusion.cpp tarball from GitHub Releases and extracts it."""
    url = f"https://github.com/{repo}/releases/download/{tag}/{BINARY_FILENAME}"
    
    os.makedirs(target_dir, exist_ok=True)
    tar_path = os.path.join(target_dir, BINARY_FILENAME)
    
    print(f"📥 Downloading stable-diffusion.cpp binary from: {url}...")
    try:
        urllib.request.urlretrieve(url, tar_path)
        print("📦 Unpacking execution binaries...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=target_dir)
            
        # Safeguard: if files were compressed without the 'bin/' subdirectory,
        # restructure them to 'bin/' so path mappings inside server.py remain clean.
        root_bin_check = os.path.join(target_dir, "sd-server")
        if os.path.exists(root_bin_check):
            print("📦 Binaries extracted at root. Restructuring to bin/ subdirectory...")
            bin_subdir = os.path.join(target_dir, "bin")
            os.makedirs(bin_subdir, exist_ok=True)
            for item in os.listdir(target_dir):
                if item in [BINARY_FILENAME, "bin"]:
                    continue
                shutil.move(os.path.join(target_dir, item), os.path.join(bin_subdir, item))
            
        # Give permission to binaries
        for bin_name in ["sd-cli", "sd-server"]:
            bin_path = os.path.join(target_dir, "bin", bin_name)
            if os.path.exists(bin_path):
                print(f"🔐 Setting execution permissions for {bin_name}...")
                os.chmod(bin_path, 0o755)
                
        print("🔥 SUCCESS: Engine fully restored and operational!")
    except Exception as e:
        print(f"❌ Error restoring binary: {e}")
        print("Please check if the GitHub Release tag exists and contains the required file.")

def python_download(url, dest_dir):
    """Fallback python downloader when aria2 is not available."""
    import urllib.request
    import time
    
    filename = url.split("/")[-1]
    dest_path = os.path.join(dest_dir, filename)
    
    print(f"📥 Downloading via Python fallback: {filename}...")
    
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            meta = response.info()
            file_size = int(meta.get("Content-Length", 0))
            
            chunk_size = 8 * 1024 * 1024  # 8MB chunks
            downloaded = 0
            start_time = time.time()
            last_print = start_time
            
            with open(dest_path, "wb") as f:
                while True:
                    buffer = response.read(chunk_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    f.write(buffer)
                    
                    current_time = time.time()
                    if current_time - last_print > 5:
                        speed = downloaded / (current_time - start_time) / (1024 * 1024)  # MB/s
                        percent = (downloaded / file_size) * 100 if file_size > 0 else 0
                        print(f"   ↳ {downloaded / (1024*1024):.1f} MB / {file_size / (1024*1024):.1f} MB ({percent:.1f}%) @ {speed:.2f} MB/s")
                        last_print = current_time
                        
            total_time = time.time() - start_time
            print(f"✅ Finished downloading {filename} in {total_time:.1f}s.")
    except Exception as e:
        print(f"❌ Error downloading {url} via Python fallback: {e}")
        if os.path.exists(dest_path):
            try:
                os.remove(dest_path)
            except Exception:
                pass
        raise e

def setup_aria2():
    """Installs aria2 if not already present, using sudo if required."""
    if shutil.which("aria2c") is not None:
        print("✅ aria2 framework is already installed.")
        return
        
    print("📥 Installing aria2 high-speed download framework...")
    try:
        is_root = (os.getuid() == 0)
    except AttributeError:
        is_root = True
        
    use_sudo = ""
    if not is_root and shutil.which("sudo") is not None:
        use_sudo = "sudo "
        
    cmd = f"{use_sudo}apt-get update -qq && {use_sudo}apt-get install -y -qq aria2"
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        print("⚠️ Failed to install aria2. Will automatically use Python fallback downloader.")

def download_models(preset="LTX-Video-2.3-Q3", models_base="/tmp/models"):
    """Downloads weights for a selected preset using aria2c (or Python fallback)."""
    if preset not in MODEL_PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Available: {list(MODEL_PRESETS.keys())}")
        
    setup_aria2()
    config = MODEL_PRESETS[preset]
    use_aria2 = shutil.which("aria2c") is not None
    
    print(f"\n--- ⚡ Starting weight downloads for preset: {preset} ---")
    for category, urls in config.items():
        cat_dir = os.path.join(models_base, category)
        os.makedirs(cat_dir, exist_ok=True)
        
        for url in urls:
            filename = url.split("/")[-1]
            dest_file = os.path.join(cat_dir, filename)
            
            # Skip if file already exists and is fully downloaded (not a small temp file)
            if os.path.exists(dest_file) and os.path.getsize(dest_file) > 10 * 1024 * 1024:
                print(f"✅ {filename} already exists. Skipping download.")
                continue
                
            print(f"→ Downloading {filename} to {cat_dir}...")
            if use_aria2:
                cmd = f'aria2c -x 16 -s 16 -k 1M -d "{cat_dir}" "{url}"'
                res = subprocess.run(cmd, shell=True)
                if res.returncode != 0:
                    print(f"⚠️ aria2c failed for {filename}. Trying Python fallback...")
                    python_download(url, cat_dir)
            else:
                python_download(url, cat_dir)
            
    print("\n🧹 Correcting model filename path structures (checking hashes)...")
    clean_filenames(preset, models_base)
    print("✅ Weights setup complete.")

def clean_filenames(preset="LTX-Video-2.3-Q3", models_base="/tmp/models"):
    """Corrects file names if huggingface redirects named files as hashes."""
    if preset == "LTX-Video-2.3-Q3":
        # 1. Main Base Model Mapping
        dit_files = glob.glob(os.path.join(models_base, "diffusion_models/*"))
        if dit_files and not dit_files[0].endswith(".gguf"):
            os.rename(dit_files[0], os.path.join(models_base, "diffusion_models/ltx-2.3-22b-distilled-1.1-Q3_K_M.gguf"))
            print("Mapped LTX-Video DiT model name.")

        # 2. Text Encoder & Connectors Sorting
        te_files = sorted(glob.glob(os.path.join(models_base, "text_encoders/*")), key=os.path.getsize)
        if len(te_files) >= 2:
            if not te_files[0].endswith(".safetensors"):
                os.rename(te_files[0], os.path.join(models_base, "text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors"))
            if not te_files[1].endswith(".gguf"):
                os.rename(te_files[1], os.path.join(models_base, "text_encoders/gemma-3-12b-it-UD-IQ2_XXS.gguf"))
            print("Mapped LTX-Video Text Encoder & Connectors names.")

        # 3. VAE Folder Sorting
        vae_files = sorted(glob.glob(os.path.join(models_base, "vae/*")), key=os.path.getsize)
        if len(vae_files) >= 2:
            if not vae_files[0].endswith(".safetensors"):
                os.rename(vae_files[0], os.path.join(models_base, "vae/ltx-2.3-22b-distilled_audio_vae.safetensors"))
            if not vae_files[1].endswith(".safetensors"):
                os.rename(vae_files[1], os.path.join(models_base, "vae/ltx-2.3-22b-distilled_video_vae.safetensors"))
            print("Mapped LTX-Video VAE model names.")

        # 4. Latent Spatial Upscaler Correction
        upscale_files = glob.glob(os.path.join(models_base, "latent_upscale_models/*"))
        if upscale_files and not upscale_files[0].endswith(".safetensors"):
            os.rename(upscale_files[0], os.path.join(models_base, "latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"))
            print("Mapped Spatial Upscaler name.")

    elif preset == "LTX-Video-2.3-FP8":
        # 1. Transformer / UNet Model Mapping
        dit_files = glob.glob(os.path.join(models_base, "diffusion_models/*"))
        if dit_files and not dit_files[0].endswith(".gguf"):
            os.rename(dit_files[0], os.path.join(models_base, "diffusion_models/ltx-2.3-22b-distilled-1.1-Q8_0.gguf"))
            print("Mapped LTX-Video FP8 Transformer model name.")

        # 2. Text Encoder & Connectors Sorting
        te_files = sorted(glob.glob(os.path.join(models_base, "text_encoders/*")), key=os.path.getsize)
        if len(te_files) >= 2:
            if not te_files[0].endswith(".safetensors"):
                os.rename(te_files[0], os.path.join(models_base, "text_encoders/ltx-2.3_text_projection_bf16.safetensors"))
            if not te_files[1].endswith(".safetensors"):
                os.rename(te_files[1], os.path.join(models_base, "text_encoders/gemma_3_12B_it_fp8_e4m3fn.safetensors"))
            print("Mapped LTX-Video FP8 Text Encoder & Connectors names.")

        # 3. VAE Folder Sorting
        vae_files = sorted(glob.glob(os.path.join(models_base, "vae/*")), key=os.path.getsize)
        if len(vae_files) >= 3:
            if not vae_files[0].endswith(".safetensors"):
                os.rename(vae_files[0], os.path.join(models_base, "vae/taeltx2_3.safetensors"))
            if not vae_files[1].endswith(".safetensors"):
                os.rename(vae_files[1], os.path.join(models_base, "vae/LTX23_audio_vae_bf16.safetensors"))
            if not vae_files[2].endswith(".safetensors"):
                os.rename(vae_files[2], os.path.join(models_base, "vae/LTX23_video_vae_bf16.safetensors"))
            print("Mapped LTX-Video FP8 VAE model names.")
        elif len(vae_files) == 2:
            if not vae_files[0].endswith(".safetensors"):
                os.rename(vae_files[0], os.path.join(models_base, "vae/LTX23_audio_vae_bf16.safetensors"))
            if not vae_files[1].endswith(".safetensors"):
                os.rename(vae_files[1], os.path.join(models_base, "vae/LTX23_video_vae_bf16.safetensors"))
            print("Mapped 2 LTX-Video FP8 VAE model names.")

        # 4. Latent Spatial Upscaler Correction
        upscale_files = glob.glob(os.path.join(models_base, "latent_upscale_models/*"))
        if upscale_files and not upscale_files[0].endswith(".safetensors"):
            os.rename(upscale_files[0], os.path.join(models_base, "latent_upscale_models/ltx-2-spatial-upscaler-x2-1.0.safetensors"))
            print("Mapped Spatial Upscaler 1.0 name.")

    elif preset == "Z-Image-Turbo-Q4":
        # 1. Main Base Model Mapping
        dit_files = glob.glob(os.path.join(models_base, "diffusion_models/*"))
        if dit_files and not dit_files[0].endswith(".gguf"):
            os.rename(dit_files[0], os.path.join(models_base, "diffusion_models/z-image-turbo-Q4_0.gguf"))
            print("Mapped Z-Image-Turbo GGUF model name.")

        # 2. Text Encoder Mapping
        te_files = glob.glob(os.path.join(models_base, "text_encoders/*"))
        if te_files and not te_files[0].endswith(".gguf"):
            os.rename(te_files[0], os.path.join(models_base, "text_encoders/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"))
            print("Mapped Qwen text encoder name.")

        # 3. VAE Mapping
        vae_files = glob.glob(os.path.join(models_base, "vae/*"))
        if vae_files and not vae_files[0].endswith(".safetensors"):
            os.rename(vae_files[0], os.path.join(models_base, "vae/ae.safetensors"))
            print("Mapped Flux VAE name.")

