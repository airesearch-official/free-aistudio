import os
import shutil
import urllib.request
import tarfile
import subprocess
import glob
import json

# Configuration for GitHub Releases binary
DEFAULT_REPO = "airesearch-official/free-aistudio"
DEFAULT_TAG = "v1.0.0"
BINARY_FILENAME = "sd_cpp_cuda_built.tar.gz"
LIGHTNING_SDC_REPO = "https://github.com/leejet/stable-diffusion.cpp.git"
LIGHTNING_SDC_TAG = "master-672-1f9ee88"

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
        raise

def build_binary_from_source(
    target_dir="/teamspace/studios/this_studio/sd_bin",
    repo_url=LIGHTNING_SDC_REPO,
    tag=LIGHTNING_SDC_TAG,
    force=False,
):
    """Builds stable-diffusion.cpp with CUDA for Lightning.ai and caches the result."""
    bin_dir = os.path.join(target_dir, "bin")
    server_bin = os.path.join(bin_dir, "sd-server")
    build_info_path = os.path.join(target_dir, "build_info.json")

    if os.path.exists(server_bin) and os.path.exists(build_info_path) and not force:
        try:
            with open(build_info_path, "r") as f:
                build_info = json.load(f)
            if build_info.get("repo_url") == repo_url and build_info.get("tag") == tag:
                print(f"Using cached Lightning stable-diffusion.cpp build: {tag}")
                return
        except Exception:
            pass

    print(f"Building stable-diffusion.cpp for Lightning CUDA from {tag}...")
    os.makedirs(target_dir, exist_ok=True)
    source_dir = os.path.join(target_dir, "stable-diffusion.cpp")
    build_dir = os.path.join(source_dir, "build")

    if not os.path.exists(source_dir):
        subprocess.check_call(["git", "clone", "--recursive", repo_url, source_dir])

    subprocess.check_call(["git", "fetch", "--tags", "origin"], cwd=source_dir)
    subprocess.check_call(["git", "checkout", tag], cwd=source_dir)
    subprocess.check_call(["git", "submodule", "update", "--init", "--recursive"], cwd=source_dir)

    os.makedirs(build_dir, exist_ok=True)
    subprocess.check_call(
        ["cmake", "..", "-DSD_CUDA=ON", "-DCMAKE_BUILD_TYPE=Release"],
        cwd=build_dir,
    )
    subprocess.check_call(
        ["cmake", "--build", ".", "--config", "Release", "--parallel"],
        cwd=build_dir,
    )

    candidates = glob.glob(os.path.join(build_dir, "**", "sd-server"), recursive=True)
    candidates += glob.glob(os.path.join(source_dir, "bin", "sd-server"))
    candidates = [p for p in candidates if os.path.isfile(p)]
    if not candidates:
        raise FileNotFoundError("Built sd-server binary was not found after stable-diffusion.cpp build.")

    built_server = candidates[0]
    built_bin_dir = os.path.dirname(built_server)
    os.makedirs(bin_dir, exist_ok=True)
    for item in glob.glob(os.path.join(built_bin_dir, "*")):
        if os.path.isfile(item):
            dest = os.path.join(bin_dir, os.path.basename(item))
            shutil.copy2(item, dest)
            os.chmod(dest, 0o755)

    with open(build_info_path, "w") as f:
        json.dump({"repo_url": repo_url, "tag": tag}, f, indent=2)

    print(f"Lightning CUDA engine build complete: {server_bin}")

def remote_file_size(url):
    """Returns the remote file size when the host provides it."""
    try:
        req = urllib.request.Request(
            url,
            method="HEAD",
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            size = response.headers.get("Content-Length")
            return int(size) if size else None
    except Exception as e:
        print(f"Warning: could not verify remote size for {url}: {e}")
        return None

def is_download_complete(url, dest_file):
    """Checks whether an existing model file is likely complete."""
    if not os.path.exists(dest_file):
        return False

    local_size = os.path.getsize(dest_file)
    if local_size <= 10 * 1024 * 1024:
        return False

    expected_size = remote_file_size(url)
    if expected_size is None:
        return True

    if local_size == expected_size:
        return True

    print(
        f"Warning: {os.path.basename(dest_file)} has size {local_size} bytes, "
        f"expected {expected_size}. Re-downloading."
    )
    return False

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
            
            # Self-healing: If a GGUF file is actually a safetensors file (due to previous renaming bugs)
            if dest_file.endswith(".gguf") and os.path.exists(dest_file):
                try:
                    with open(dest_file, "rb") as f:
                        header = f.read(4)
                        # GGUF files start with "GGUF" magic bytes (0x47, 0x47, 0x55, 0x46)
                        if header != b"GGUF":
                            print(f"⚠️ Warning: Detected corrupted/mismatched GGUF file format for {filename}. Deleting and re-downloading...")
                            os.remove(dest_file)
                except Exception as e:
                    print(f"Error checking file header: {e}")

            # Skip if file already exists and is fully downloaded (not a small temp file)
            if is_download_complete(url, dest_file):
                print(f"✅ {filename} already exists. Skipping download.")
                continue
                
            print(f"→ Downloading {filename} to {cat_dir}...")
            if os.path.exists(dest_file):
                print(f"Warning: removing incomplete download before retrying: {dest_file}")
                os.remove(dest_file)

            if use_aria2:
                cmd = f'aria2c -x 16 -s 16 -k 1M -d "{cat_dir}" -o "{filename}" "{url}"'
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
        if dit_files and not dit_files[0].endswith(".gguf") and not dit_files[0].endswith(".safetensors"):
            os.rename(dit_files[0], os.path.join(models_base, "diffusion_models/ltx-2.3-22b-distilled-1.1-Q8_0.gguf"))
            print("Mapped LTX-Video FP8 Transformer model name.")

        # 2. Text Encoder & Connectors Sorting
        te_files = sorted(glob.glob(os.path.join(models_base, "text_encoders/*")), key=os.path.getsize)
        if len(te_files) >= 2:
            if not te_files[0].endswith(".safetensors"):
                os.rename(te_files[0], os.path.join(models_base, "text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors"))
            if not te_files[1].endswith(".safetensors"):
                os.rename(te_files[1], os.path.join(models_base, "text_encoders/gemma-3-12b-it-UD-IQ2_XXS.gguf"))
            print("Mapped LTX-Video FP8 Text Encoder & Connectors names.")

        # 3. VAE Folder Sorting
        vae_files = sorted(glob.glob(os.path.join(models_base, "vae/*")), key=os.path.getsize)
        if len(vae_files) >= 3:
            if not vae_files[0].endswith(".safetensors"):
                os.rename(vae_files[0], os.path.join(models_base, "vae/ltx-2.3-22b-distilled_audio_vae.safetensors"))
            if not vae_files[1].endswith(".safetensors"):
                os.rename(vae_files[1], os.path.join(models_base, "vae/ltx-2.3-22b-distilled_video_vae.safetensors"))
            if not vae_files[2].endswith(".safetensors"):
                os.rename(vae_files[2], os.path.join(models_base, "vae/taeltx2_3.safetensors"))
            print("Mapped LTX-Video FP8 VAE model names.")
        elif len(vae_files) == 2:
            if not vae_files[0].endswith(".safetensors"):
                os.rename(vae_files[0], os.path.join(models_base, "vae/ltx-2.3-22b-distilled_audio_vae.safetensors"))
            if not vae_files[1].endswith(".safetensors"):
                os.rename(vae_files[1], os.path.join(models_base, "vae/ltx-2.3-22b-distilled_video_vae.safetensors"))
            print("Mapped 2 LTX-Video FP8 VAE model names.")

        # 4. Latent Spatial Upscaler Correction
        upscale_files = glob.glob(os.path.join(models_base, "latent_upscale_models/*"))
        if upscale_files and not upscale_files[0].endswith(".safetensors"):
            os.rename(upscale_files[0], os.path.join(models_base, "latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"))
            print("Mapped Spatial Upscaler 1.1 name.")

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
