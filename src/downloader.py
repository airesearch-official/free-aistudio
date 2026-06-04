import os
import sys
import shutil
import urllib.request
import tarfile
import subprocess
import glob
import json
import zipfile

def safe_rename(src, dst):
    """Safely renames a file only if the source and target are different paths."""
    src_abs = os.path.abspath(src)
    dst_abs = os.path.abspath(dst)
    if src_abs == dst_abs:
        return
    print(f"Renaming {src_abs} to {dst_abs}...")
    if os.path.exists(dst_abs):
        try:
            if os.path.isdir(dst_abs):
                shutil.rmtree(dst_abs)
            else:
                os.remove(dst_abs)
        except Exception as e:
            print(f"Warning: Could not remove existing file/directory {dst_abs}: {e}")
    try:
        os.rename(src_abs, dst_abs)
    except Exception as e:
        print(f"Error: Failed to rename {src_abs} to {dst_abs}: {e}")

# Configuration for GitHub Releases binary
DEFAULT_REPO = "airesearch-official/free-aistudio"
DEFAULT_TAG = "v1.0.0"
BINARY_FILENAME = "sd_cpp_cuda_built.tar.gz"
LIGHTNING_BINARY_FILENAME = "sd-cpp-linux-cuda-a100-colab-build.zip"
LIGHTNING_SDC_REPO = "https://github.com/leejet/stable-diffusion.cpp.git"
LIGHTNING_SDC_TAG = "master-672-1f9ee88"

def detect_cuda_architecture(default="80"):
    """Returns a CMake CUDA architecture value such as 80 for A100."""
    env_arch = os.environ.get("FREE_AISTUDIO_CUDA_ARCH")
    if env_arch:
        return env_arch

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        caps = [line.strip().replace(".", "") for line in out.splitlines() if line.strip()]
        if caps:
            return caps[0]
    except Exception:
        pass

    return default

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
            "https://huggingface.co/unsloth/LTX-2.3-GGUF/resolve/main/distilled/ltx-2.3-22b-distilled-Q8_0.gguf"
        ],
        "text_encoders": [
            "https://huggingface.co/unsloth/gemma-3-12b-it-GGUF/resolve/main/gemma-3-12b-it-Q6_K.gguf",
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

def restore_lightning_binary(
    repo=DEFAULT_REPO,
    tag=DEFAULT_TAG,
    target_dir="/teamspace/studios/this_studio/sd_bin",
    filename=LIGHTNING_BINARY_FILENAME,
    force=False,
):
    """Downloads the Lightning CUDA zip release and installs binaries into target_dir/bin."""
    url = f"https://github.com/{repo}/releases/download/{tag}/{filename}"
    bin_dir = os.path.join(target_dir, "bin")
    server_bin = os.path.join(bin_dir, "sd-server")
    build_info_path = os.path.join(target_dir, "build_info.json")

    if os.path.exists(server_bin) and os.path.exists(build_info_path) and not force:
        try:
            with open(build_info_path, "r") as f:
                build_info = json.load(f)
            if (
                build_info.get("type") == "release-zip"
                and build_info.get("repo") == repo
                and build_info.get("tag") == tag
                and build_info.get("filename") == filename
            ):
                print(f"Using cached Lightning release binary: {filename}")
                return
        except Exception:
            pass

    os.makedirs(target_dir, exist_ok=True)
    zip_path = os.path.join(target_dir, filename)
    extract_dir = os.path.join(target_dir, "_lightning_release_extract")

    print(f"Downloading Lightning CUDA binary from: {url}...")
    urllib.request.urlretrieve(url, zip_path)

    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir, exist_ok=True)

    print("Unpacking Lightning CUDA binary zip...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_path = os.path.normpath(member.filename)
            if member_path.startswith("..") or os.path.isabs(member_path):
                raise ValueError(f"Unsafe path in zip file: {member.filename}")
            zf.extract(member, extract_dir)

    candidates = glob.glob(os.path.join(extract_dir, "**", "sd-server"), recursive=True)
    candidates = [p for p in candidates if os.path.isfile(p)]
    if not candidates:
        raise FileNotFoundError(f"sd-server was not found inside {filename}")

    built_server = candidates[0]
    built_bin_dir = os.path.dirname(built_server)
    os.makedirs(bin_dir, exist_ok=True)
    for item in glob.glob(os.path.join(built_bin_dir, "*")):
        if os.path.isfile(item):
            dest = os.path.join(bin_dir, os.path.basename(item))
            shutil.copy2(item, dest)
            os.chmod(dest, 0o755)

    with open(build_info_path, "w") as f:
        json.dump(
            {
                "type": "release-zip",
                "repo": repo,
                "tag": tag,
                "filename": filename,
            },
            f,
            indent=2,
        )

    print(f"Lightning CUDA release binary restored: {server_bin}")

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
    cuda_arch = detect_cuda_architecture()

    if os.path.exists(server_bin) and os.path.exists(build_info_path) and not force:
        try:
            with open(build_info_path, "r") as f:
                build_info = json.load(f)
            if build_info.get("repo_url") == repo_url and build_info.get("tag") == tag and build_info.get("cuda_arch") == cuda_arch:
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
        [
            "cmake",
            "..",
            "-DSD_CUDA=ON",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}",
        ],
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
        json.dump({"repo_url": repo_url, "tag": tag, "cuda_arch": cuda_arch}, f, indent=2)

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

def setup_ffmpeg():
    """Installs ffmpeg using imageio-ffmpeg or apt-get if not already present."""
    if shutil.which("ffmpeg") is not None:
        print("✅ ffmpeg is already installed.")
        return
        
    try:
        import imageio_ffmpeg
        print("✅ imageio-ffmpeg is already installed.")
        return
    except ImportError:
        pass

    print("📥 Installing imageio-ffmpeg static binary framework...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "imageio-ffmpeg>=0.4.9"])
        print("✅ Finished installing imageio-ffmpeg.")
        import imageio_ffmpeg
        return
    except Exception as e:
        print(f"⚠️ Failed to pip install imageio-ffmpeg: {e}")
        
    # Fallback to apt-get
    print("📥 Trying system apt-get install fallback...")
    try:
        is_root = (os.getuid() == 0)
    except AttributeError:
        is_root = True
        
    use_sudo = ""
    if not is_root and shutil.which("sudo") is not None:
        use_sudo = "sudo "
        
    cmd = f"{use_sudo}apt-get update -qq && {use_sudo}apt-get install -y -qq ffmpeg"
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        print("⚠️ Failed to install ffmpeg. Video previews might not show properly in the Gradio player.")

def download_models(preset="LTX-Video-2.3-Q3", models_base="/tmp/models"):
    """Downloads weights for a selected preset using aria2c (or Python fallback)."""
    if preset not in MODEL_PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Available: {list(MODEL_PRESETS.keys())}")
        
    setup_aria2()
    setup_ffmpeg()
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
        # 1. Main Base Model
        expected_dit = os.path.join(models_base, "diffusion_models/ltx-2.3-22b-distilled-1.1-Q3_K_M.gguf")
        if not os.path.exists(expected_dit):
            dit_files = glob.glob(os.path.join(models_base, "diffusion_models/*"))
            unrecognized = [f for f in dit_files if os.path.basename(f) != "ltx-2.3-22b-distilled-1.1-Q3_K_M.gguf"]
            if unrecognized:
                safe_rename(unrecognized[0], expected_dit)
                print("Mapped LTX-Video DiT model name.")

        # 2. Text Encoder & Connectors
        expected_conn = os.path.join(models_base, "text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors")
        expected_te = os.path.join(models_base, "text_encoders/gemma-3-12b-it-UD-IQ2_XXS.gguf")
        
        te_files = glob.glob(os.path.join(models_base, "text_encoders/*"))
        expected_basenames = ["ltx-2.3-22b-distilled_embeddings_connectors.safetensors", "gemma-3-12b-it-UD-IQ2_XXS.gguf"]
        unrecognized = sorted([f for f in te_files if os.path.basename(f) not in expected_basenames], key=os.path.getsize)
        
        if unrecognized:
            if not os.path.exists(expected_conn) and not os.path.exists(expected_te) and len(unrecognized) >= 2:
                safe_rename(unrecognized[0], expected_conn)
                safe_rename(unrecognized[1], expected_te)
            elif not os.path.exists(expected_conn):
                safe_rename(unrecognized[0], expected_conn)
            elif not os.path.exists(expected_te):
                safe_rename(unrecognized[-1], expected_te)
            print("Mapped LTX-Video Text Encoder & Connectors names.")

        # 3. VAE Folder
        expected_audio = os.path.join(models_base, "vae/ltx-2.3-22b-distilled_audio_vae.safetensors")
        expected_video = os.path.join(models_base, "vae/ltx-2.3-22b-distilled_video_vae.safetensors")
        
        vae_files = glob.glob(os.path.join(models_base, "vae/*"))
        expected_basenames = ["ltx-2.3-22b-distilled_audio_vae.safetensors", "ltx-2.3-22b-distilled_video_vae.safetensors"]
        unrecognized = sorted([f for f in vae_files if os.path.basename(f) not in expected_basenames], key=os.path.getsize)
        
        if unrecognized:
            if not os.path.exists(expected_audio) and not os.path.exists(expected_video) and len(unrecognized) >= 2:
                safe_rename(unrecognized[0], expected_audio)
                safe_rename(unrecognized[1], expected_video)
            elif not os.path.exists(expected_audio):
                safe_rename(unrecognized[0], expected_audio)
            elif not os.path.exists(expected_video):
                safe_rename(unrecognized[-1], expected_video)
            print("Mapped LTX-Video VAE model names.")

        # 4. Latent Spatial Upscaler
        expected_upscale = os.path.join(models_base, "latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
        if not os.path.exists(expected_upscale):
            upscale_files = glob.glob(os.path.join(models_base, "latent_upscale_models/*"))
            unrecognized = [f for f in upscale_files if os.path.basename(f) != "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"]
            if unrecognized:
                safe_rename(unrecognized[0], expected_upscale)
                print("Mapped Spatial Upscaler name.")

    elif preset == "LTX-Video-2.3-FP8":
        # 1. Main Base Model
        expected_dit = os.path.join(models_base, "diffusion_models/ltx-2.3-22b-distilled-Q8_0.gguf")
        if not os.path.exists(expected_dit):
            dit_files = glob.glob(os.path.join(models_base, "diffusion_models/*"))
            unrecognized = [f for f in dit_files if os.path.basename(f) != "ltx-2.3-22b-distilled-Q8_0.gguf"]
            if unrecognized:
                safe_rename(unrecognized[0], expected_dit)
                print("Mapped LTX-Video FP8 Transformer model name.")

        # 2. Text Encoder & Connectors
        expected_conn = os.path.join(models_base, "text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors")
        expected_te = os.path.join(models_base, "text_encoders/gemma-3-12b-it-Q6_K.gguf")
        
        te_files = glob.glob(os.path.join(models_base, "text_encoders/*"))
        expected_basenames = ["ltx-2.3-22b-distilled_embeddings_connectors.safetensors", "gemma-3-12b-it-Q6_K.gguf"]
        unrecognized = sorted([f for f in te_files if os.path.basename(f) not in expected_basenames], key=os.path.getsize)
        
        if unrecognized:
            if not os.path.exists(expected_conn) and not os.path.exists(expected_te) and len(unrecognized) >= 2:
                safe_rename(unrecognized[0], expected_conn)
                safe_rename(unrecognized[1], expected_te)
            elif not os.path.exists(expected_conn):
                safe_rename(unrecognized[0], expected_conn)
            elif not os.path.exists(expected_te):
                safe_rename(unrecognized[-1], expected_te)
            print("Mapped LTX-Video FP8 Text Encoder & Connectors names.")

        # 3. VAE Folder
        expected_audio = os.path.join(models_base, "vae/ltx-2.3-22b-distilled_audio_vae.safetensors")
        expected_video = os.path.join(models_base, "vae/ltx-2.3-22b-distilled_video_vae.safetensors")
        expected_tae = os.path.join(models_base, "vae/taeltx2_3.safetensors")
        
        vae_files = glob.glob(os.path.join(models_base, "vae/*"))
        expected_basenames = ["ltx-2.3-22b-distilled_audio_vae.safetensors", "ltx-2.3-22b-distilled_video_vae.safetensors", "taeltx2_3.safetensors"]
        unrecognized = sorted([f for f in vae_files if os.path.basename(f) not in expected_basenames], key=os.path.getsize)
        
        if unrecognized:
            if not os.path.exists(expected_audio) and not os.path.exists(expected_video) and len(unrecognized) >= 2:
                safe_rename(unrecognized[0], expected_audio)
                safe_rename(unrecognized[1], expected_video)
                if len(unrecognized) >= 3 and not os.path.exists(expected_tae):
                    safe_rename(unrecognized[2], expected_tae)
            elif not os.path.exists(expected_audio):
                safe_rename(unrecognized[0], expected_audio)
            elif not os.path.exists(expected_video):
                safe_rename(unrecognized[-1], expected_video)
            print("Mapped LTX-Video FP8 VAE model names.")

        # 4. Latent Spatial Upscaler
        expected_upscale = os.path.join(models_base, "latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
        if not os.path.exists(expected_upscale):
            upscale_files = glob.glob(os.path.join(models_base, "latent_upscale_models/*"))
            unrecognized = [f for f in upscale_files if os.path.basename(f) != "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"]
            if unrecognized:
                safe_rename(unrecognized[0], expected_upscale)
                print("Mapped Spatial Upscaler 1.1 name.")

    elif preset == "Z-Image-Turbo-Q4":
        # 1. Main Base Model
        expected_dit = os.path.join(models_base, "diffusion_models/z-image-turbo-Q4_0.gguf")
        if not os.path.exists(expected_dit):
            dit_files = glob.glob(os.path.join(models_base, "diffusion_models/*"))
            unrecognized = [f for f in dit_files if os.path.basename(f) != "z-image-turbo-Q4_0.gguf"]
            if unrecognized:
                safe_rename(unrecognized[0], expected_dit)
                print("Mapped Z-Image-Turbo GGUF model name.")

        # 2. Text Encoder
        expected_te = os.path.join(models_base, "text_encoders/Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
        if not os.path.exists(expected_te):
            te_files = glob.glob(os.path.join(models_base, "text_encoders/*"))
            unrecognized = [f for f in te_files if os.path.basename(f) != "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"]
            if unrecognized:
                safe_rename(unrecognized[0], expected_te)
                print("Mapped Qwen text encoder name.")

        # 3. VAE
        expected_vae = os.path.join(models_base, "vae/ae.safetensors")
        if not os.path.exists(expected_vae):
            vae_files = glob.glob(os.path.join(models_base, "vae/*"))
            unrecognized = [f for f in vae_files if os.path.basename(f) != "ae.safetensors"]
            if unrecognized:
                safe_rename(unrecognized[0], expected_vae)
                print("Mapped Flux VAE name.")
