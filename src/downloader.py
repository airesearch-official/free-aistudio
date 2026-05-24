import os
import shutil
import urllib.request
import tarfile
import subprocess
import glob

# Configuration for GitHub Releases binary
DEFAULT_REPO = "parijha/free-aistudio"
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

def setup_aria2():
    """Installs aria2 on Kaggle if not already present."""
    if shutil.which("aria2c") is None:
        print("📥 Installing aria2 high-speed download framework...")
        subprocess.run("apt-get update -qq && apt-get install -y -qq aria2", shell=True)
    else:
        print("✅ aria2 framework is already installed.")

def download_models(preset="LTX-Video-2.3-Q3", models_base="/tmp/models"):
    """Downloads weights for a selected preset using aria2c."""
    if preset not in MODEL_PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Available: {list(MODEL_PRESETS.keys())}")
        
    setup_aria2()
    config = MODEL_PRESETS[preset]
    
    print(f"\n--- ⚡ Starting weight downloads for preset: {preset} ---")
    for category, urls in config.items():
        cat_dir = os.path.join(models_base, category)
        os.makedirs(cat_dir, exist_ok=True)
        
        for url in urls:
            filename = url.split("/")[-1]
            print(f"→ Downloading {filename} to {cat_dir}...")
            # aria2c command for fast multi-threaded download
            cmd = f'aria2c -x 16 -s 16 -k 1M -d "{cat_dir}" "{url}"'
            subprocess.run(cmd, shell=True)
            
    print("\n🧹 Correcting model filename path structures (checking hashes)...")
    clean_filenames(models_base)
    print("✅ Weights setup complete.")

def clean_filenames(models_base="/tmp/models"):
    """Corrects file names if huggingface redirects named files as hashes."""
    # 1. Main Base Model Mapping
    dit_files = glob.glob(os.path.join(models_base, "diffusion_models/*"))
    if dit_files and not dit_files[0].endswith(".gguf"):
        os.rename(dit_files[0], os.path.join(models_base, "diffusion_models/ltx-2.3-22b-distilled-1.1-Q3_K_M.gguf"))
        print("Mapped DiT model name.")

    # 2. Text Encoder & Connectors Sorting
    te_files = sorted(glob.glob(os.path.join(models_base, "text_encoders/*")), key=os.path.getsize)
    if len(te_files) >= 2:
        if not te_files[0].endswith(".safetensors"):
            os.rename(te_files[0], os.path.join(models_base, "text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors"))
        if not te_files[1].endswith(".gguf"):
            os.rename(te_files[1], os.path.join(models_base, "text_encoders/gemma-3-12b-it-UD-IQ2_XXS.gguf"))
        print("Mapped Text Encoder & Connectors names.")

    # 3. VAE Folder Sorting
    vae_files = sorted(glob.glob(os.path.join(models_base, "vae/*")), key=os.path.getsize)
    if len(vae_files) >= 2:
        if not vae_files[0].endswith(".safetensors"):
            os.rename(vae_files[0], os.path.join(models_base, "vae/ltx-2.3-22b-distilled_audio_vae.safetensors"))
        if not vae_files[1].endswith(".safetensors"):
            os.rename(vae_files[1], os.path.join(models_base, "vae/ltx-2.3-22b-distilled_video_vae.safetensors"))
        print("Mapped VAE model names.")

    # 4. Latent Spatial Upscaler Correction
    upscale_files = glob.glob(os.path.join(models_base, "latent_upscale_models/*"))
    if upscale_files and not upscale_files[0].endswith(".safetensors"):
        os.rename(upscale_files[0], os.path.join(models_base, "latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"))
        print("Mapped Spatial Upscaler name.")
