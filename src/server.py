import os
import subprocess
import time
import sys

def find_cuda_library_paths():
    """Gathers all potential CUDA and Nvidia library paths in the environment."""
    import glob
    paths = []
    
    # 1. Active Conda/Python environment paths
    conda_prefix = os.environ.get("CONDA_PREFIX", sys.prefix)
    if conda_prefix:
        paths.append(os.path.join(conda_prefix, "lib"))
        
    # 2. Nvidia Pip packages in site-packages (where pip installs CUDA runtime/cublas)
    for sp in sys.path:
        if "site-packages" in sp:
            # Match directories like site-packages/nvidia/cuda_runtime/lib, site-packages/nvidia/cublas/lib, etc.
            pattern = os.path.join(sp, "nvidia", "*", "lib")
            for lib_dir in glob.glob(pattern):
                if os.path.isdir(lib_dir):
                    paths.append(lib_dir)
            # Also search for torch/lib which contains PyTorch's internal CUDA libraries
            torch_lib = os.path.join(sp, "torch", "lib")
            if os.path.isdir(torch_lib):
                paths.append(torch_lib)
                    
    # 3. Standard system CUDA paths
    system_paths = [
        "/usr/local/cuda/lib64",
        "/usr/local/cuda-12/lib64",
        "/usr/lib/x86_64-linux-gnu"
    ]
    paths.extend(system_paths)
    
    # 4. Glob search under /usr/local for any other CUDA installations
    for path in glob.glob("/usr/local/cuda-12.*/lib64"):
        paths.append(path)
    for path in glob.glob("/usr/local/cuda-*/lib64"):
        paths.append(path)
        
    # Filter only existing directories and remove duplicates
    unique_paths = []
    for p in paths:
        if p and os.path.isdir(p) and p not in unique_paths:
            unique_paths.append(p)
            
    return unique_paths

def start_server(
    preset="LTX-Video-2.3-Q3",
    bin_path="/tmp/sd_bin/bin/sd-server",
    models_base="/tmp/models",
    load_audio_vae=True,
    log_path="/kaggle/working/server.log",
    port=1234,
    threads=4,
    offload_to_cpu=False,
    wait_timeout=120,
    fail_on_timeout=False,
    diffusion_fa=True
):
    """Spawns the stable-diffusion.cpp API server in the background and saves logs."""
    
    if preset == "LTX-Video-2.3-Q3":
        upscaler_model = os.path.join(models_base, "latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
        upscaler_dir = os.path.dirname(upscaler_model)
        
        required_paths = [
            bin_path,
            os.path.join(models_base, "diffusion_models/ltx-2.3-22b-distilled-1.1-Q3_K_M.gguf"),
            os.path.join(models_base, "vae/ltx-2.3-22b-distilled_video_vae.safetensors"),
            os.path.join(models_base, "text_encoders/gemma-3-12b-it-UD-IQ2_XXS.gguf"),
            os.path.join(models_base, "text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors"),
            upscaler_model,
        ]
        
        if load_audio_vae:
            required_paths.append(os.path.join(models_base, "vae/ltx-2.3-22b-distilled_audio_vae.safetensors"))
            
        missing = [p for p in required_paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                "Missing required files for LTX-Video:\n" + "\n".join(missing) +
                "\nPlease run the downloader first!"
            )
            
        print("Starting stable-diffusion.cpp API server with LTX-Video paths...")
        server_cmd = [
            bin_path,
            "--listen-ip", "127.0.0.1",
            "--listen-port", str(port),
            "--threads", str(threads),
            "--diffusion-model", os.path.join(models_base, "diffusion_models/ltx-2.3-22b-distilled-1.1-Q3_K_M.gguf"),
            "--vae", os.path.join(models_base, "vae/ltx-2.3-22b-distilled_video_vae.safetensors"),
            "--llm", os.path.join(models_base, "text_encoders/gemma-3-12b-it-UD-IQ2_XXS.gguf"),
            "--embeddings-connectors", os.path.join(models_base, "text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors"),
            "--hires-upscalers-dir", upscaler_dir,
            "--offload-to-cpu",
            "--vae-tiling",
            "-v",
        ]
        if diffusion_fa:
            server_cmd += ["--diffusion-fa"]
        if load_audio_vae:
            server_cmd += ["--audio-vae", os.path.join(models_base, "vae/ltx-2.3-22b-distilled_audio_vae.safetensors")]

    elif preset == "LTX-Video-2.3-FP8":
        upscaler_model = os.path.join(models_base, "latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
        upscaler_dir = os.path.dirname(upscaler_model)
        
        required_paths = [
            bin_path,
            os.path.join(models_base, "diffusion_models/ltx-2.3-22b-distilled-Q8_0.gguf"),
            os.path.join(models_base, "vae/ltx-2.3-22b-distilled_video_vae.safetensors"),
            os.path.join(models_base, "text_encoders/gemma-3-12b-it-Q6_K.gguf"),
            os.path.join(models_base, "text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors"),
            upscaler_model,
        ]
        
        if load_audio_vae:
            required_paths.append(os.path.join(models_base, "vae/ltx-2.3-22b-distilled_audio_vae.safetensors"))
            
        missing = [p for p in required_paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                "Missing required files for LTX-Video FP8:\n" + "\n".join(missing) +
                "\nPlease run the downloader first!"
            )
            
        print("Starting stable-diffusion.cpp API server with LTX-Video FP8 paths...")
        server_cmd = [
            bin_path,
            "--listen-ip", "127.0.0.1",
            "--listen-port", str(port),
            "--threads", str(threads),
            "--diffusion-model", os.path.join(models_base, "diffusion_models/ltx-2.3-22b-distilled-Q8_0.gguf"),
            "--vae", os.path.join(models_base, "vae/ltx-2.3-22b-distilled_video_vae.safetensors"),
            "--llm", os.path.join(models_base, "text_encoders/gemma-3-12b-it-Q6_K.gguf"),
            "--embeddings-connectors", os.path.join(models_base, "text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors"),
            "--hires-upscalers-dir", upscaler_dir,
            "--vae-tiling",
            "-v",
        ]
        if diffusion_fa:
            server_cmd += ["--diffusion-fa"]
        if offload_to_cpu:
            server_cmd += ["--offload-to-cpu"]
        if load_audio_vae:
            server_cmd += ["--audio-vae", os.path.join(models_base, "vae/ltx-2.3-22b-distilled_audio_vae.safetensors")]

    elif preset == "Z-Image-Turbo-Q4":
        lora_dir = os.path.join(models_base, "loras")
        os.makedirs(lora_dir, exist_ok=True)
        
        required_paths = [
            bin_path,
            os.path.join(models_base, "diffusion_models/z-image-turbo-Q4_0.gguf"),
            os.path.join(models_base, "vae/ae.safetensors"),
            os.path.join(models_base, "text_encoders/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"),
        ]
        
        missing = [p for p in required_paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                "Missing required files for Z-Image-Turbo:\n" + "\n".join(missing) +
                "\nPlease run the downloader first!"
            )
            
        print("Starting stable-diffusion.cpp API server with Z-Image-Turbo paths...")
        server_cmd = [
            bin_path,
            "--listen-ip", "127.0.0.1",
            "--listen-port", str(port),
            "--threads", str(threads),
            "--diffusion-model", os.path.join(models_base, "diffusion_models/z-image-turbo-Q4_0.gguf"),
            "--vae", os.path.join(models_base, "vae/ae.safetensors"),
            "--llm", os.path.join(models_base, "text_encoders/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"),
            "--lora-model-dir", lora_dir,
            "--vae-tiling",
            "-v",
        ]
        if diffusion_fa:
            server_cmd += ["--diffusion-fa"]
    else:
        raise ValueError(f"Unknown preset: {preset}")
        
    env = os.environ.copy()
    
    # Dynamically inject CUDA & Conda library paths for stable-diffusion.cpp runtime dependency resolution
    valid_paths = find_cuda_library_paths()
    
    # Search for libcudart.so.12 in the gathered paths and print debug information
    found_at = []
    for p in valid_paths:
        if os.path.exists(os.path.join(p, "libcudart.so.12")):
            found_at.append(p)
            
    if found_at:
        print(f"🎯 CUDA Runtime libcudart.so.12 found in: {found_at}")
    else:
        print("⚠️ Warning: libcudart.so.12 was not found in any checked directory! Inference may fail if CUDA is not globally installed.")
        print(f"Searched directories: {valid_paths}")
        
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    if existing_ld:
        env["LD_LIBRARY_PATH"] = ":".join(valid_paths) + ":" + existing_ld
    else:
        env["LD_LIBRARY_PATH"] = ":".join(valid_paths)
        
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "w")
    
    process = subprocess.Popen(
        server_cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env
    )
    
    print(f"⏱️ Waiting for API server to become responsive on port {port}...")
    start_time = time.time()
    while time.time() - start_time < wait_timeout:
        if process.poll() is not None:
            log_file.close()
            logs = tail_logs(log_path, line_count=40)
            raise RuntimeError(
                "The stable-diffusion.cpp server stopped during startup.\n"
                f"Exit code: {process.returncode}\n"
                f"Recent logs:\n{logs}"
            )
        try:
            import urllib.request
            # Check if capabilities endpoint is active (indicates model is fully loaded and listening)
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/sdcpp/v1/capabilities", timeout=2) as response:
                if response.status == 200:
                    print("🔥 API Server is up and ready!")
                    break
        except Exception:
            time.sleep(2)
    else:
        if fail_on_timeout:
            log_file.close()
            raise TimeoutError(
                "Timeout waiting for server response.\n"
                f"Recent logs:\n{tail_logs(log_path, line_count=40)}"
            )
        print("⚠️ Warning: Timeout waiting for server response. Proceeding anyway...")
        
    print(f"API Server active checks loaded. Preset: {preset}")
    print(f"Logging active in: {log_path}")
    return process

def tail_logs(log_path="/kaggle/working/server.log", line_count=20):
    """Utility to print the last few lines of the server logs."""
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            lines = f.readlines()
            return "".join(lines[-line_count:])
    return "Waiting for server logs to initialize..."
