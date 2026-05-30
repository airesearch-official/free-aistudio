import os
import subprocess
import time

def start_server(
    preset="LTX-Video-2.3-Q3",
    bin_path="/tmp/sd_bin/bin/sd-server",
    models_base="/tmp/models",
    load_audio_vae=True,
    log_path="/kaggle/working/server.log",
    port=1234,
    threads=4
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
            "--diffusion-fa",
            "--offload-to-cpu",
            "--vae-tiling",
            "-v",
        ]
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
            "--diffusion-fa",
            "-v",
        ]
    else:
        raise ValueError(f"Unknown preset: {preset}")
        
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "w")
    
    process = subprocess.Popen(server_cmd, stdout=log_file, stderr=subprocess.STDOUT)
    
    print("⏱️ Waiting 90 seconds for models to load into RAM/VRAM...")
    time.sleep(90)
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
