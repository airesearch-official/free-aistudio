import os
from pathlib import Path
import glob
import time
import json
import base64
import requests
import warnings
import gradio as gr

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*browser-compatible container.*")

SERVER_URL = "http://127.0.0.1:1234"
LOG_PATH = "/kaggle/working/server.log"
UPSCALER_MODEL = "/tmp/models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
UPSCALER_MODEL_NAME = os.path.splitext(os.path.basename(UPSCALER_MODEL))[0]

def get_vae_tiling_params(enable_upscale):
    if enable_upscale:
        return {
            "enabled": True,
            "temporal_tiling": True,
            "tile_size_x": 16,
            "tile_size_y": 16,
            "target_overlap": 0.25,
            "rel_size_x": 0.0,
            "rel_size_y": 0.0,
            "extra_tiling_args": "temporal_tile_frames=4,temporal_tile_overlap=1",
        }
    return {
        "enabled": True,
        "temporal_tiling": False,
        "tile_size_x": 32,
        "tile_size_y": 32,
        "target_overlap": 0.5,
        "rel_size_x": 0.0,
        "rel_size_y": 0.0,
        "extra_tiling_args": "",
    }

def get_live_logs():
    """Reads the tail end of the server log file to stream into the interface."""
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r") as f:
            lines = f.readlines()
            return "".join(lines[-20:])
    return "Waiting for server logs to initialize..."

def scan_history():
    """Scans the working directory for generated video outputs."""
    video_files = glob.glob("/kaggle/working/gen_*.webm") + glob.glob("/kaggle/working/gen_*.avi")
    video_files.sort(key=os.path.getmtime, reverse=True)
    return video_files

def build_failure_message(status_res):
    """Turns the server job response and recent logs into a beginner-readable UI message."""
    parts = []
    for key in ("error", "message", "detail"):
        if status_res.get(key):
            parts.append(str(status_res[key]))

    result = status_res.get("result")
    if isinstance(result, dict):
        for key in ("error", "message", "detail"):
            if result.get(key):
                parts.append(str(result[key]))

    recent_logs = get_live_logs()
    important_logs = [
        line for line in recent_logs.splitlines()
        if "[ERROR]" in line or "[WARN" in line or "requires hires upscaler" in line.lower()
    ]
    if important_logs:
        parts.append("Recent engine log:\n" + "\n".join(important_logs[-6:]))
        lower_logs = "\n".join(important_logs).lower()
        if "out of memory" in lower_logs or "failed to allocate" in lower_logs:
            parts.append("What happened: the video was generated and upscaled, but final VAE decoding needed more GPU memory than available. Try the 360p preset, fewer frames, or keep the smaller upscale VAE tile settings in this notebook.")

    if not parts:
        parts.append(json.dumps(status_res, indent=2)[:2000])

    return "Video generation failed.\n\n" + "\n\n".join(parts)

def handle_generation(prompt, negative_prompt, steps, resolution_preset, use_custom_resolution, custom_width, custom_height, duration_seconds, input_image, enable_upscale):
    """Processes frontend inputs and posts generation parameters to the server."""
    if use_custom_resolution:
        width, height = int(custom_width), int(custom_height)
        if width % 32 != 0 or height % 32 != 0:
            raise gr.Error("Custom width and height must both be divisible by 32.")
        if width < 256 or height < 256:
            raise gr.Error("Custom width and height must be at least 256 pixels.")
        if width > 1920 or height > 1088:
            raise gr.Error("Custom resolution is capped at 1920x1088 for this Kaggle notebook.")
    elif "360p" in resolution_preset:
        width, height = 480, 360  # Proven fast baseline
    elif "480p" in resolution_preset:
        width, height = 640, 368  # Proven balanced size
    else:
        width, height = 832, 480

    fps = 12
    target_frames = max(9, int(round(float(duration_seconds) * fps)))
    frames = min(121, ((target_frames - 1 + 7) // 8) * 8 + 1)  # LTX video frame count rule: 8N + 1.

    payload = {
        "prompt": str(prompt),
        "negative_prompt": str(negative_prompt),
        "width": int(width),
        "height": int(height),
        "strength": 0.75 if input_image else 1.0,
        "seed": -1,
        "video_frames": int(frames),
        "fps": fps,
        "moe_boundary": 0.875,
        "vace_strength": 1.0,
        "sample_params": {
            "scheduler": "discrete",
            "sample_method": "euler",
            "sample_steps": int(steps),
            "flow_shift": 1.3568,
            "guidance": {"txt_cfg": 5.5, "img_cfg": 5.5, "distilled_guidance": 3.5},
        },
        "vae_tiling_params": get_vae_tiling_params(enable_upscale),
        "output_format": "avi",
        "output_compression": 100,
    }

    if "audio" not in payload["prompt"].lower():
        payload["prompt"] = f"{payload['prompt']}, high quality clear audio"

    if enable_upscale:
        if not os.path.exists(UPSCALER_MODEL):
            raise gr.Error(f"Upscaling is enabled, but the upscaler model is missing:\n{UPSCALER_MODEL}\n\nRun download step first.")

        payload["hires"] = {
            "enabled": True,
            "upscaler": UPSCALER_MODEL_NAME,
            "scale": 2.0,
            "steps": 10,
            "denoising_strength": 0.7,
        }

    if input_image is not None and os.path.exists(input_image):
        with open(input_image, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
        image_payload = f"data:image/png;base64,{img_base64}"
        payload["init_image"] = image_payload
        payload["input_image"] = image_payload

    try:
        r = requests.post(f"{SERVER_URL}/sdcpp/v1/vid_gen", json=payload, timeout=30)
        r.raise_for_status()
        job_id = r.json()["id"]

        while True:
            status_res = requests.get(f"{SERVER_URL}/sdcpp/v1/jobs/{job_id}", timeout=10).json()
            status = status_res.get("status", "unknown")

            if status == "completed":
                video_bytes = base64.b64decode(status_res["result"]["b64_json"])
                base_video_path = f"/kaggle/working/gen_{job_id}.avi"
                with open(base_video_path, "wb") as f:
                    f.write(video_bytes)
                return base_video_path

            if status in ("failed", "cancelled"):
                raise gr.Error(build_failure_message(status_res))

            time.sleep(4)

    except Exception as e:
        raise gr.Error(f"Could not communicate with the generation server.\n\n{type(e).__name__}: {e}\n\nRecent logs:\n{get_live_logs()}")

def build_app():
    """Constructs and returns the Gradio app blocks."""
    with gr.Blocks(theme=gr.themes.Soft()) as app:
        gr.Markdown("# LTX-Video 2.3 Studio Cloud Interface")
        gr.Markdown("Generate videos with the controls below. Use the engine logs tab for live progress and error details.")

        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(label="Text Prompt", placeholder="Describe the video actions and sounds clearly...")
                neg_prompt = gr.Textbox(label="Negative Prompt", value="blurry, worst quality, low quality, glitch, distortion")

                resolution_preset = gr.Dropdown(
                    choices=[
                        "360p (480x360) - Fastest Testing Baseline",
                        "480p (640x368) - Optimized Safe Balanced Size",
                        "720p (832x480) - High Resolution Cinematic Layout",
                    ],
                    value="480p (640x368) - Optimized Safe Balanced Size",
                    label="Core Video Generation Dimensions",
                )

                use_custom_resolution = gr.Checkbox(label="Use Custom Resolution", value=False)
                with gr.Row():
                    custom_width = gr.Slider(minimum=256, maximum=1920, value=640, step=32, label="Custom Width")
                    custom_height = gr.Slider(minimum=256, maximum=1088, value=384, step=32, label="Custom Height")

                duration_seconds = gr.Slider(minimum=1, maximum=10, value=5, step=0.5, label="Duration Seconds (rounded to valid LTX frame count)")
                steps = gr.Slider(minimum=4, maximum=30, value=8, step=1, label="Sampling Steps (LTX 2.3 Distilled Sweet Spot: 8-12)")

                enable_upscale = gr.Checkbox(label="Enable Native Hi-Res Upscaling Pass", value=False)
                input_image = gr.Image(label="Input Image (For Image-to-Video)", type="filepath")
                generate_btn = gr.Button("Generate New Video", variant="primary")

            with gr.Column(scale=1):
                output_video = gr.Video(label="Generated Result Screen")
                with gr.Tab("Active Engine Logs"):
                    log_box = gr.Textbox(label="Live C++ Terminal Output Stream", value="", lines=10, interactive=False)
                    log_timer = gr.Timer(value=3.0, active=True)
                    log_timer.tick(fn=get_live_logs, outputs=log_box)

                with gr.Tab("Generation History"):
                    refresh_history_btn = gr.Button("Refresh History Archive", variant="secondary")
                    history_gallery = gr.File(label="Generated Video Vault Files", file_count="multiple")

        generate_btn.click(
            fn=handle_generation,
            inputs=[prompt, neg_prompt, steps, resolution_preset, use_custom_resolution, custom_width, custom_height, duration_seconds, input_image, enable_upscale],
            outputs=output_video,
        ).then(fn=scan_history, outputs=history_gallery)

        refresh_history_btn.click(fn=scan_history, outputs=history_gallery)
        app.load(fn=scan_history, outputs=history_gallery)

    app.queue()
    return app

def launch():
    """Convenience function to start UI immediately in Kaggle."""
    app = build_app()
    app.launch(share=True, inline=False)

# =====================================================================
# Z-Image-Turbo Image Generation UI Components
# =====================================================================

LORA_DIR = "/tmp/models/loras"
RES_PRESETS = [
    ("1:1 (256x256)", 256, 256),
    ("1:1 (512x512)", 512, 512),
    ("1:1 (768x768)", 768, 768),
    ("1:1 (1024x1024)", 1024, 1024),
    ("16:9 (640x384)", 640, 384),
    ("16:9 (896x512)", 896, 512),
    ("16:9 (1024x576)", 1024, 576),
    ("9:16 (384x640)", 384, 640),
    ("9:16 (512x896)", 512, 896),
    ("9:16 (576x1024)", 576, 1024),
    ("4:3 (640x480)", 640, 480),
    ("4:3 (768x576)", 768, 576),
    ("3:2 (768x512)", 768, 512),
    ("2:3 (512x768)", 512, 768),
]
SIZE_OPTIONS = sorted({s for _, w, h in RES_PRESETS for s in (w, h)})

def get_lora_list():
    """List available LoRA files in the loras directory."""
    lora_path = Path(LORA_DIR)
    if not lora_path.exists():
        return []
    return [f.name for f in lora_path.glob("*.safetensors")]

def apply_preset(preset_label):
    for name, w, h in RES_PRESETS:
        if name == preset_label:
            return w, h
    return gr.update(), gr.update()

def scan_image_history():
    """Scans the working directory for generated image outputs."""
    image_files = glob.glob("/kaggle/working/gen_*.png")
    image_files.sort(key=os.path.getmtime, reverse=True)
    return image_files

def handle_image_generation(prompt, width, height, steps, seed, cfg_scale, selected_loras, lora_strength):
    """Processes image params and posts to the API server."""
    # Append LoRA tags to prompt
    final_prompt = prompt
    if selected_loras:
        from pathlib import Path
        for lora in selected_loras:
            lora_name = Path(lora).stem
            final_prompt += f" <lora:{lora_name}:{lora_strength}>"

    payload = {
        "prompt": str(final_prompt),
        "negative_prompt": "",
        "width": int(width),
        "height": int(height),
        "seed": int(seed) if int(seed) > 0 else -1,
        "sample_params": {
            "scheduler": "discrete",
            "sample_method": "euler",
            "sample_steps": int(steps),
            "cfg_scale": float(cfg_scale),
        },
        "output_format": "png",
        "output_compression": 100,
    }

    try:
        r = requests.post(f"{SERVER_URL}/sdcpp/v1/img_gen", json=payload, timeout=30)
        r.raise_for_status()
        job_id = r.json()["id"]

        while True:
            status_res = requests.get(f"{SERVER_URL}/sdcpp/v1/jobs/{job_id}", timeout=10).json()
            status = status_res.get("status", "unknown")

            if status == "completed":
                image_bytes = base64.b64decode(status_res["result"]["b64_json"])
                base_image_path = f"/kaggle/working/gen_{job_id}.png"
                with open(base_image_path, "wb") as f:
                    f.write(image_bytes)
                return base_image_path

            if status in ("failed", "cancelled"):
                raise gr.Error(build_failure_message(status_res))

            time.sleep(1.5) # Fast polling for image

    except Exception as e:
        raise gr.Error(f"Could not communicate with the generation server.\n\n{type(e).__name__}: {e}\n\nRecent logs:\n{get_live_logs()}")

def build_image_app():
    """Constructs and returns the Gradio app blocks for Z-Image-Turbo."""
    from pathlib import Path
    with gr.Blocks(theme=gr.themes.Soft()) as app:
        gr.Markdown("# Z-Image-Turbo Cloud Studio")
        gr.Markdown("Generate high-speed images using the stable-diffusion.cpp backend on Kaggle.")

        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(label="Prompt", value="A large orange octopus on an ocean floor, cinematic, 8k", lines=3)
                
                with gr.Row():
                    preset = gr.Dropdown([n for n, _, _ in RES_PRESETS], value="1:1 (512x512)", label="Resolution Preset")
                    steps = gr.Slider(1, 50, value=8, step=1, label="Steps")
                
                with gr.Row():
                    width = gr.Dropdown(SIZE_OPTIONS, value=512, label="Width")
                    height = gr.Dropdown(SIZE_OPTIONS, value=512, label="Height")
                
                with gr.Row():
                    cfg_scale = gr.Slider(0.0, 10.0, value=1.0, step=0.1, label="CFG Scale")
                    seed = gr.Number(value=0, label="Seed (0 = random)")
                
                with gr.Group():
                    gr.Markdown("### LoRA Support (Place inside `/tmp/models/loras/`)")
                    with gr.Row():
                        lora_list = gr.CheckboxGroup(choices=get_lora_list(), label="Select LoRAs")
                        refresh_btn = gr.Button("Refresh LoRAs", variant="secondary", size="sm")
                    with gr.Row():
                        lora_strength = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="LoRA Strength")
                    
                    def refresh_loras():
                        return gr.update(choices=get_lora_list())
                    refresh_btn.click(refresh_loras, outputs=[lora_list])

                generate_btn = gr.Button("Generate Image", variant="primary")

            with gr.Column(scale=1):
                img = gr.Image(label="Result", interactive=False, type="filepath")
                with gr.Tab("Active Engine Logs"):
                    log_box = gr.Textbox(label="Live C++ Terminal Output Stream", value="", lines=10, interactive=False)
                    log_timer = gr.Timer(value=3.0, active=True)
                    log_timer.tick(fn=get_live_logs, outputs=log_box)

                with gr.Tab("Generation History"):
                    refresh_history_btn = gr.Button("Refresh History Archive", variant="secondary")
                    history_gallery = gr.File(label="Generated Image Vault Files", file_count="multiple")

        preset.change(apply_preset, inputs=[preset], outputs=[width, height])

        generate_btn.click(
            fn=handle_image_generation,
            inputs=[prompt, width, height, steps, seed, cfg_scale, lora_list, lora_strength],
            outputs=img,
        ).then(fn=scan_image_history, outputs=history_gallery)

        refresh_history_btn.click(fn=scan_image_history, outputs=history_gallery)
        app.load(fn=scan_image_history, outputs=history_gallery)

    app.queue()
    return app

def launch_image():
    """Convenience function to start Z-Image-Turbo UI immediately in Kaggle."""
    app = build_image_app()
    app.launch(share=True, inline=False)
