import os
import sys
import signal
import time

# Ensure import paths resolve correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.downloader import restore_binary, download_models
from src.server import start_server
from src.ui import build_app

def main():
    print("🚀 Starting AI Studio on Lightning.ai...")
    
    # Setup paths for Lightning.ai persistent workspace storage
    models_base = "/teamspace/studios/this_studio/models"
    bin_dir = "/teamspace/studios/this_studio/sd_bin"
    bin_path = os.path.join(bin_dir, "bin/sd-server")
    log_path = "/teamspace/studios/this_studio/server.log"
    
    # 1. Restore the C++ compilation binary if missing
    if not os.path.exists(bin_path):
        restore_binary(repo="airesearch-official/free-aistudio", tag="v1.0.0", target_dir=bin_dir)
        
    # 2. Download LTX-Video FP8 weights (downloader skips already completed downloads)
    download_models(preset="LTX-Video-2.3-FP8", models_base=models_base)
    
    # 3. Start the background stable-diffusion.cpp inference server
    server_process = start_server(
        preset="LTX-Video-2.3-FP8",
        bin_path=bin_path,
        models_base=models_base,
        log_path=log_path,
        load_audio_vae=True
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
    app.launch(share=True, inline=False)
    
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
