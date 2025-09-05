# src/model_loader.py
import torch
from ultralytics import YOLO
import os

MODEL_PATH = "yolov8s.pt"   # you can replace with yolov8n.pt for faster demo

def load_model():
    """
    Loads YOLO model with automatic device and backend selection:
    - If GPU with TensorRT available -> use TensorRT
    - Else if GPU available -> normal CUDA
    - Else fallback to CPU
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Device detected: {device}")

    # Load model
    model = YOLO(MODEL_PATH)

    if device == "cuda":
        try:
            # Export TensorRT engine if not already done
            engine_file = MODEL_PATH.replace(".pt", "_trt.engine")
            if not os.path.exists(engine_file):
                print("⚡ Exporting YOLO model to TensorRT (this takes some time)...")
                model.export(format="engine", device=0)  # Exports TensorRT engine

            # Load TensorRT engine
            print("🚀 Running with TensorRT backend")
            model = YOLO(engine_file)
        except Exception as e:
            print(f"⚠️ TensorRT export failed, falling back to normal CUDA. Reason: {e}")
            model.to(device)
    else:
        print("💻 Running on CPU (slower).")
        model.to(device)

    return model
