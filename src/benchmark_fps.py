# src/benchmark_fps.py
import time
import cv2
from model_loader import load_model

def benchmark(model, source=0, num_frames=100):
    """
    Run inference on webcam frames to measure average FPS.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("❌ Could not open video source")
        return

    print(f"⚡ Running benchmark on {num_frames} frames...")

    frame_count = 0
    start_time = time.time()

    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        _ = model.predict(frame, imgsz=640, verbose=False)

        frame_count += 1

    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0

    cap.release()
    return fps

if __name__ == "__main__":
    print("🔹 Loading model...")
    model = load_model()

    # Run benchmark
    avg_fps = benchmark(model, source=0, num_frames=100)
    print(f"✅ Average FPS: {avg_fps:.2f}")
