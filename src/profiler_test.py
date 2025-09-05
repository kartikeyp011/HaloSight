import cv2
import time
from ultralytics import YOLO
import numpy as np

def main():
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ ERROR: Could not open webcam")
        return

    test_duration = 10
    start_time = time.time()
    frame_count = 0

    # Timing accumulators
    t_capture, t_inference, t_post = 0, 0, 0

    print("⚡ Running profiler for", test_duration, "seconds...")

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        t1 = time.time()

        results = model(frame, imgsz=640, verbose=False)
        t2 = time.time()

        # Draw boxes (post-processing)
        annotated = results[0].plot()
        _ = cv2.putText(annotated, "Profiling...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        t3 = time.time()

        # Update timings
        t_capture += (t1 - t0)
        t_inference += (t2 - t1)
        t_post += (t3 - t2)
        frame_count += 1

        if time.time() - start_time > test_duration:
            break

    cap.release()

    # Calculate averages
    elapsed = time.time() - start_time
    avg_fps = frame_count / elapsed
    avg_cap = (t_capture / frame_count) * 1000
    avg_inf = (t_inference / frame_count) * 1000
    avg_post = (t_post / frame_count) * 1000

    print("\n✅ Profiling complete!")
    print(f"⏱ Duration: {elapsed:.2f} sec | Frames: {frame_count}")
    print(f"⚡ Avg FPS: {avg_fps:.2f}")
    print(f"📷 Capture time: {avg_cap:.2f} ms/frame")
    print(f"🤖 Inference time: {avg_inf:.2f} ms/frame")
    print(f"🎨 Post-process time: {avg_post:.2f} ms/frame")

if __name__ == "__main__":
    main()
