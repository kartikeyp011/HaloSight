import cv2
import time
import csv
from ultralytics import YOLO
import os

def main():
    # Load YOLOv8 small model
    model = YOLO("yolov8n.pt")

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ ERROR: Could not open webcam")
        return

    # Baseline test duration (seconds)
    test_duration = 10
    start_time = time.time()
    frame_count = 0
    detections = []

    print("🔍 Running baseline test for", test_duration, "seconds...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO prediction
        results = model(frame, imgsz=640, verbose=False)

        # Count frame
        frame_count += 1

        # Collect detection info
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                detections.append([frame_count, cls, conf, xyxy])

        # Stop after duration
        if time.time() - start_time > test_duration:
            break

    cap.release()

    # Calculate FPS
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time

    # Save detections to CSV
    os.makedirs("logs", exist_ok=True)
    log_path = "logs/baseline_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "class_id", "confidence", "bbox(xyxy)"])
        writer.writerows(detections)

    # Print summary
    print("\n✅ Baseline test complete!")
    print(f"⏱  Duration: {elapsed_time:.2f} sec")
    print(f"📸  Frames processed: {frame_count}")
    print(f"⚡  Avg FPS: {fps:.2f}")
    print(f"📝  Detections saved in {log_path}")

if __name__ == "__main__":
    main()
