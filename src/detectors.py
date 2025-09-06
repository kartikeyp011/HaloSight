# src/detectors.py
import torch
from ultralytics import YOLO

class MultiDetector:
    def __init__(self):
        print("🔹 Initializing Detector for MVP...")

        # General-purpose detection (COCO classes: person, tools, etc.)
        self.general_model = YOLO("yolov8n.pt")  # small, fast model

        # Decide device (GPU if available)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.general_model.to(self.device)

        print(f"✅ General model loaded on {self.device}")

    def detect(self, frame):
        """Run the general YOLO model and return detections"""
        detections = []

        results = self.general_model.predict(frame, imgsz=640, verbose=False)
        for r in results[0].boxes:
            cls = self.general_model.names[int(r.cls)]
            conf = float(r.conf)
            xyxy = r.xyxy.cpu().numpy().tolist()[0]  # [x1, y1, x2, y2]
            detections.append({
                "type": cls,
                "conf": conf,
                "bbox": xyxy
            })

        return detections
