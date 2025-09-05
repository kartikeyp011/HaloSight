import cv2
import numpy as np
import torch
from ultralytics import YOLO
import winsound  # Windows only

# -------------------- SETTINGS --------------------
# Safety zone coordinates (example)
DANGER_ZONE = {"x1": 100, "y1": 100, "x2": 400, "y2": 400}

# Alert colors
ALERT_COLORS = {
    "Info": (255, 0, 0),
    "Caution": (0, 255, 255),
    "Warning": (0, 165, 255),
    "Critical": (0, 0, 255)
}

# Classes mapping (simulate mission-critical items)
PPE_CLASSES = ["helmet", "gloves"]
TOOL_CLASSES = ["wrench", "tool"]
HAZARD_CLASSES = ["hazard_sign"]

# Confidence threshold
CONF_THRESHOLD = 0.25

# -------------------- FUNCTIONS --------------------
def in_danger_zone(x1, y1, x2, y2, zone):
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return zone['x1'] <= cx <= zone['x2'] and zone['y1'] <= cy <= zone['y2']

def get_alert_level(class_name, box):
    x1, y1, x2, y2 = box
    if class_name in PPE_CLASSES:
        return "Info"
    elif class_name in TOOL_CLASSES and in_danger_zone(x1, y1, x2, y2, DANGER_ZONE):
        return "Caution"
    elif class_name in HAZARD_CLASSES:
        return "Warning"
    # Simulate missing PPE for demo (random or manual trigger)
    # Example: if class_name == "person" and PPE missing:
    #     return "Critical"
    return "Info"

def draw_alert(frame, box, class_name, alert_level, conf):
    x1, y1, x2, y2 = map(int, box)
    color = ALERT_COLORS.get(alert_level, (255, 255, 255))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"{class_name} {conf:.2f} | {alert_level}"
    cv2.putText(frame, label, (x1, max(y1-8, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def play_alert(alert_level):
    if alert_level == "Critical":
        winsound.Beep(1000, 200)

# -------------------- MAIN LOOP --------------------
def main():
    # Load YOLO model (nano for speed)
    model = YOLO("yolov8n.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 416)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 416)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Try index 1/2 or close apps using webcam.")

    print("Press ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, imgsz=320, conf=CONF_THRESHOLD, verbose=False)

        # Draw safety zone
        cv2.rectangle(frame, (DANGER_ZONE['x1'], DANGER_ZONE['y1']),
                      (DANGER_ZONE['x2'], DANGER_ZONE['y2']), (0,0,255), 2)
        cv2.putText(frame, "Danger Zone", (DANGER_ZONE['x1'], DANGER_ZONE['y1']-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # Process detections
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy()

            for box, conf, cls_id in zip(boxes, confs, cls_ids):
                class_name = model.names[int(cls_id)]
                alert_level = get_alert_level(class_name, box)
                draw_alert(frame, box, class_name, alert_level, conf)
                if alert_level == "Critical":
                    play_alert(alert_level)

        # Show FPS and device info
        cv2.putText(frame, f"Device: {device}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow("HaloSight MVP — Step 2", frame)

        if cv2.waitKey(1) == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
