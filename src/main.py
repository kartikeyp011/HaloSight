import cv2
import torch
import os
import winsound
import pandas as pd
from datetime import datetime
from collections import deque
from ultralytics import YOLO
from zone_overlay import draw_zone
from collections import deque
from alerts import play_sound, log_event, save_clip, draw_object_alert, draw_banner
from recorder import ClipRecorder

# ==============================
# CONFIG
# ==============================
MODEL_PATH = "yolov8s.pt"   # use small model for better accuracy
IMG_SIZE = 416              # slightly higher for better detection
BUFFER_SIZE = 150           # frames (~5s at 30fps)
LOG_PATH = "logs"
CLIP_PATH = "logs/critical_event.avi"

# ==============================
# SETUP
# ==============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

from model_loader import load_model
model = load_model()

# after model load
model.to(device)
os.makedirs(LOG_PATH, exist_ok=True)

# rolling video buffer
frame_buffer = deque(maxlen=100)

# ==============================
# ZONE DRAWING (interactive)
# ==============================
zone = []
drawing = False

def get_next_logfile(log_dir, base_name="detection_log"):
    os.makedirs(log_dir, exist_ok=True)
    existing = [f for f in os.listdir(log_dir) if f.startswith(base_name) and f.endswith(".csv")]
    
    if not existing:
        return os.path.join(log_dir, f"{base_name}_1.csv")
    
    nums = []
    for f in existing:
        try:
            n = int(f.replace(base_name + "_", "").replace(".csv", ""))
            nums.append(n)
        except:
            pass
    next_num = max(nums) + 1 if nums else 1
    return os.path.join(log_dir, f"{base_name}_{next_num}.csv")

def set_zone(event, x, y, flags, param):
    global zone, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        zone = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        zone.append((x, y))

# ==============================
# HELPERS
# ==============================

def is_inside_zone(box, zone_coords):
    """Check if bounding box overlaps with safety zone"""
    x1, y1, x2, y2 = map(int, box)
    zx1, zy1, zx2, zy2 = zone_coords

    overlap_x = max(0, min(x2, zx2) - max(x1, zx1))
    overlap_y = max(0, min(y2, zy2) - max(y1, zy1))
    overlap_area = overlap_x * overlap_y

    return overlap_area > 0

def get_alert_level(class_name, box, zone_coords):
    """Assign alert level based on rules"""
    # Debug print
    print(f"[DEBUG] Checking {class_name} box={box}")

    if class_name == "person":
        if is_inside_zone(box, zone_coords):
            return "Caution"

    if class_name in ["knife", "scissors", "cell phone"]:
        return "Warning"

    if class_name in ["no_helmet"]:   # example custom hazard
        return "Critical"

    return "Info"


# ==============================
# MAIN
# ==============================
def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 416)
    cap.set(4, 416)

    # --- Step 1: User draws safety zone
    cv2.namedWindow("Set Safety Zone")
    cv2.setMouseCallback("Set Safety Zone", set_zone)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        temp_frame = frame.copy()

        if len(zone) == 2:
            # just draw the rectangle directly while setting zone
            cv2.rectangle(temp_frame, zone[0], zone[1], (0, 255, 0), 2)

        cv2.imshow("Set Safety Zone", temp_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # ENTER = confirm
            break
        elif key == 27:  # ESC = exit
            cap.release()
            cv2.destroyAllWindows()
            return

    cv2.destroyWindow("Set Safety Zone")

    # Now we finalize zone_coords AFTER confirmation
    zx1, zy1 = min(zone[0][0], zone[1][0]), min(zone[0][1], zone[1][1])
    zx2, zy2 = max(zone[0][0], zone[1][0]), max(zone[0][1], zone[1][1])
    zone_coords = (zx1, zy1, zx2, zy2)
    print(f"[INFO] Safety zone set: {zone_coords}")

    detections_log = []

    # --- Step 2: Start detection loop
    recorder = ClipRecorder(buffer_size=50)  # ~5s buffer
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # update recorder
        recorder.update(frame)

        # Run detection, check for critical event
        # if alert_level == "Critical":
        #     recorder.save_clip(fps=10)

        results = model(frame, imgsz=IMG_SIZE, verbose=False)
        frame_detections = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls.item())
                class_name = model.names[cls]
                conf = float(box.conf.item())
                xyxy = box.xyxy.cpu().numpy().tolist()[0]

                alert_level = get_alert_level(class_name, xyxy, zone_coords)

                # 1. Draw object bounding box + label
                draw_object_alert(frame, xyxy, class_name, alert_level.lower(), conf)

                # 2. Log event
                detection = {"alert": alert_level, "class": class_name, "conf": conf}
                log_event(alert_level.lower(), class_name, conf)
                detections_log.append(detection)
                frame_detections.append(detection)

                # 3. Play sound + save clip if warning/critical
                play_sound(alert_level.lower())
                if alert_level.lower() in ["warning", "critical"]:
                    # save last ~5s from recorder buffer
                    recorder.save_clip(fps=10)
                    save_clip(list(frame_buffer), alert_level.lower())

        # --- Determine zone alert level for this frame ---
        zone_alert = "safe"
        for d in frame_detections:
            if d["alert"] == "Critical":
                zone_alert = "critical"
                break
            elif d["alert"] == "Warning":
                zone_alert = "warning"
            elif d["alert"] == "Caution" and zone_alert not in ["warning", "critical"]:
                zone_alert = "caution"
            elif d["alert"] == "Info" and zone_alert == "safe":
                zone_alert = "info"

        # --- Draw glowing safety zone overlay + top banner ---
        frame = draw_zone(frame, zone_coords, alert_level=zone_alert)
        frame = draw_banner(frame, zone_alert)


        cv2.imshow("HaloSight MVP", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save CSV log
    # Save CSV log
    csv_path = get_next_logfile(LOG_PATH, "detection_log")
    df = pd.DataFrame(detections_log)
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Log saved: {csv_path}")
# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    main()
