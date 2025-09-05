import os
import cv2
import winsound
import pandas as pd
from datetime import datetime

# ==============================
# Directories
# ==============================
LOG_DIR = "logs"
CLIP_DIR = "event_clips"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CLIP_DIR, exist_ok=True)


# ==============================
# LOG FILE HANDLING
# ==============================
def get_new_logfile():
    """Auto-increment log file name"""
    existing = [f for f in os.listdir(LOG_DIR) if f.startswith("detection_log")]
    if not existing:
        return os.path.join(LOG_DIR, "detection_log_1.csv")
    nums = [int(f.split("_")[-1].split(".")[0]) for f in existing]
    new_num = max(nums) + 1
    return os.path.join(LOG_DIR, f"detection_log_{new_num}.csv")


LOG_FILE = get_new_logfile()


# ==============================
# ALERT SYSTEM
# ==============================
def play_sound(level: str):
    """Play different tones for different alert levels."""
    if level == "caution":
        winsound.Beep(1000, 300)  # short beep
    elif level == "warning":
        winsound.Beep(1200, 500)  # longer beep
    elif level == "critical":
        winsound.Beep(1500, 800)  # siren-like
        winsound.Beep(1000, 800)


# --- Object-level bounding box alerts ---
def draw_object_alert(frame, box, obj_class: str, level: str, conf: float):
    """Draw bounding box + label on detected object"""
    color_map = {
        "info": (0, 255, 0),      # green
        "caution": (0, 255, 255), # yellow
        "warning": (0, 165, 255), # orange
        "critical": (0, 0, 255)   # red
    }
    color = color_map.get(level, (255, 255, 255))
    x1, y1, x2, y2 = map(int, box)

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Label
    label = f"{obj_class} ({level}) {conf:.2f}"
    cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


# --- Zone-level banner alerts ---
def draw_banner(frame, level: str):
    """Overlay top banner with overall alert status"""
    color_map = {
        "safe": (0, 200, 0),
        "info": (0, 255, 0),
        "caution": (0, 255, 255),
        "warning": (0, 165, 255),
        "critical": (0, 0, 255)
    }
    color = color_map.get(level, (255, 255, 255))

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), color, -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.putText(frame, f"ZONE STATUS: {level.upper()}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2)
    return frame


# ==============================
# LOGGING
# ==============================
def log_event(alert_level: str, obj_class: str, confidence: float):
    """Append event to CSV log with consistent alert_level column"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {"time": timestamp, "alert_level": alert_level,
           "object": obj_class, "confidence": confidence}
    df = pd.DataFrame([row])

    if not os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, index=False)
    else:
        df.to_csv(LOG_FILE, mode="a", header=False, index=False)


# ==============================
# VIDEO CLIP SAVING
# ==============================
def save_clip(frames, level: str):
    """Save last 5s of frames if warning/critical alert"""
    if level not in ["warning", "critical"] or not frames:
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(CLIP_DIR, f"{level}_event_{ts}.avi")

    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"XVID"), 10, (w, h))
    for f in frames:
        out.write(f)
    out.release()
