# src/multi_cam.py
"""
Multi-camera HaloSight demo
- Run: python src/multi_cam.py 0 1            # two local webcams
- Or:   python src/multi_cam.py 0 rtsp://...
"""
import sys
import os
import cv2
import threading
import time
from collections import deque
from datetime import datetime
import pandas as pd

# Try to import model_loader if you created it earlier. If not, fall back to directly loading YOLO.
try:
    from model_loader import load_model
    def _load_model():
        return load_model()
except Exception:
    from ultralytics import YOLO
    import torch
    def _load_model():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Loading YOLO on device: {device}")
        m = YOLO("yolov8s.pt")  # change model path if needed
        try:
            m.to(device)
        except Exception:
            pass
        return m

# --------------- Configuration ---------------
IMG_SIZE = 416
CONF_THRESH = 0.25
BUFFER_SIZE = 150                 # frames saved in rolling buffer (~5s @ 30fps)
LOG_DIR = "logs"
CLIPS_DIR = os.path.join(LOG_DIR, "clips")
os.makedirs(CLIPS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --------------- Utilities ---------------
def get_next_filename(dirpath, base, ext):
    os.makedirs(dirpath, exist_ok=True)
    existing = [f for f in os.listdir(dirpath) if f.startswith(base) and f.endswith(ext)]
    if not existing:
        return os.path.join(dirpath, f"{base}_1{ext}")
    nums = []
    for f in existing:
        try:
            n = int(f.replace(base + "_", "").replace(ext, ""))
            nums.append(n)
        except:
            pass
    next_num = max(nums) + 1 if nums else 1
    return os.path.join(dirpath, f"{base}_{next_num}{ext}")

# --------------- Camera Worker ---------------
class CameraWorker(threading.Thread):
    def __init__(self, cam_id, source, model, model_lock, zone_coords=None):
        super().__init__(daemon=True)
        self.cam_id = cam_id
        self.source = source
        self.model = model
        self.model_lock = model_lock
        self.zone = zone_coords  # (x1,y1,x2,y2) or None
        self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW if os.name == "nt" else 0)
        # Try to set resolution if local camera integer index
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_SIZE)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_SIZE)
        except:
            pass
        self.running = True
        self.frame = None
        self.display_frame = None
        self.frame_buffer = deque(maxlen=BUFFER_SIZE)
        self.log_rows = []
        self.last_clip_saved_at = 0

    def run(self):
        print(f"[INFO] CameraWorker {self.cam_id} started for source={self.source}")
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                # if stream dropped, wait and retry
                time.sleep(0.1)
                continue
            self.frame = frame.copy()
            self.frame_buffer.append(frame.copy())

            # Inference (safely share model using lock)
            try:
                with self.model_lock:
                    results = self.model.predict(frame, imgsz=IMG_SIZE, conf=CONF_THRESH, verbose=False)
            except Exception as e:
                # fallback: try calling model(frame) (older API)
                try:
                    with self.model_lock:
                        results = self.model(frame, imgsz=IMG_SIZE, verbose=False)
                except Exception as e2:
                    print(f"[WARN] Cam {self.cam_id}: inference failed: {e} / {e2}")
                    results = []

            # Draw results on a copy for display
            disp = frame.copy()
            highest_level = "Safe"
            for r in results:
                boxes = getattr(r, "boxes", None)
                if boxes is None:
                    continue
                # boxes.xyxy, boxes.conf, boxes.cls
                xyxy_arr = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else []
                confs = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else []
                clsids = boxes.cls.cpu().numpy() if hasattr(boxes, "cls") else []

                for box, conf, cid in zip(xyxy_arr, confs, clsids):
                    x1, y1, x2, y2 = map(int, box)
                    clsid = int(cid)
                    # Safe guard: get class name if available
                    try:
                        clsname = self.model.names[clsid]
                    except Exception:
                        clsname = str(clsid)
                    # Determine alert level (simple rules)
                    alert_level = "Info"
                    if clsname == "person" and self.zone and rect_overlap((x1,y1,x2,y2), self.zone):
                        alert_level = "Caution"
                    if clsname in ["knife", "scissors", "cell phone"]:
                        alert_level = "Warning"
                    if clsname in ["no_helmet"]:
                        alert_level = "Critical"

                    # Update highest_level for zone glow
                    highest_level = max_alert_level(highest_level, alert_level)

                    # draw bbox
                    color = alert_color(alert_level)
                    cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)
                    label = f"{clsname} {float(conf):.2f} {alert_level}"
                    cv2.putText(disp, label, (x1, max(12, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # log row
                    row = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                        "camera": self.cam_id,
                        "object": clsname,
                        "alert": alert_level,
                        "confidence": float(conf),
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2
                    }
                    self.log_rows.append(row)

                    # handle critical: save clip (limit to one per 5s)
                    if alert_level == "Critical":
                        now = time.time()
                        if now - self.last_clip_saved_at > 5:
                            clip_path = get_next_filename(CLIPS_DIR, f"cam{self.cam_id}_clip", ".avi")
                            save_clip(list(self.frame_buffer), clip_path)
                            print(f"[INFO] Cam{self.cam_id}: saved critical clip -> {clip_path}")
                            self.last_clip_saved_at = now

            # draw zone and glow depending on highest_level
            if self.zone:
                zx1, zy1, zx2, zy2 = self.zone
                glow_color = zone_glow_color(highest_level)
                # Draw filled transparent rectangle for glow
                overlay = disp.copy()
                alpha = 0.25
                cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), glow_color, -1)
                cv2.addWeighted(overlay, alpha, disp, 1 - alpha, 0, disp)
                # border
                cv2.rectangle(disp, (zx1, zy1), (zx2, zy2), glow_color, 2)
                cv2.putText(disp, f"Zone: {highest_level}", (zx1, max(12, zy1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, glow_color, 2)

            # annotate camera id
            cv2.putText(disp, f"Cam {self.cam_id}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            self.display_frame = disp

        # cleanup
        try:
            self.cap.release()
        except:
            pass
        print(f"[INFO] CameraWorker {self.cam_id} stopped")

    def stop(self):
        self.running = False

# --------------- Helper functions ---------------
def rect_overlap(boxA, boxB):
    # box = (x1,y1,x2,y2)
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    overlap_x = max(0, min(ax2, bx2) - max(ax1, bx1))
    overlap_y = max(0, min(ay2, by2) - max(ay1, by1))
    return (overlap_x * overlap_y) > 0

def alert_color(level):
    return {
        "Info": (255, 255, 0),
        "Caution": (0, 255, 255),
        "Warning": (0, 165, 255),
        "Critical": (0, 0, 255)
    }.get(level, (255,255,255))

def zone_glow_color(level):
    return {
        "Safe": (0, 255, 0),
        "Info": (200, 200, 200),
        "Caution": (0, 255, 255),
        "Warning": (0, 165, 255),
        "Critical": (0, 0, 255)
    }.get(level, (0,255,0))

def max_alert_level(a, b):
    """Return the most severe alert between two strings."""
    order = {"Safe":0, "Info":1, "Caution":2, "Warning":3, "Critical":4}
    return a if order.get(a,0) >= order.get(b,0) else b

def save_clip(frames, filename):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, 20, (w, h))
    for f in frames:
        out.write(f)
    out.release()

# --------------- Zone drawing utility ---------------
def draw_zone_interactive(source, winname="Set Zone"):
    """
    Let user draw a rectangle on the source stream window.
    Returns (x1,y1,x2,y2) or None if cancelled.
    """
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW if os.name == "nt" else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_SIZE)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_SIZE)

    zone = []
    drawing = {"on": False}

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing["on"] = True
            zone.clear()
            zone.append((x,y))
        elif event == cv2.EVENT_LBUTTONUP:
            drawing["on"] = False
            zone.append((x,y))

    cv2.namedWindow(winname)
    cv2.setMouseCallback(winname, mouse_cb)
    print(f"[INFO] Draw safety zone for source={source}. Drag with mouse, press ENTER to confirm, ESC to skip.")
    selected = None
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        temp = frame.copy()
        if len(zone) == 1 and drawing["on"]:
            cv2.circle(temp, zone[0], 4, (0,0,255), -1)
        if len(zone) == 2:
            cv2.rectangle(temp, zone[0], zone[1], (0,0,255), 2)
        cv2.imshow(winname, temp)
        k = cv2.waitKey(1) & 0xFF
        if k == 13 and len(zone) == 2:  # ENTER
            x1,y1 = zone[0]; x2,y2 = zone[1]
            selected = (min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2))
            break
        elif k == 27:  # ESC
            break
    cap.release()
    cv2.destroyWindow(winname)
    return selected

# --------------- Display utils ---------------
def make_grid(frames, cols=None):
    """Arrange frames into a grid for display."""
    if not frames:
        return None
    n = len(frames)
    cols = cols or int(n**0.5 + 0.999)
    rows = (n + cols - 1) // cols
    h = max(f.shape[0] for f in frames)
    w = max(f.shape[1] for f in frames)
    # pad frames to same size
    padded = []
    for f in frames:
        fh, fw = f.shape[:2]
        # create black canvas
        canvas = 255 * np.ones((h, w, 3), dtype=f.dtype)
        canvas[0:fh, 0:fw] = f
        padded.append(canvas)
    # stack
    rows_imgs = []
    for r in range(rows):
        row_imgs = padded[r*cols:(r+1)*cols]
        # if last row has fewer, pad with white
        while len(row_imgs) < cols:
            row_imgs.append(255 * np.ones((h,w,3), dtype=frames[0].dtype))
        row = cv2.hconcat(row_imgs)
        rows_imgs.append(row)
    full = cv2.vconcat(rows_imgs)
    return full

# import numpy here (used by make_grid)
import numpy as np

# --------------- Main function ---------------
def main():
    # parse arguments as sources
    args = sys.argv[1:]
    if not args:
        print("Usage: python src/multi_cam.py <source1> <source2> ...")
        print("Examples: python src/multi_cam.py 0 1")
        print("          python src/multi_cam.py 0 rtsp://192.168.1.2:554/stream")
        return

    # convert numeric strings to ints (device indices)
    sources = []
    for a in args:
        try:
            sources.append(int(a))
        except:
            sources.append(a)

    print(f"[INFO] Sources: {sources}")

    # load model once and share with workers (protected by a lock)
    model = _load_model()
    model_lock = threading.Lock()

    # Step A: for each source let user draw a zone (sequentially)
    zones = []
    for i, s in enumerate(sources):
        zone = draw_zone_interactive(s, winname=f"Set Zone - Cam{i}")
        zones.append(zone)
        print(f"[INFO] Cam{i} zone: {zone}")

    # Step B: start camera workers
    workers = []
    for i, s in enumerate(sources):
        worker = CameraWorker(i, s, model, model_lock, zone_coords=zones[i])
        worker.start()
        workers.append(worker)

    # Step C: main display loop - show all camera frames in a grid
    try:
        while True:
            frames = []
            for w in workers:
                if w.display_frame is None:
                    # placeholder blank image
                    frames.append(255 * np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))
                else:
                    frames.append(w.display_frame)
            grid = make_grid(frames)
            if grid is not None:
                cv2.imshow("HaloSight - MultiCam", grid)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
    except KeyboardInterrupt:
        pass
    finally:
        print("[INFO] Stopping workers...")
        for w in workers:
            w.stop()
        time.sleep(0.5)
        cv2.destroyAllWindows()

        # save per-camera logs (auto-numbered)
        for w in workers:
            if not w.log_rows:
                continue
            df = pd.DataFrame(w.log_rows)
            csv_path = get_next_filename(LOG_DIR, f"cam{w.cam_id}_detection_log", ".csv")
            df.to_csv(csv_path, index=False)
            print(f"[INFO] Cam{w.cam_id} log saved: {csv_path}")

if __name__ == "__main__":
    main()
