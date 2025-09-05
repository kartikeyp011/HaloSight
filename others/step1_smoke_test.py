import time
import cv2
from ultralytics import YOLO
import torch

def draw_boxes(frame, result, class_names):
    # result.boxes: xyxy, conf, cls
    boxes = result.boxes
    if boxes is None:
        return frame
    xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else []
    conf = boxes.conf.cpu().numpy() if boxes.conf is not None else []
    cls  = boxes.cls.cpu().numpy()  if boxes.cls is not None else []

    for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
        label = f"{class_names.get(int(k), str(int(k)))} {float(c):.2f}"
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(y1-8, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return frame

def main():
    # Smallest YOLOv8 model for speed
    model = YOLO("yolov8n.pt")  # auto-downloads on first run

    # Device choice (auto)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Try index 1/2 or close apps using the webcam.")

    # Optional: set resolution for stability
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    names = model.names if hasattr(model, "names") else {}
    print("Press ESC to quit.")

    # Warmup (helps with consistent FPS)
    # _ = model.predict(source=cv2.imread.__doc__ is None, imgsz=640, verbose=False)

    prev_t = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.time()
        # Inference on current frame
        results = model.predict(frame, imgsz=640, conf=0.25, verbose=False)
        dt = time.time() - t0
        fps = 1.0 / dt if dt > 0 else 0.0

        # Draw on frame
        vis = frame.copy()
        for r in results:
            vis = draw_boxes(vis, r, names)

        # Overlay FPS
        cv2.putText(vis, f"FPS: {fps:.1f} | Device: {device}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.imshow("HaloSight — Step 1 (Baseline YOLO)", vis)
        if cv2.waitKey(1) == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
