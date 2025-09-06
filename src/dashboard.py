# src/dashboard.py
import os
import streamlit as st
import cv2
from collections import deque
import pandas as pd
from fpdf import FPDF
from detectors import MultiDetector
from alerts import draw_object_alert, draw_banner
from zone_overlay import draw_zone
from PIL import Image
import base64
import io
zone_coords = (50, 50, 366, 366)  # default safety zone



def image_to_url(image: Image.Image) -> str:
    """Converts a PIL Image to a data URL."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# Now import st_canvas
from streamlit_drawable_canvas import st_canvas

# --------------------------
# Streamlit Setup
# --------------------------
st.set_page_config(page_title="HaloSight Dashboard", layout="wide")
st.title("HaloSight Multi-Camera Operator Dashboard")

# --------------------------
# Camera Setup
# --------------------------
camera_sources = [0, 1]  # replace with your camera IDs or RTSP URLs
caps = []
for src in camera_sources:
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        st.warning(f"Camera {src} cannot be opened")
    else:
        caps.append(cap)

if not caps:
    st.error("No cameras available. Exiting dashboard.")
    st.stop()

frame_buffers = [deque(maxlen=100) for _ in caps]

# --------------------------
# Multi-Model Detector
# --------------------------
detector = MultiDetector()
events_log = []

# --------------------------
# Interactive Safety Zone Setup
# --------------------------
# st.sidebar.header("Safety Zone Setup")
# zone_coords = (50, 50, 366, 366)  # default

# # Show first camera as base for drawing
# ret, base_frame = caps[0].read()
# if ret and base_frame is not None:
#     base_frame_rgb = cv2.cvtColor(base_frame, cv2.COLOR_BGR2RGB)
#     base_pil = Image.fromarray(base_frame_rgb)
#     canvas_result = st_canvas(
#         fill_color="rgba(0,0,0,0)",
#         stroke_width=3,
#         stroke_color="green",
#         background_image=base_pil,
#         height=base_frame_rgb.shape[0],
#         width=base_frame_rgb.shape[1],
#         drawing_mode="rect",
#         key="safety_zone_canvas",
#     )
#     if canvas_result.json_data and canvas_result.json_data.get("objects"):
#         obj = canvas_result.json_data["objects"][-1]
#         x1, y1 = int(obj["left"]), int(obj["top"])
#         x2, y2 = int(obj["left"] + obj["width"]), int(obj["top"] + obj["height"])
#         zone_coords = (x1, y1, x2, y2)
#         st.sidebar.write(f"Safety Zone: {zone_coords}")

# --------------------------
# Streamlit placeholders for live video
# --------------------------
cols = st.columns(len(caps))
video_slots = [col.empty() for col in cols]

st.write("🔹 Starting live detection...")

# Use a session state to control stopping
if "stop_dashboard" not in st.session_state:
    st.session_state.stop_dashboard = False

if st.button("Stop Dashboard"):
    st.session_state.stop_dashboard = True

# --------------------------
# Main Detection + Display Loop
# --------------------------
for i, cap in enumerate(caps):
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    frame_buffers[i].append(frame.copy())
    detections = detector.detect(frame)

    alert = "Info"  # default
    for d in detections:
        class_name = d["type"]
        conf = d["conf"]
        bbox = d["bbox"]

        if class_name.startswith("hazard_") or "helmet_missing" in class_name:
            alert = "Critical"
        elif class_name.startswith("ppe_") or class_name in ["knife", "scissors"]:
            alert = "Warning"

        draw_object_alert(frame, bbox, class_name, alert.lower(), conf)

        events_log.append({
            "Timestamp": pd.Timestamp.now(),
            "Camera": i,
            "Class": class_name,
            "Alert": alert,
            "Confidence": conf
        })

    frame = draw_zone(frame, zone_coords, alert_level="safe")
    frame = draw_banner(frame, "safe")
    video_slots[i].image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# --------------------------
# Export PDF Report Button
# --------------------------
if st.button("Export Incident Report"):
    if not events_log:
        st.warning("No events yet!")
    else:
        df = pd.DataFrame(events_log)
        os.makedirs("logs", exist_ok=True)
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "HaloSight Incident Report", ln=True, align="C")
        pdf.set_font("Arial", "", 12)
        pdf.ln(10)

        for idx, row in df.iterrows():
            line = f"{row['Timestamp']} | Camera {row['Camera']} | {row['Class']} | {row['Alert']} | {row['Confidence']:.2f}"
            pdf.cell(0, 8, line, ln=True)

        pdf_file = "logs/incident_report.pdf"
        pdf.output(pdf_file)
        st.success(f"PDF saved to {pdf_file}")

# --------------------------
# Clean up
# --------------------------
for cap in caps:
    cap.release()
cv2.destroyAllWindows()
