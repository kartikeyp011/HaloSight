# review_app.py
# Streamlit dashboard for reviewing HaloSight logs & clips.

import streamlit as st
import pandas as pd
import os
import os
import subprocess

# ==============================
# Ensure Folder Structure
# ==============================
def ensure_dirs():
    folders = ["logs", "event_clips", "config"]
    for f in folders:
        if not os.path.exists(f):
            os.makedirs(f)
            print(f"📂 Created folder: {f}")
# ==============================
# Directories
# ==============================
LOG_DIR = "logs"
CLIP_DIR = "event_clips"   # match your alerts.py

st.set_page_config(page_title="HaloSight Review Dashboard", layout="wide")
st.title("🚀 HaloSight Event Review Dashboard")

# ==============================
# Load and Separate Logs
# ==============================
alert_logs = []
baseline_logs = []

if os.path.exists(LOG_DIR):
    for file in sorted(os.listdir(LOG_DIR)):
        if file.endswith(".csv"):
            path = os.path.join(LOG_DIR, file)
            try:
                df = pd.read_csv(path)

                # --- Processed Alert Logs (with alert_level) ---
                if "alert_level" in df.columns:
                    df["log_file"] = file
                    alert_logs.append(df)

                # --- Raw YOLO Baseline Logs (frame/class_id format) ---
                elif "frame" in df.columns and "class_id" in df.columns:
                    df["log_file"] = file
                    baseline_logs.append(df)

            except Exception as e:
                st.error(f"❌ Error reading {file}: {e}")

# ==============================
# Display Processed Alert Logs
# ==============================
if alert_logs:
    st.subheader("📑 Processed Alert Logs")
    all_alerts = pd.concat(alert_logs, ignore_index=True)

    # Filter dropdown
    level_filter = st.selectbox(
        "Filter by Alert Level",
        ["All", "Info", "Caution", "Warning", "Critical"]
    )
    if level_filter != "All":
        filtered = all_alerts[all_alerts["alert_level"].str.lower() == level_filter.lower()]
    else:
        filtered = all_alerts

    st.dataframe(filtered, use_container_width=True)
else:
    st.info("No processed alert logs found yet.")

# ==============================
# Display Raw Baseline Logs
# ==============================
if baseline_logs:
    st.subheader("📊 Raw Detection Logs (Baseline)")
    all_baseline = pd.concat(baseline_logs, ignore_index=True)
    st.dataframe(all_baseline, use_container_width=True)

# ==============================
# Review Critical Clips
# ==============================
st.subheader("🎥 Critical Event Clips")

if os.path.exists(CLIP_DIR):
    clips = sorted(os.listdir(CLIP_DIR))
    if clips:
        for clip in clips:
            if clip.endswith(".avi") or clip.endswith(".mp4"):
                st.write(f"**{clip}**")
                st.video(os.path.join(CLIP_DIR, clip))
    else:
        st.info("No clips recorded yet.")
else:
    st.info("No clips directory found yet.")

# ==============================
# Ensure Folder Structure
# ==============================
def ensure_dirs():
    folders = ["logs", "event_clips", "config"]
    for f in folders:
        if not os.path.exists(f):
            os.makedirs(f)
            print(f"📂 Created folder: {f}")

# ==============================
# Main Launcher
# ==============================
def main():
    print("🚀 Launching HaloSight...")
    ensure_dirs()
    subprocess.run(["python", "src/main.py"])

if __name__ == "__main__":
    main()