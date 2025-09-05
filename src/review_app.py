# Streamlit dashboard for reviewing logs & clips.
import streamlit as st
import pandas as pd
import os

LOG_DIR = "logs"
CLIP_DIR = os.path.join(LOG_DIR, "clips")

st.set_page_config(page_title="HaloSight Review Dashboard", layout="wide")

st.title("🚀 HaloSight Event Review Dashboard")

# === Load Logs ===
logs = []
for file in sorted(os.listdir(LOG_DIR)):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(LOG_DIR, file))
        df["log_file"] = file
        logs.append(df)

if logs:
    all_logs = pd.concat(logs, ignore_index=True)
    st.subheader("📑 Detection Logs")
    level_filter = st.selectbox("Filter by Alert Level", ["All", "Info", "Caution", "Warning", "Critical"])
    if level_filter != "All":
        filtered = all_logs[all_logs["alert_level"].str.lower() == level_filter.lower()]
    else:
        filtered = all_logs

    st.dataframe(filtered, use_container_width=True)
else:
    st.warning("No logs found yet!")

# === Review Clips ===
st.subheader("🎥 Critical Event Clips")

if os.path.exists(CLIP_DIR):
    clips = sorted(os.listdir(CLIP_DIR))
    for clip in clips:
        if clip.endswith(".mp4"):
            st.write(f"**{clip}**")
            st.video(os.path.join(CLIP_DIR, clip))
else:
    st.info("No clips recorded yet.")

