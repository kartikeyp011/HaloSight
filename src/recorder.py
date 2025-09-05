# rolling buffer recording and saving clips when a critical event occurs.
import cv2
import os
from collections import deque
from datetime import datetime

class ClipRecorder:
    def __init__(self, buffer_size=150, output_dir="logs/clips"):
        """
        buffer_size: number of frames to keep in memory (5s if running ~30 FPS, 10s if ~15 FPS)
        """
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def update(self, frame):
        """Save frame to rolling buffer"""
        self.buffer.append(frame.copy())

    def save_clip(self, fps=15):
        """Save current buffer to a timestamped video clip"""
        if not self.buffer:
            return None

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"clip_{ts}.mp4")

        h, w, _ = self.buffer[0].shape
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        for f in self.buffer:
            out.write(f)
        out.release()

        print(f"[Recorder] Saved critical event clip: {filename}")
        return filename
