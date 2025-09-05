# run.py
import os
import subprocess

def ensure_dirs():
    folders = ["logs", "event_clips", "config"]
    for f in folders:
        if not os.path.exists(f):
            os.makedirs(f)
            print(f"📂 Created folder: {f}")

def main():
    print("🚀 Launching HaloSight...")
    ensure_dirs()

    # Start main detection pipeline
    detector = subprocess.Popen(["python", "src/main.py"])

    # Start Streamlit dashboard
    dashboard = subprocess.Popen(
        ["streamlit", "run", "src/review_app.py", "--server.headless", "true"]
    )

    detector.wait()
    dashboard.terminate()

if __name__ == "__main__":
    main()
