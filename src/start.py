# run.py
import os
import subprocess

def ensure_dirs():
    folders = ["logs", "logs/clips", "config"]
    for f in folders:
        if not os.path.exists(f):
            os.makedirs(f)
            print(f"Created folder: {f}")

def main():
    print("🚀 Starting HaloSight...")
    ensure_dirs()
    subprocess.run(["python", "src/main.py"])

if __name__ == "__main__":
    main()
