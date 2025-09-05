import os

def get_next_filename(folder: str, base: str, ext: str) -> str:
    """
    Returns an auto-incremented filename like logs/base_1.ext, base_2.ext, etc.
    """
    os.makedirs(folder, exist_ok=True)
    i = 1
    while True:
        path = os.path.join(folder, f"{base}_{i}.{ext}")
        if not os.path.exists(path):
            return path
        i += 1
