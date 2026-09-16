import os, sys, json, subprocess, tarfile, io, requests
from PIL import Image
import numpy as np

WLD_DIR = "E:\\Saad_audi deepfakes\\data\\WLD"
WD_DIR = "E:\\Saad_audi deepfakes\\data\\WildDeepfake"
os.makedirs(WLD_DIR, exist_ok=True)
os.makedirs(WD_DIR, exist_ok=True)

# === 1. Download WildDeepfake subset ===
print("=== Downloading WildDeepfake subset ===")
# Download first fake_train tar.gz and first real_train tar.gz
base_url = "https://huggingface.co/datasets/xingjunm/WildDeepfake/resolve/main/deepfake_in_the_wild"
files_to_get = [
    "fake_train/1.tar.gz",
    "real_train/1.tar.gz",
    "fake_test/1.tar.gz",
    "real_test/1.tar.gz",
]

for fname in files_to_get:
    url = f"{base_url}/{fname}"
    local_path = os.path.join(WD_DIR, fname.replace("/", "_"))
    if os.path.exists(local_path):
        print(f"  {fname}: already exists ({os.path.getsize(local_path)/1e6:.1f} MB)")
        continue
    print(f"  Downloading {url}...")
    try:
        r = requests.get(url, timeout=300)
        with open(local_path, "wb") as f:
            f.write(r.content)
        print(f"    -> {local_path} ({len(r.content)/1e6:.1f} MB)")
    except Exception as e:
        print(f"    ERROR: {e}")

# Check what's inside
print("\n=== WildDeepfake contents ===")
for fname in os.listdir(WD_DIR):
    if fname.endswith(".tar.gz"):
        path = os.path.join(WD_DIR, fname)
        size = os.path.getsize(path)
        try:
            with tarfile.open(path, "r:gz") as tar:
                members = tar.getmembers()
                print(f"  {fname}: {size/1e6:.1f} MB, {len(members)} images")
                if members:
                    m = members[0]
                    f = tar.extractfile(m)
                    img = Image.open(io.BytesIO(f.read()))
                    print(f"    First: {m.name}, size={img.size}")
        except Exception as e:
            print(f"  {fname}: {size/1e6:.1f} MB, ERROR: {e}")

# === 2. Download WLD YouTube videos ===
print("\n=== Downloading WLD YouTube videos ===")
wld_csv_url = "https://raw.githubusercontent.com/matyasbohacek/protecting-world-leaders-against-deep-fakes/main/world-leaders.csv"
r = requests.get(wld_csv_url)
lines = r.text.strip().split("\n")
urls = [l.split(";")[1] for l in lines[1:]]
print(f"Found {len(urls)} URLs")

yt_dlp = os.path.join(os.path.dirname(sys.executable), "yt-dlp.exe")
for i, url in enumerate(urls):
    out_path = os.path.join(WLD_DIR, f"video_{i:02d}.mp4")
    if os.path.exists(out_path):
        print(f"  [{i}] already exists: {out_path}")
        continue
    print(f"  [{i}] Downloading {url[:80]}...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "yt_dlp", "-f", "best[height<=480]", "-o", out_path, url],
            capture_output=True, text=True, timeout=300
        )
        if os.path.exists(out_path):
            print(f"    -> {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")
        else:
            print(f"    FAILED: {result.stderr[:200]}")
    except Exception as e:
        print(f"    ERROR: {e}")

print("\nDone!")
