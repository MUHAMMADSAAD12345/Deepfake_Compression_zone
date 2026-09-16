"""Download the full WildDeepfake dataset from Hugging Face"""
import os, sys, json, requests, time, tarfile, io
from PIL import Image

WD_DIR = "E:\\Saad_audi deepfakes\\data\\WildDeepfake"
os.makedirs(WD_DIR, exist_ok=True)

api_url = "https://huggingface.co/api/datasets/xingjunm/WildDeepfake"
print("Fetching file listing from HF API...")
r = requests.get(api_url)
data = r.json()
siblings = [s["rfilename"] for s in data.get("siblings", []) if s["rfilename"].endswith(".tar.gz")]

categories = {}
for s in siblings:
    parts = s.replace("deepfake_in_the_wild/", "", 1).split("/")
    if len(parts) == 2:
        cat = parts[0]
        categories.setdefault(cat, []).append(int(parts[1].replace(".tar.gz", "")))

for cat, nums in sorted(categories.items()):
    print(f"  {cat}: {len(nums)} shards [{min(nums)}-{max(nums)}]")

base_url = "https://huggingface.co/datasets/xingjunm/WildDeepfake/resolve/main"

def count_images(tar_path):
    try:
        with tarfile.open(tar_path, "r") as tar:
            return sum(1 for m in tar.getmembers() if m.isfile() and m.name.endswith(".png"))
    except:
        return 0

existing_valid = set()
for f in os.listdir(WD_DIR):
    if f.endswith(".tar.gz"):
        path = os.path.join(WD_DIR, f)
        if os.path.getsize(path) > 1_000_000:
            existing_valid.add(f)
            n = count_images(path)
            print(f"  Already have: {f} ({os.path.getsize(path)/1e6:.1f} MB, {n} images)")

total_downloaded = 0
total_images = 0
failed = []

for s in siblings:
    parts = s.replace("deepfake_in_the_wild/", "", 1).split("/")
    cat, fname = parts[0], parts[1]
    safe_name = f"{cat}_{fname}"
    local_path = os.path.join(WD_DIR, safe_name)

    if safe_name in existing_valid:
        n = count_images(local_path)
        total_images += n
        continue

    url = f"{base_url}/{s}"
    print(f"Downloading {safe_name}...", end=" ", flush=True)
    try:
        resp = requests.get(url, timeout=300)
        if len(resp.content) < 1000:
            print(f"SKIP (empty: {len(resp.content)} bytes)")
            continue
        with open(local_path, "wb") as f:
            f.write(resp.content)
        n = count_images(local_path)
        print(f"{len(resp.content)/1e6:.1f} MB, {n} images")
        total_images += n
        total_downloaded += 1
    except Exception as e:
        print(f"FAILED: {e}")
        failed.append(s)
        if os.path.exists(local_path):
            os.remove(local_path)

print(f"\nDone. Downloaded {total_downloaded} new shards.")
print(f"Total images across all valid shards: {total_images}")
if failed:
    print(f"Failed: {len(failed)} shards")
    for f in failed[:10]:
        print(f"  {f}")
