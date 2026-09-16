"""Pre-resize all WildDeepfake images to 224x224 for fast training. Parallel workers."""
import os, sys, multiprocessing as mp
from PIL import Image

SRC = "E:\\Saad_audi deepfakes\\data\\WildDeepfake_extracted"
DST = "E:\\Saad_audi deepfakes\\data\\WildDeepfake_resized"
SIZE = (224, 224)
WORKERS = 14

def process_one(args):
    src_path, dst_path = args
    if os.path.exists(dst_path):
        return
    try:
        with Image.open(src_path) as im:
            im = im.convert("RGB").resize(SIZE, Image.BILINEAR)
            im.save(dst_path, "JPEG", quality=90)
        return 1
    except Exception as e:
        return 0

def process_split(split):
    base_src = os.path.join(SRC, split)
    base_dst = os.path.join(DST, split)
    tasks = []
    for label in os.listdir(base_src):
        src_dir = os.path.join(base_src, label)
        dst_dir = os.path.join(base_dst, label)
        os.makedirs(dst_dir, exist_ok=True)
        for fname in os.listdir(src_dir):
            src = os.path.join(src_dir, fname)
            dst = os.path.join(dst_dir, fname.replace(".png", ".jpg"))
            tasks.append((src, dst))
    print(f"[{split}] {len(tasks)} images", flush=True)
    done = 0
    with mp.Pool(WORKERS) as pool:
        for r in pool.imap_unordered(process_one, tasks, chunksize=256):
            done += 1
            if done % 50000 == 0:
                print(f"[{split}] {done}/{len(tasks)}", flush=True)
    print(f"[{split}] DONE {done}", flush=True)

if __name__ == "__main__":
    mp.freeze_support()
    for split in ["train", "test"]:
        process_split(split)
    print("ALL DONE", flush=True)
