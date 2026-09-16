"""Extract WildDeepfake into ImageFolder structure: {train,test}/{real,fake}/*.png"""
import os, tarfile, glob, shutil, re

SRC = "E:\\Saad_audi deepfakes\\data\\WildDeepfake"
DST = "E:\\Saad_audi deepfakes\\data\\WildDeepfake_extracted"

def extract_shard(shard_path):
    name = os.path.basename(shard_path).replace(".tar.gz", "").replace(".tar", "")
    m = re.match(r"(fake|real)_(train|test)_(\d+)", name)
    if not m:
        return 0
    label, split, _ = m.groups()
    target_split = "train" if split == "train" else "test"
    out_dir = os.path.join(DST, target_split, label)
    os.makedirs(out_dir, exist_ok=True)

    count = 0
    with tarfile.open(shard_path, "r") as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith(".png"):
                f = tar.extractfile(member)
                if f is None:
                    continue
                src_name = os.path.basename(member.name)
                dst_path = os.path.join(out_dir, f"{name}_{src_name}")
                with open(dst_path, "wb") as out:
                    shutil.copyfileobj(f, out)
                count += 1
    return count

shards = sorted(glob.glob(os.path.join(SRC, "*.tar.gz")))
print(f"Found {len(shards)} shards")

total = 0
for i, shard in enumerate(shards):
    c = extract_shard(shard)
    total += c
    if (i + 1) % 100 == 0 or i == len(shards) - 1:
        print(f"  [{i+1}/{len(shards)}] Extracted {c} images (running total: {total})")

print(f"\nDone. Total images: {total}")
