import tarfile, json, os

TAR_PATH = r"E:\Downloads\LAV-DF.tar"
OUT_PATH = r"E:\audi deepfakes\data\lavdf_ground_truth.json"

with tarfile.open(TAR_PATH) as tar:
    f = tar.extractfile(tar.getmember("LAV-DF/metadata.min.json"))
    meta = json.load(f)

ground_truth = {}
for entry in meta:
    file_path = entry["file"]
    is_fake = entry["n_fakes"] > 0
    ground_truth[file_path] = {
        "label": "fake" if is_fake else "real",
        "split": entry["split"],
        "modify_video": entry["modify_video"],
        "modify_audio": entry["modify_audio"],
        "n_fakes": entry["n_fakes"],
    }

with open(OUT_PATH, "w") as f:
    json.dump(ground_truth, f, indent=2)

stats = {"real": 0, "fake": 0}
for v in ground_truth.values():
    stats[v["label"]] += 1
print(f"Ground truth saved to {OUT_PATH}")
print(f"Videos: {len(ground_truth)} total, {stats['real']} real, {stats['fake']} fake")
split_counts = {}
for v in ground_truth.values():
    split_counts[v["split"]] = split_counts.get(v["split"], 0) + 1
print(f"Splits: {split_counts}")
