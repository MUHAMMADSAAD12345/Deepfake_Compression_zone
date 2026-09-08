"""CASR Step 2b-v2: FF++ c23 multi-codec corpus (codec-transfer matrix material).

Samples real + Deepfakes videos from the mirrored FF++ c23 pack, re-encodes
with H.264 / HEVC / AV1 at matched CRF levels, records bitrates.

Outputs (mirrors step2 conventions):
  data/ffpp_compressed/<codec>_crf<N>/<split>/<video>.mp4
  results/step2b_manifest.csv
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

FFPP_ROOT = r"E:\Datasets\FaceForensicsC23\FaceForensics++_C23"
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(ROOT, "data", "ffpp_compressed")
MANIFEST = os.path.join(ROOT, "results", "step2b_manifest.csv")
os.makedirs(OUT_ROOT, exist_ok=True)
os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)

CONDITIONS = [
    ("h264", 18), ("h264", 28), ("h264", 38),
    ("hevc", 18), ("hevc", 28), ("hevc", 38),
    ("av1", 20), ("av1", 30), ("av1", 40),
]
PRESETS = {"h264": "ultrafast", "hevc": "ultrafast", "av1": 10}


def enc_args(codec, crf):
    if codec == "h264":
        return ["-c:v", "libx264", "-preset", PRESETS[codec], "-crf", str(crf)]
    if codec == "hevc":
        return ["-c:v", "libx265", "-preset", PRESETS[codec], "-crf", str(crf),
                "-x265-params", "log-level=error"]
    return ["-c:v", "libsvtav1", "-preset", str(PRESETS[codec]), "-crf", str(crf)]


def probe(path):
    r = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration,bit_rate,size", "-of", "json", path]).decode()
    return json.loads(r)["format"]


def reencode_one(item):
    video, codec, crf = item
    src = os.path.join(FFPP_ROOT, video["rel"])
    dst_dir = os.path.join(OUT_ROOT, f"{codec}_crf{crf}", video["split"])
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, video["name"])
    if os.path.exists(dst):
        try:
            fmt = probe(dst)
            return {"video": video["name"], "split": video["split"],
                    "codec": codec, "crf": crf, "ok": 1,
                    "bitrate": fmt.get("bit_rate"), "size": fmt.get("size")}
        except Exception:  # noqa: BLE001  -- truncated/corrupt stub: re-encode
            os.remove(dst)
    try:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", src, *enc_args(codec, crf), "-an", "-map_metadata", "-1", dst],
            check=True, capture_output=True, text=True, timeout=1800)
    except Exception as e:  # noqa: BLE001
        print(f"[ERR] {video['name']} {codec}{crf}: {str(e)[:200]}", file=sys.stderr)
        return {"video": video["name"], "split": video["split"],
                "codec": codec, "crf": crf, "ok": 0,
                "bitrate": None, "size": None}
    try:
        fmt = probe(dst)
    except Exception as e:  # noqa: BLE001
        print(f"[ERR] probe {video['name']} {codec}{crf}: {str(e)[:200]}",
              file=sys.stderr)
        return {"video": video["name"], "split": video["split"],
                "codec": codec, "crf": crf, "ok": 0,
                "bitrate": None, "size": None}
    return {"video": video["name"], "split": video["split"],
            "codec": codec, "crf": crf, "ok": 1,
            "bitrate": fmt.get("bit_rate"), "size": fmt.get("size")}


METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
# FF++ standard protocol splits by numeric video id (1000 per group)
SPLITS = {"train": (0, 719), "val": (720, 859), "test": (860, 999)}


def list_videos(group, split):
    """group in METHODS or 'real'; positional selection over sorted names
    (FF++ protocol: sorted index 0-719 train, 720-859 val, 860-999 test).
    NOTE: mirror names are 'NNN_NNN.mp4' (not sequential ints) -- positional
    selection avoids int-parsing altogether."""
    if group == "real":
        d = os.path.join(FFPP_ROOT, "real")
        prefix = "real"
    else:
        d = os.path.join(FFPP_ROOT, "fake", group)
        prefix = os.path.join("fake", group)
    lo, hi = SPLITS[split]
    names = sorted(os.listdir(d))
    sel = names[lo:hi + 1]
    return [(os.path.join(prefix, n), n) for n in sel]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=METHODS,
                    help="manipulation methods (use --methods none for real only)")
    ap.add_argument("--include_real", action="store_true")
    ap.add_argument("--split", default="test", choices=list(SPLITS),
                    help="FF++ protocol split to sample")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    items = []
    groups = list(args.methods)
    if args.include_real:
        groups.append("real")
    for g in groups:
        for rel, name in list_videos(g, args.split):
            items.append({"rel": rel, "name": name, "split": g})

    jobs = [(v, c, q) for v in items for c, q in CONDITIONS]
    print(f"{len(jobs)} encode jobs over {len(items)} videos "
          f"(split={args.split}, groups={groups})", flush=True)

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, row in enumerate(ex.map(reencode_one, jobs)):
            if row is None:
                continue
            rows.append(row)
            if (i + 1) % 200 == 0:
                ok = sum(1 for r in rows if r["ok"])
                print(f"{i + 1}/{len(jobs)} (ok={ok})", flush=True)

    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "split", "codec", "crf",
                                          "ok", "bitrate", "size"])
        w.writeheader()
        w.writerows(rows)
    ok = sum(1 for r in rows if r["ok"])
    print(f"DONE {ok}/{len(rows)} -> {MANIFEST}")


if __name__ == "__main__":
    main()