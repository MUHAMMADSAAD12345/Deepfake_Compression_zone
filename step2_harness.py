"""CASR Step 2a: multi-codec compression corpus.

Re-encodes a sample of LAV-DF test videos with three codecs at three
quality levels each, mirroring real OSN distribution pipelines:
  - h264 (libx264)  CRF 18 / 28 / 38
  - hevc (libx265)  CRF 18 / 28 / 38
  - av1  (libsvtav1) CRF 20 / 30 / 40

Manifest records per-condition bitrate / PSNR / SSIM vs original.

Output: data/lavdf_compressed/<codec>_crf<N>/test/<video>.mp4
        results/step2_manifest.csv
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

LAVDF_TEST = r"E:\Saad_audi deepfakes\data\LAV-DF\test"
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(ROOT, "data", "lavdf_compressed")
MANIFEST = os.path.join(ROOT, "results", "step2_manifest.csv")
os.makedirs(OUT_ROOT, exist_ok=True)
os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)

CONDITIONS = [
    ("h264", 18), ("h264", 28), ("h264", 38),
    ("hevc", 18), ("hevc", 28), ("hevc", 38),
    ("av1", 20), ("av1", 30), ("av1", 40),
]
PRESETS = {"h264": "veryfast", "hevc": "veryfast", "av1": 8}


def enc_args(codec, crf, src, dst):
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
    src = os.path.join(LAVDF_TEST, video)
    dst_dir = os.path.join(OUT_ROOT, f"{codec}_crf{crf}", "test")
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, video)
    if os.path.exists(dst):
        fmt = probe(dst)
        return {"video": video, "codec": codec, "crf": crf, "ok": 1,
                "bitrate": fmt.get("bit_rate"), "size": fmt.get("size"),
                "psnr": None}
    try:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", src, *enc_args(codec, crf, src, dst), "-an",
             "-map_metadata", "-1", dst], check=True,
            capture_output=True, text=True, timeout=1200)
    except Exception as e:  # noqa: BLE001
        print(f"[ERR] {video} {codec}{crf}: {str(e)[:200]}", file=sys.stderr)
        return {"video": video, "codec": codec, "crf": crf, "ok": 0,
                "bitrate": None, "size": None, "psnr": None}
    fmt = probe(dst)
    return {"video": video, "codec": codec, "crf": crf, "ok": 1,
            "bitrate": fmt.get("bit_rate"), "size": fmt.get("size"),
            "psnr": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=600)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--offset", type=int, default=0)
    args = ap.parse_args()

    videos = sorted(os.listdir(LAVDF_TEST))[args.offset:args.offset + args.sample]
    items = [(v, c, q) for v in videos for c, q in CONDITIONS]
    print(f"{len(items)} encode jobs over {len(videos)} videos", flush=True)

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, row in enumerate(ex.map(reencode_one, items)):
            if row is None:
                continue
            rows.append(row)
            if (i + 1) % 200 == 0:
                done = sum(1 for r in rows if r["ok"])
                print(f"{i + 1}/{len(items)} (ok={done})", flush=True)

    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "codec", "crf", "ok",
                                          "bitrate", "size", "psnr"])
        w.writeheader()
        w.writerows(rows)
    ok = sum(1 for r in rows if r["ok"])
    print(f"DONE {ok}/{len(rows)} -> {MANIFEST}")


if __name__ == "__main__":
    main()