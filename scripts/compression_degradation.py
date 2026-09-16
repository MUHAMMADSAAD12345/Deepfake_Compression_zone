"""Compression degradation experiment for the KB pipeline on LAV-DF.

Measures how FER/SER/KB signals and final classification degrade under:
  1. JPEG frame compression (Q=10..95) on full frames BEFORE face detection
  2. H.264 CRF video re-encoding with ORIGINAL audio preserved
  3. H.264 CRF video re-encoding with FULL OSN pipeline (AAC audio re-encode)

Outputs: JSON (per-video), CSV (summary rows), NPZ (large probs arrays).
Parallel: multiprocessing over videos with spawn (Windows-safe).
"""
import os, sys, json, time, subprocess, tempfile, argparse, gc
import multiprocessing as mp

sys.path.insert(0, "E:\\Saad_audi deepfakes")

import numpy as np
import cv2
import torch
import torch.multiprocessing as tmp
tmp.set_start_method("spawn", force=True)

from src.utils.preprocessing import extract_frames, extract_audio_segments, detect_face, sync_frames_and_segments

DATA_DIR = "E:\\Saad_audi deepfakes\\data\\LAV-DF"
GT_PATH = "E:\\Saad_audi deepfakes\\data\\lavdf_ground_truth.json"
CHK_DIR = "E:\\Saad_audi deepfakes\\checkpoints"
OUT_DIR = "E:\\Saad_audi deepfakes\\results\\compression_degradation"
TMP_DIR = "E:\\Saad_audi deepfakes\\results\\compression_degradation\\tmp_h264"

JPEG_QS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
H264_CRFS = [18, 23, 28, 33, 38, 43]

FER_INFOS = {
    "resnet50": {
        "weights": os.path.join(CHK_DIR, "fer_resnet50_best.pt"),
        "is_vgg19": False,
    },
    "vgg19": {
        "weights": os.path.join(CHK_DIR, "fer2013_vgg19.pt"),
        "is_vgg19": True,
    },
}


def compress_frames_jpeg(frames_rgb, quality):
    """JPEG-encode/decode full RGB frames (in-memory)."""
    out = []
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    for f in frames_rgb:
        enc = cv2.imencode(".jpg", cv2.cvtColor(f, cv2.COLOR_RGB2BGR), params)[1]
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        out.append(cv2.cvtColor(dec, cv2.COLOR_BGR2RGB))
    return out


def compress_video_h264(input_path, output_path, crf, reencode_audio, preset="veryfast"):
    """Re-encode video with libx264. If reencode_audio: full OSN (AAC 96k).
    Else: copy original audio stream unchanged."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", input_path,
        "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
    ]
    if reencode_audio:
        cmd += ["-c:a", "aac", "-b:a", "96k"]
    else:
        cmd += ["-c:a", "copy"]
    cmd += [output_path]
    subprocess.run(cmd, check=True, capture_output=True)


def process_one_video(args):
    """Worker for a single (video_rel_path, model_type). Runs all compression
    conditions for that video + model and returns a results dict."""
    rel_path, model_type, use_full_osn = args["rel_path"], args["model_type"], args["use_full_osn"]
    fer_weights = FER_INFOS[model_type]["weights"]

    # Import inside worker (spawn-safe)
    from src.models.fer_model import FERModel
    from src.models.ser_model import SERModel
    from src.reasoning.intra_modality import intra_modality_reasoning
    from src.reasoning.inter_modality import inter_modality_reasoning
    from src.reasoning.classifier import compute_fake_probabilities, classify_modality, voting_classifier

    vpath = os.path.join(DATA_DIR, rel_path)
    vid = rel_path.replace("\\", "/").split("/")[-1].replace(".mp4", "")

    # Load models ONCE per video, reuse across all conditions
    fer = FERModel(fer_weights, model_type=model_type)
    ser = SERModel(os.path.join(CHK_DIR, "ser_weights.pt"))

    result = {
        "rel_path": rel_path,
        "video_id": vid,
        "model_type": model_type,
        "conditions": {},
    }

    # --- Baseline: clean original video ---
    frames, frame_ts, fps = extract_frames(vpath)
    segments, segment_ts, raw_audio, sr = extract_audio_segments(vpath)
    face_data = [(detect_face(f), ts) for f, ts in zip(frames, frame_ts)]
    face_data = [(f, ts) for f, ts in face_data if f is not None]
    if not face_data:
        result["error"] = "no_faces_base"
        return result
    base_faces, base_frame_ts = zip(*face_data)
    base_faces, base_frame_ts = list(base_faces), list(base_frame_ts)
    base_sk = sync_frames_and_segments(frames, base_frame_ts, segments, segment_ts)

    def run_kb(faces, segs, sync):
        ve, vp = fer.predict_batch(faces)
        ae, ap = ser.predict_batch(segs)
        vt_r, vt_fi = intra_modality_reasoning(ve)
        at_r, at_fi = intra_modality_reasoning(ae)
        im_r, im_fi = inter_modality_reasoning(ve, ae, sync)
        pd = compute_fake_probabilities(vt_r, at_r, im_r)
        mr = classify_modality(pd)
        pred = voting_classifier(mr)
        return {
            "prediction": pred,
            "modality": mr,
            "probs": {k: v["probability"] for k, v in pd.items()},
            "ve": ve, "vp": vp, "ae": ae, "ap": ap,
            "n_faces": len(faces),
            "n_seg": len(segs),
        }

    base = run_kb(base_faces, segments, base_sk)
    result["conditions"]["clean"] = {
        "prediction": base["prediction"],
        "modality": base["modality"],
        "probs": base["probs"],
        "n_faces": base["n_faces"],
        "n_seg": base["n_seg"],
    }
    if args["collect_probs"]:
        result["conditions"]["clean"]["vp"] = base["vp"].tolist()
        result["conditions"]["clean"]["ap"] = base["ap"].tolist()

    # --- JPEG frame compression (full frames before face detection) ---
    for q in JPEG_QS:
        cframes = compress_frames_jpeg(frames, q)
        cface_data = [(detect_face(f), ts) for f, ts in zip(cframes, frame_ts)]
        cface_data = [(f, ts) for f, ts in cface_data if f is not None]
        if not cface_data:
            result["conditions"][f"jpeg_{q}"] = {"error": "no_faces", "n_faces": 0}
            continue
        cf, cts = zip(*cface_data)
        cf, cts = list(cf), list(cts)
        c_sk = sync_frames_and_segments(cframes, cts, segments, segment_ts)
        r = run_kb(cf, segments, c_sk)
        result["conditions"][f"jpeg_{q}"] = {
            "prediction": r["prediction"],
            "modality": r["modality"],
            "probs": r["probs"],
            "n_faces": r["n_faces"],
            "n_seg": r["n_seg"],
        }
        if args["collect_probs"]:
            result["conditions"][f"jpeg_{q}"]["vp"] = r["vp"].tolist()
            result["conditions"][f"jpeg_{q}"]["ap"] = r["ap"].tolist()
        del r
        gc.collect()

    # --- H.264 CRF (original audio preserved) ---
    if not args["skip_h264"]:
        os.makedirs(TMP_DIR, exist_ok=True)
        for crf in H264_CRFS:
            out_path = os.path.join(TMP_DIR, f"{vid}_{model_type}_crf{crf}_keepaudio.mp4")
            try:
                compress_video_h264(vpath, out_path, crf, reencode_audio=False)
                hframes, hts, hfps = extract_frames(out_path)
                hsegs, hseg_ts, hraw, hsr = extract_audio_segments(out_path)
                hface = [(detect_face(f), ts) for f, ts in zip(hframes, hts)]
                hface = [(f, ts) for f, ts in hface if f is not None]
                if not hface:
                    result["conditions"][f"h264_{crf}_keepaudio"] = {"error": "no_faces", "n_faces": 0}
                    continue
                hf, hcts = zip(*hface)
                hf, hcts = list(hf), list(hcts)
                h_sk = sync_frames_and_segments(hframes, hcts, hsegs, hseg_ts)
                r = run_kb(hf, hsegs, h_sk)
                result["conditions"][f"h264_{crf}_keepaudio"] = {
                    "prediction": r["prediction"],
                    "modality": r["modality"],
                    "probs": r["probs"],
                    "n_faces": r["n_faces"],
                    "n_seg": r["n_seg"],
                }
                if args["collect_probs"]:
                    result["conditions"][f"h264_{crf}_keepaudio"]["vp"] = r["vp"].tolist()
                    result["conditions"][f"h264_{crf}_keepaudio"]["ap"] = r["ap"].tolist()
                del r
                gc.collect()
            except Exception as e:
                result["conditions"][f"h264_{crf}_keepaudio"] = {"error": str(e)[:200]}
            finally:
                if os.path.exists(out_path):
                    os.remove(out_path)

    print(f"[worker] {rel_path} ({model_type}) done", flush=True)
    return result


def main():
    torch.multiprocessing.freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200, help="Number of test videos")
    parser.add_argument("--model", choices=list(FER_INFOS.keys()), default=None,
                        help="FER model type; default: both")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip_h264", action="store_true")
    # NOTE: use_full_osn is a second pass flag passed per-run; kept for clarity
    parser.add_argument("--use_full_osn", action="store_true",
                        help="Re-encode audio too (AAC 96k). Run WITHOUT this flag first (keep original audio), then WITH it.")
    parser.add_argument("--collect_probs", action="store_true",
                        help="Store per-face/prob arrays (for Cohen's d and entropy analyses)")
    parser.add_argument("--tag", default="", help="Optional suffix for output files")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)

    with open(GT_PATH) as f:
        gt = json.load(f)
    items = [(k, v) for k, v in sorted(gt.items()) if v["split"] == "test"][: args.limit]
    print(f"Videos: {len(items)} | models: {args.model or 'both'} | workers: {args.workers} | "
          f"full_osn: {args.use_full_osn}")

    models = [args.model] if args.model else list(FER_INFOS.keys())

    for model_type in models:
        tag = f"{model_type}_{args.tag}" if args.tag else model_type
        osn_tag = "_osn" if args.use_full_osn else "_keepaudio"
        out_json = os.path.join(OUT_DIR, f"results_{tag}{osn_tag}.json")
        out_csv = os.path.join(OUT_DIR, f"results_{tag}{osn_tag}.csv")
        out_npz = os.path.join(OUT_DIR, f"results_{tag}{osn_tag}.npz")

        # Skip if already done
        if os.path.exists(out_json):
            print(f"[skip] {out_json} exists")
            continue

        job_args = [
            {
                "rel_path": rel,
                "model_type": model_type,
                "use_full_osn": args.use_full_osn,
                "collect_probs": args.collect_probs,
                "skip_h264": args.skip_h264,
            }
            for rel, _ in items
        ]

        t0 = time.time()
        if args.workers > 1:
            with mp.Pool(args.workers) as pool:
                results = pool.map(process_one_video, job_args, chunksize=1)
        else:
            results = [process_one_video(j) for j in job_args]
        elapsed = time.time() - t0
        print(f"\n[{model_type}] {len(results)} videos in {elapsed:.0f}s "
              f"({elapsed/max(len(results),1):.2f}s/video)")

        # Save JSON
        with open(out_json, "w") as f:
            json.dump({"spec": {"jpeg_qs": JPEG_QS, "h264_crfs": H264_CRFS,
                                "use_full_osn": args.use_full_osn,
                                "n_videos": len(results), "elapsed_s": elapsed},
                       "results": results}, f, indent=1)

        # Save CSV (one row per video per condition)
        header = ["rel_path", "video_id", "model_type", "condition",
                  "prediction", "ground_truth", "VT", "AT", "IM",
                  "n_faces", "n_seg"]
        with open(out_csv, "w", newline="") as f:
            import csv
            w = csv.writer(f)
            w.writerow(header)
            for r in results:
                gt_label = gt.get(r["rel_path"], {}).get("label", "?")
                for cond, c in r["conditions"].items():
                    if "error" in c:
                        w.writerow([r["rel_path"], r["video_id"], r["model_type"], cond,
                                    "error", gt_label, "", "", "", c.get("n_faces", 0), ""])
                    else:
                        w.writerow([r["rel_path"], r["video_id"], r["model_type"], cond,
                                    c["prediction"], gt_label,
                                    c["probs"].get("VT", ""), c["probs"].get("AT", ""),
                                    c["probs"].get("IM", ""), c["n_faces"], c["n_seg"]])

        # Save NPZ (big probability arrays when collect_probs)
        if args.collect_probs:
            npz_arrays = {}
            for r in results:
                for cond, c in r["conditions"].items():
                    if "vp" in c:
                        npz_arrays[f"{r['video_id']}_{cond}_vp"] = np.array(c["vp"])
                        npz_arrays[f"{r['video_id']}_{cond}_ap"] = np.array(c["ap"])
            if npz_arrays:
                with open(out_npz, "wb") as f:
                    np.savez(f, **npz_arrays)
                print(f"[{model_type}] NPZ: {len(npz_arrays)} arrays")

        print(f"[{model_type}] Saved: {out_json}\n  {out_csv}\n  {out_npz if args.collect_probs else '(no npz)'}")
        gc.collect()


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()