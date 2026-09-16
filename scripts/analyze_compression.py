"""Analyze compression degradation experiment results.

Loads all 4 result sets (resnet50/vgg19 x keepaudio/osn) and produces:
  - Accuracy vs compression level (JPEG Q, H.264 CRF) with always-fake baseline
  - VT/AT/IM mean fake probs vs compression (real vs fake separated)
  - FER/SER mean confidence + entropy vs compression
  - Cohen's d (clean vs compressed) for probs, confidence, entropy
  - CSVs (long-format) + summary JSON for downstream plotting
"""
import os, json
import numpy as np
import pandas as pd

OUT = "E:\\Saad_audi deepfakes\\results\\compression_degradation"
GT_PATH = "E:\\Saad_audi deepfakes\\data\\lavdf_ground_truth.json"
JPEG_QS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
H264_CRFS = [18, 23, 28, 33, 38, 43]

with open(GT_PATH) as f:
    GT = json.load(f)


def cohens_d(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    sp = np.sqrt(((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2) / (na + nb - 2))
    if sp == 0:
        return np.nan
    return (a.mean() - b.mean()) / sp


def shannon_entropy(probs_row):
    p = np.clip(np.asarray(probs_row, dtype=float), 1e-12, 1.0)
    return float(-(p * np.log(p)).sum(axis=1).mean())


def load_results(tag):
    with open(os.path.join(OUT, f"results_{tag}.json")) as f:
        return json.load(f)


def analyze_tag(tag):
    data = load_results(tag)
    results = data["results"]
    rows = []
    for r in results:
        gt_label = GT.get(r["rel_path"], {}).get("label", "?")
        for cond, c in r["conditions"].items():
            base = dict(rel_path=r["rel_path"], video_id=r["video_id"], model_type=r["model_type"],
                        condition=cond, ground_truth=gt_label)
            if "error" in c:
                base.update(ok=False, n_faces=c.get("n_faces", 0))
                rows.append(base)
                continue
            base.update(ok=True, n_faces=c["n_faces"], n_seg=c["n_seg"],
                        prediction=c["prediction"],
                        VT=c["probs"]["VT"], AT=c["probs"]["AT"], IM=c["probs"]["IM"])
            if "vp" in c:
                cp = np.array(c["vp"])
                used = cp[: c["n_faces"]]
                conf = float(used.max(axis=1).mean())
                ent = shannon_entropy(used)
                base.update(fer_conf=conf, fer_entropy=ent)
            if "ap" in c:
                ap_ = np.array(c["ap"])
                used_a = ap_[: c["n_seg"]]
                conf_a = float(used_a.max(axis=1).mean())
                ent_a = shannon_entropy(used_a)
                base.update(ser_conf=conf_a, ser_entropy=ent_a)
            rows.append(base)
    df = pd.DataFrame(rows)
    return df


def summarize(df, tag):
    print(f"\n{'='*70}\nTAG: {tag}\n{'='*70}")
    fake_ratio = (df[df.condition == "clean"].ground_truth == "fake").mean()
    print(f"Always-fake baseline: {fake_ratio:.4f}")
    print(f"\n-- Accuracy by condition (videos with ok) --")
    accs = {}
    for cond, g in df[df.ok & (df.ground_truth != "?")].groupby("condition"):
        acc = (g.prediction == g.ground_truth).mean()
        accs[cond] = acc
    order = ["clean"] + [f"jpeg_{q}" for q in JPEG_QS] + [f"h264_{c}_keepaudio" for c in H264_CRFS]
    for cond in order:
        if cond in accs:
            n = len(df[(df.condition == cond) & df.ok])
            print(f"  {cond:12s}: acc={accs[cond]:.4f} (n={n})")

    print(f"\n-- Mean probs (real vs fake) by condition --")
    for cond in order:
        g = df[(df.condition == cond) & df.ok]
        if g.empty: continue
        vr = g[g.ground_truth == "real"].VT.mean()
        vf = g[g.ground_truth == "fake"].VT.mean()
        ar = g[g.ground_truth == "real"].AT.mean()
        af = g[g.ground_truth == "fake"].AT.mean()
        imr = g[g.ground_truth == "real"].IM.mean()
        imf = g[g.ground_truth == "fake"].IM.mean()
        print(f"  {cond:12s}: VT r={vr:.3f}/f={vf:.3f}  AT r={ar:.3f}/f={af:.3f}  IM r={imr:.3f}/f={imf:.3f}")

    print(f"\n-- FER/SER confidence & entropy (clean vs compressed) --")
    clean = df[(df.condition == "clean") & df.ok]
    for col, name in [("fer_conf", "FER conf"), ("fer_entropy", "FER ent"),
                      ("ser_conf", "SER conf"), ("ser_entropy", "SER ent")]:
        if col not in clean: continue
        base_val = clean[col].mean()
        print(f"  {name}: clean={base_val:.4f}", end="")
        for cond in [f"jpeg_{q}" for q in [10, 50, 95]] + [f"h264_{c}_keepaudio" for c in [18, 28, 43]]:
            g = df[(df.condition == cond) & df.ok & df[col].notna()]
            if not g.empty:
                print(f" | {cond}={g[col].mean():.4f}", end="")
        print()

    print(f"\n-- Cohen's d (clean vs compressed), mean probs --")
    for cond in order:
        if cond == "clean": continue
        g = df[(df.condition == cond) & df.ok]
        c = clean
        if g.empty: continue
        d_vt = cohens_d(c.VT, g.VT)
        d_at = cohens_d(c.AT, g.AT)
        d_im = cohens_d(c.IM, g.IM)
        print(f"  {cond:12s}: d_VT={d_vt:+.3f}  d_AT={d_at:+.3f}  d_IM={d_im:+.3f}")

    # Save long-format CSV without embeddings
    cols = ["rel_path", "video_id", "model_type", "condition", "ground_truth",
            "ok", "n_faces", "n_seg", "prediction", "VT", "AT", "IM",
            "fer_conf", "fer_entropy", "ser_conf", "ser_entropy"]
    df[cols].to_csv(os.path.join(OUT, f"long_{tag}.csv"), index=False)
    print(f"\nLong-format saved: long_{tag}.csv")

    return accs


def main():
    tags = ["resnet50_keepaudio", "vgg19_keepaudio", "resnet50_osn", "vgg19_osn"]
    summary_out = {}
    for tag in tags:
        df = analyze_tag(tag)
        summarize(df, tag)
        summary_out[tag] = {"n_videos": df[df.condition == "clean"].shape[0],
                            "always_fake": float((df[df.condition == "clean"].ground_truth == "fake").mean())}
    with open(os.path.join(OUT, "analysis_summary.json"), "w") as f:
        json.dump(summary_out, f, indent=2)
    print("\nAll tags analyzed.")


if __name__ == "__main__":
    main()