"""Generate plots for the compression degradation experiment."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "E:\\Saad_audi deepfakes\\results\\compression_degradation"
JPEG_QS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
H264_CRFS = [18, 23, 28, 33, 38, 43]
PLOT_DIR = os.path.join(OUT, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def per_frame_agreement(results, cond_clean="clean"):
    """Return dict cond -> mean FER per-frame argmax agreement vs clean."""
    ag = {}
    for v in results:
        c = v["conditions"].get(cond_clean)
        if c is None or "vp" not in c:
            continue
        for cond, t in v["conditions"].items():
            if cond == cond_clean or "vp" not in t:
                continue
            vpc = np.array(c["vp"])[: t["n_faces"]]
            vpt = np.array(t["vp"])[: t["n_faces"]]
            n = min(len(vpc), len(vpt))
            if n == 0:
                continue
            ag.setdefault(cond, []).append((vpc[:n].argmax(1) == vpt[:n].argmax(1)).mean())
    return {k: float(np.mean(v)) for k, v in ag.items()}


def load(tag):
    with open(os.path.join(OUT, f"results_{tag}.json")) as f:
        return json.load(f)["results"]


tags = ["resnet50_keepaudio", "vgg19_keepaudio", "resnet50_osn", "vgg19_osn"]
data = {t: load(t) for t in tags}
agreement = {t: per_frame_agreement(data[t]) for t in tags}

# --- Plot 1: FER per-frame agreement vs JPEG Q ---
fig, ax = plt.subplots(figsize=(7, 5))
for tag, c in [("resnet50_keepaudio", "tab:blue"), ("vgg19_keepaudio", "tab:orange")]:
    vals = [agreement[tag].get(f"jpeg_{q}") for q in JPEG_QS]
    ax.plot(JPEG_QS, vals, marker="o", label=tag.split("_")[0], color=c)
ax.set_xlabel("JPEG Quality (Q)")
ax.set_ylabel("FER per-frame agreement vs clean")
ax.set_title("FER emotion label stability under JPEG compression")
ax.legend()
ax.grid(alpha=0.3)
ax.invert_xaxis()
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "fer_jpeg_agreement.png"), dpi=150)
plt.close(fig)

# --- Plot 2: FER per-frame agreement vs H.264 CRF ---
fig, ax = plt.subplots(figsize=(7, 5))
for tag, c in [("resnet50_keepaudio", "tab:blue"), ("vgg19_keepaudio", "tab:orange")]:
    vals = [agreement[tag].get(f"h264_{crf}_keepaudio") for crf in H264_CRFS]
    ax.plot(H264_CRFS, vals, marker="s", label=tag.split("_")[0], color=c)
ax.set_xlabel("H.264 CRF (higher = more compression)")
ax.set_ylabel("FER per-frame agreement vs clean")
ax.set_title("FER emotion label stability under H.264 re-encode")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "fer_h264_agreement.png"), dpi=150)
plt.close(fig)

# --- Plot 3: KB fake probs (VT/IM) vs compression, real vs fake ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, mod in [("VT", "VT"), ("IM", "IM")]:
    for tag, c in [("resnet50_keepaudio", "tab:blue")]:
        pass
# load long CSV for probs
import pandas as pd
df = pd.read_csv(os.path.join(OUT, "long_resnet50_keepaudio.csv"))
ax = axes[0]
for cls, c in [("real", "tab:green"), ("fake", "tab:red")]:
    g = df[(df.condition == "clean") & (df.ground_truth == cls)]
    xs = [g.VT.mean()]
    for q in JPEG_QS:
        xs.append(df[(df.condition == f"jpeg_{q}") & (df.ground_truth == cls)].VT.mean())
    ax.plot([0] + JPEG_QS, xs, marker="o", label=f"VT {cls}", color=c)
ax.axhline(0.47, ls="--", color="gray", lw=0.8)
ax.set_xlabel("condition index (0=clean, 1-10=JPEG Q)")
ax.set_ylabel("VT fake prob")
ax.set_title("VT fake probability (ResNet50)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax = axes[1]
for cls, c in [("real", "tab:green"), ("fake", "tab:red")]:
    g = df[(df.condition == "clean") & (df.ground_truth == cls)]
    xs = [g.IM.mean()]
    for q in JPEG_QS:
        xs.append(df[(df.condition == f"jpeg_{q}") & (df.ground_truth == cls)].IM.mean())
    ax.plot([0] + JPEG_QS, xs, marker="o", label=f"IM {cls}", color=c)
ax.axhline(0.47, ls="--", color="gray", lw=0.8)
ax.set_xlabel("condition index (0=clean, 1-10=JPEG Q)")
ax.set_ylabel("IM fake prob")
ax.set_title("IM fake probability (ResNet50)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "kb_probs_vs_jpeg.png"), dpi=150)
plt.close(fig)

# --- Plot 4: FER confidence & entropy vs compression ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax_i, (col, name) in enumerate([("fer_conf", "FER mean confidence"), ("fer_entropy", "FER mean entropy")]):
    ax = axes[ax_i]
    for tag, c in [("resnet50_keepaudio", "tab:blue"), ("vgg19_keepaudio", "tab:orange")]:
        d = pd.read_csv(os.path.join(OUT, f"long_{tag}.csv"))
        vals = [d[(d.condition == "clean")][col].mean()]
        for q in JPEG_QS:
            vals.append(d[(d.condition == f"jpeg_{q}")][col].mean())
        ax.plot([0] + JPEG_QS, vals, marker="o", label=tag.split("_")[0], color=c)
    ax.set_xlabel("condition index (0=clean, 1-10=JPEG Q)")
    ax.set_ylabel(name)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "fer_conf_entropy_vs_jpeg.png"), dpi=150)
plt.close(fig)

print("Plots saved to", PLOT_DIR)

# Save agreement summary
with open(os.path.join(OUT, "per_frame_agreement.json"), "w") as f:
    json.dump(agreement, f, indent=1)
print("agreement summary saved")
