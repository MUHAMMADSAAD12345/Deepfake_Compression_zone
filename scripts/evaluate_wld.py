"""Evaluate KB approach on WLD videos"""
import sys, os
sys.path.insert(0, "E:\\Saad_audi deepfakes")
from src.pipeline import DeepfakeDetector

WLD_DIR = "E:\\Saad_audi deepfakes\\data\\WLD"
FER_WEIGHTS = "E:\\Saad_audi deepfakes\\checkpoints\\fer_resnet50_best.pt"
SER_WEIGHTS = "E:\\Saad_audi deepfakes\\checkpoints\\ser_weights.pt"
MODEL_TYPE = "resnet50"

detector = DeepfakeDetector(fer_weights=FER_WEIGHTS, ser_weights=SER_WEIGHTS, model_type=MODEL_TYPE)

videos = sorted([f for f in os.listdir(WLD_DIR) if f.endswith(".mp4")])
print(f"Found {len(videos)} WLD videos\n")

for v in videos:
    path = os.path.join(WLD_DIR, v)
    print(f"{'='*60}")
    print(f"Video: {v} ({os.path.getsize(path)/1e6:.1f} MB)")
    print(f"{'='*60}")
    try:
        result = detector.predict_video(path)
        if "error" in result:
            print(f"  ERROR: {result['error']}")
            continue
        pred = result.get("prediction", "N/A")
        probs = result.get("probabilities", {})
        vt_p = probs.get('VT', {}).get('probability', 0)
        at_p = probs.get('AT', {}).get('probability', 0)
        im_p = probs.get('IM', {}).get('probability', 0)
        print(f"  Prediction: {pred}")
        print(f"  Probs: VT={vt_p:.3f} AT={at_p:.3f} IM={im_p:.3f}")
        mod = result.get("modality_results", {})
        print(f"  Modalities: VT={mod.get('VT','?')} AT={mod.get('AT','?')} IM={mod.get('IM','?')}")
    except Exception as e:
        print(f"  ERROR: {e}")
    print()
