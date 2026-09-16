import json

calib = json.load(open(r"E:\audi deepfakes\results\calibration.json"))
print("=== Calibration Results (200 dev videos) ===")
print(f"  AUC: {calib['auc']:.4f}")
print(f"  Best mean threshold: {calib['best_mean_threshold']:.2f}, acc={calib['best_mean_accuracy']:.2%}")
print(f"  Best vote threshold: {calib['best_vote_threshold']:.2f}, acc={calib['best_vote_accuracy']:.2%}")
print()

results = json.load(open(r"E:\audi deepfakes\results\lavdf_eval_full.json"))
details = results["details"]
real_preds = [v for v in details.values() if v["ground_truth"] == "real"]
fake_preds = [v for v in details.values() if v["ground_truth"] == "fake"]
real_correct = sum(1 for v in real_preds if v["prediction"] == "real")
fake_correct = sum(1 for v in fake_preds if v["prediction"] == "fake")

print("=== Full Test Results (26 videos) ===")
print(f"  Accuracy: {results['accuracy']:.2%}")
print(f"  Real: {real_correct}/{len(real_preds)} correct")
print(f"  Fake: {fake_correct}/{len(fake_preds)} correct")

real_probs = [(v["VT_prob"], v["AT_prob"], v["IM_prob"]) for v in real_preds]
fake_probs = [(v["VT_prob"], v["AT_prob"], v["IM_prob"]) for v in fake_preds]
if real_probs:
    print(f"  Mean probs REAL: VT={sum(p[0] for p in real_probs)/len(real_probs):.1%}, "
          f"AT={sum(p[1] for p in real_probs)/len(real_probs):.1%}, "
          f"IM={sum(p[2] for p in real_probs)/len(real_probs):.1%}")
if fake_probs:
    print(f"  Mean probs FAKE: VT={sum(p[0] for p in fake_probs)/len(fake_probs):.1%}, "
          f"AT={sum(p[1] for p in fake_probs)/len(fake_probs):.1%}, "
          f"IM={sum(p[2] for p in fake_probs)/len(fake_probs):.1%}")
