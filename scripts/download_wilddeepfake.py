import sys
sys.path.insert(0, "E:\\Saad_audi deepfakes")

try:
    from datasets import load_dataset
    ds = load_dataset("xingjunm/WildDeepfake", split="train", streaming=True)
    print("Dataset loaded (streaming)")
    for i, example in enumerate(ds):
        if i < 3:
            print(f"Example {i}: keys={list(example.keys())}")
            img = example["png"]
            print(f"  image type: {type(img)}, size: {img.size if hasattr(img, 'size') else 'N/A'}")
            print(f"  key: {example['__key__']}")
        else:
            break
    print("WildDeepfake is downloadable via HF datasets")

    # Count total examples
    total = 0
    for _ in ds:
        total += 1
        if total >= 10000:
            break
    print(f"Counted {total} examples (of ~994k train)")

    # Get class distribution
    ds2 = load_dataset("xingjunm/WildDeepfake", split="train", streaming=True)
    counts = {}
    for i, ex in enumerate(ds2):
        key = ex["__key__"]
        label = "fake" if "/fake/" in key else "real" if "/real/" in key else "unknown"
        counts[label] = counts.get(label, 0) + 1
        if i >= 2000:
            break
    print(f"Class distribution (first 2000): {counts}")

except ImportError:
    print("datasets library not installed, need to install")
except Exception as e:
    print(f"Error: {e}")
