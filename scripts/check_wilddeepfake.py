import os, sys, json, requests
sys.path.insert(0, "E:\\Saad_audi deepfakes")

api_url = "https://huggingface.co/api/datasets/xingjunm/WildDeepfake"
r = requests.get(api_url)
data = r.json()
siblings = [s for s in data.get("siblings", []) if s["rfilename"].endswith(".tar.gz")]
total_size = 0
for s in siblings:
    size_mb = s.get("size", 0) / 1e6
    total_size += s.get("size", 0)
    print(f'  {s["rfilename"]}: {size_mb:.1f} MB')
print(f"\nTotal: {len(siblings)} files, {total_size/1e9:.2f} GB")
