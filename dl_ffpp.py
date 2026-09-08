from huggingface_hub import snapshot_download
p = snapshot_download(repo_id="bitmind/FaceForensicsC23", repo_type="dataset", local_dir=r"E:\Datasets\FaceForensicsC23")
print("DONE", p)