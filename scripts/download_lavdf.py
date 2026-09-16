import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "lavdf")
os.makedirs(DATA_DIR, exist_ok=True)

def download_lavdf():
    """
    Download LAV-DF dataset from Hugging Face.
    Requirements:
      1. HF account: https://huggingface.co/join
      2. Accept terms: https://huggingface.co/datasets/ControlNet/LAV-DF
      3. Generate token: https://huggingface.co/settings/tokens
      4. Set HF_TOKEN env var or pass via --token
    """
    from huggingface_hub import snapshot_download
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: Set HF_TOKEN environment variable or login with `huggingface-cli login`")
        print("1. Create account: https://huggingface.co/join")
        print("2. Accept terms: https://huggingface.co/datasets/ControlNet/LAV-DF")
        print("3. Generate token: https://huggingface.co/settings/tokens")
        print("4. Run: set HF_TOKEN=hf_... && python scripts/download_lavdf.py")
        return False
    print("Downloading LAV-DF dataset (~25.6 GB)...")
    snapshot_download(
        repo_id="ControlNet/LAV-DF",
        local_dir=DATA_DIR,
        token=token,
        resume_download=True,
    )
    print(f"LAV-DF downloaded to {DATA_DIR}")
    return True

if __name__ == "__main__":
    download_lavdf()
