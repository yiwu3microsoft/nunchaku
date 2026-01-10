
import os
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "WeiChow/CrispEdit-2M"
REPO_TYPE = "dataset"
REMOTE_DIR = "data"
LOCAL_DIR = "/home/aiscuser/data/CrispEdit-2M/data"
save_dir = "/tmp/output/47b53331-628a-411d-b96f-14f13b73a109_3c53764c/CrispEdit-2M/data"

os.environ["HF_HUB_ENABLE_XET"] = "0"
os.environ["HF_HUB_DISABLE_XET_WARNING"] = "1"

api = HfApi()
remote_files = [
    f for f in api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)
    if f.startswith(f"{REMOTE_DIR}/") and f.endswith(".parquet")
]

os.makedirs(LOCAL_DIR, exist_ok=True)
existing = set(os.listdir(save_dir))

for idx, rf in enumerate(remote_files):
    base = os.path.basename(rf)
    out_path = os.path.join(save_dir, base)

    if base in existing and os.path.getsize(out_path) > 0:
        # already present, skip
        continue

    print(f"Downloading {idx}/{len(remote_files)}, {rf} ...")

    # Download just this file
    hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=rf,                # e.g., "data/add_00042.parquet"
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False,
        force_download=False        # default; will reuse cache
    )

    print(f"Moving to {out_path} ...")
    # copy to save_dir
    os.rename(
        os.path.join(LOCAL_DIR, base),
        out_path
    )
