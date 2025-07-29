import os
from huggingface_hub import snapshot_download

MODEL_ID = FOLDER_NAME = "microsoft/layoutlmv3-base"

if __name__ == '__main__':
  try:
    target_dir = f"/home/hans/dev/GPT/models/{FOLDER_NAME}"
    print(f"\n[DEBUG] Starting download of {MODEL_ID} to {target_dir}", flush=True)
    os.makedirs(target_dir, exist_ok = True)
    ret = snapshot_download(
      repo_id         = MODEL_ID  ,
      local_dir       = target_dir,
      force_download  = True      ,
      resume_download = True
    )
    print(f"[DEBUG] Download complete for {MODEL_ID}\n", flush=True)
    print("Returned '%s'" % (ret), flush = True)
  except Exception as e:
    print(f"[ERROR] Failed to download {MODEL_ID}: {e}\n", flush=True)



