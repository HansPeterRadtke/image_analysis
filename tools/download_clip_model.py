import os
from huggingface_hub import snapshot_download

MODELS = [
  ("openai/clip-vit-base-patch32", "clip-vit-base-patch32"),
  ("google/siglip-base-patch16-224", "siglip-base-patch16-224"),
  ("laion/CLIP-ViT-B-32-laion2B-s34B-b79K", "laion-clip-vit-b-32-laion2B"),
  ("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "laion-clip-vit-h-14-laion2B"),
  ("ybelkada/clip-rsicd-vit-b16", "clip-rsicd-vit-b16")
]

if __name__ == '__main__':
  for model_id, folder_name in MODELS:
    try:
      target_dir = f"/home/hans/dev/GPT/models/{folder_name}"
      print(f"\n[DEBUG] Starting download of {model_id} to {target_dir}", flush=True)
      os.makedirs(target_dir, exist_ok=True)
      snapshot_download(
        repo_id=model_id,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        resume_download=True
      )
      print(f"[DEBUG] Download complete for {model_id}\n", flush=True)
    except Exception as e:
      print(f"[ERROR] Failed to download {model_id}: {e}\n", flush=True)