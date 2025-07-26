import os
import clip
import torch

print("[DEBUG] starting CLIP model download")

try:
  model_name = "ViT-B/32"
  model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models"))

  model, preprocess = clip.load(model_name, device="cpu", download_root=model_path)

  print("[DEBUG] CLIP model downloaded successfully")
  print(f"[DEBUG] saved to: {model_path}")
except Exception as e:
  print(f"[ERROR] {str(e)}")