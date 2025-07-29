import os
import traceback
import random
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

print("[DEBUG] Starting classify.py", flush=True)

try:
  model_path = "/home/hans/dev/GPT/models/laion-clip-vit-b-32-laion2B"
  model = CLIPModel.from_pretrained(model_path, local_files_only=True)
  processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = model.to(device)
  print("[DEBUG] Model loaded", flush=True)
except Exception:
  print("[ERROR] Model load failed", flush=True)
  print(traceback.format_exc(), flush=True)

categories = ["empty", "diagram", "drawing", "photo"]
base_path = Path("/home/hans/dev/GPT/github/image_analysis/classifier/data")

examples = {}
for category in categories:
  folder = base_path / category
  files = sorted([f for f in folder.glob("*.jpg")])
  if not files:
    print(f"[ERROR] No files in {category}", flush=True)
    continue
  example = random.choice(files)
  examples[category] = example
  print(f"[DEBUG] Example for {category}: {example.name}", flush=True)

def encode(path):
  try:
    image = Image.open(path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
      features = model.get_image_features(**inputs)
      features /= features.norm(dim=-1, keepdim=True)
    return features[0].cpu().numpy()
  except Exception:
    print(f"[ERROR] Encode failed: {path}", flush=True)
    print(traceback.format_exc(), flush=True)
    return None

example_features = {}
for category, path in examples.items():
  vec = encode(path)
  if vec is not None:
    example_features[category] = vec

results = []
for category in categories:
  folder = base_path / category
  test_files = sorted([f for f in folder.glob("*.jpg") if f != examples.get(category)])
  for f in test_files:
    vec = encode(f)
    if vec is None:
      continue
    distances = {c: np.linalg.norm(vec - ftr) for c, ftr in example_features.items()}
    predicted = min(distances, key=distances.get)
    results.append((f.name, category, predicted, distances))
    print(f"[DEBUG] {f.name}: actual={category}, predicted={predicted}", flush=True)

print("[DEBUG] Finished classify.py", flush=True)