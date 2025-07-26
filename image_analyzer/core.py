import os
import time
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import sys

model     = None
processor = None

with open("/home/hans/dev/GPT/github/image_analysis/azure_categories.txt", "r") as f:
  TEXT_CATEGORIES = [f"a {line.strip()}" for line in f if line.strip()]
#with open("/home/hans/dev/GPT/github/image_analysis/openimages_full_categories.txt", "r") as f:
#  TEXT_CATEGORIES = [f"a {line.strip()}" for line in f if line.strip()]

def load_model():
  global model, processor
  print("[DEBUG] load_model called")
  try:
    start_time = time.time()
    processor  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model      = CLIPModel    .from_pretrained("openai/clip-vit-base-patch32")
    print(f"[DEBUG] model and processor loaded in {time.time() - start_time:.2f} seconds")
  except Exception as e:
    print("[ERROR] loading model or processor:", str(e))

def classify_image(image_path):
  print(f"[DEBUG] classify_image called with path: {image_path}")
  try:
    if (not os.path.exists(image_path)):
      print("[ERROR] image file not found")
      return []

    start_time = time.time()
    image      = Image.open(image_path)
    print("[DEBUG] image loaded")

    inputs = processor(text=TEXT_CATEGORIES, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs            = logits_per_image.softmax(dim=1)[0]

    top_matches = sorted([(TEXT_CATEGORIES[i], probs[i].item()) for i in range(len(TEXT_CATEGORIES))], key=lambda x: x[1], reverse=True)[:3]

    for label, score in top_matches:
      print(f"[DEBUG] matched: {label} ({score:.4f})")
    print(f"[DEBUG] classification time: {time.time() - start_time:.2f} seconds")
    return top_matches
  except Exception as e:
    print("[ERROR] classification failed:", str(e))
    return []

def classify_folder(folder_path):
  print(f"[DEBUG] classify_folder called with path: {folder_path}")
  summary = []
  try:
    for filename in os.listdir(folder_path):
      if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
        full_path = os.path.join(folder_path, filename)
        matches   = classify_image(full_path)
        summary.append((filename, matches))
  except Exception as e:
    print("[ERROR] folder classification failed:", str(e))

  print("\n[DEBUG] FINAL SUMMARY:")
  for fname, matches in summary:
    match_str = ", ".join([f"{label} ({score:.2f})" for label, score in matches])
    print(f"{fname}: {match_str}")

if (__name__ == '__main__'):
  if (len(sys.argv) != 2):
    print("Usage: python3 -m image_analyzer.core <image_path>")
    sys.exit(1)
  load_model()
  classify_image(sys.argv[1])
