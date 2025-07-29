print("Started")
import os
import torch
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_CACHE"  ] = "/dev/null"
os.environ["HF_HOME"       ] = "/dev/null"
from PIL import Image
print("Importing ... ", end = "", flush = True)
from transformers import AutoImageProcessor, AutoModelForImageClassification
print("DONE", flush = True)

def load_model(model_path=None):
  print("[DEBUG] Loading DocumentFigureClassifier model", flush = True)
  model_path = model_path or "../../models/microsoft/layoutlmv3-base"
  print("Using '%s'" % (model_path), flush = True)
  processor  = AutoImageProcessor             .from_pretrained(model_path, local_files_only = True)
  print("Processor created", flush = True)
  model      = AutoModelForImageClassification.from_pretrained(model_path, local_files_only = True)
  print("Model created", flush = True)
  model.eval()
  print("[DEBUG] Model loaded", flush=True)
  return model, processor

def classify_image(model, processor, image_path):
  print(f"[DEBUG] Classifying image: {image_path}", flush=True)
  image = Image.open(image_path).convert("RGB")
  inputs = processor(images=image, return_tensors="pt")
  with torch.no_grad():
    outputs = model(**inputs)
  logits = outputs.logits
  predicted_class_idx = logits.argmax(-1).item()
  predicted_label = model.config.id2label[predicted_class_idx]
  print(f"[DEBUG] Prediction complete: {predicted_label}", flush=True)
  return predicted_label

def classify_folder(model, processor, folder_path):
  print(f"[DEBUG] Classifying folder: {folder_path}", flush=True)
  results = {}
  for fname in sorted(os.listdir(folder_path)):
    if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
      continue
    fpath = os.path.join(folder_path, fname)
    label = classify_image(model, processor, fpath)
    results[fname] = label
  return results

