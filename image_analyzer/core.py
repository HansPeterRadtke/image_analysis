print("[DEBUG] (image_analyzer:core) started")

import time
import json
t1 = time.time()
d = json.load(open("base_categories.json"))
print(d)
l_categories = []
d_sub2main   = {}
for k, l in d.items():
  l_categories.append(k)
  d_sub2main[k] = k
  for a in l:
    l_categories.append(a)
    d_sub2main[a] = k

print(l_categories)
print(len(l_categories))
print(d_sub2main)
print(len(d_sub2main))

from transformers import CLIPProcessor, CLIPModel
print("[DEBUG] (image_analyzer:core) importing of transformers took %6.2f sec; " % (time.time() - t1), flush = True)
from PIL import Image
import sys
import os


model     = None
processor = None

# prevent Hugging Face from using internet or cache
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_CACHE"  ] = "/dev/null"
os.environ["HF_HOME"       ] = "/dev/null"

print("[DEBUG] (image_analyzer:core) DONE")

def load_model():
  global model, processor
  print("[DEBUG] load_model called", flush=True)
  try:
    start_time = time.time()
    model_path = "/home/hans/dev/GPT/models/clip-vit-base-patch32"
    print(f"[DEBUG] loading processor and model from {model_path}", flush=True)
    processor  = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
    model      = CLIPModel    .from_pretrained(model_path, local_files_only=True)
    print(f"[DEBUG] model and processor loaded in {time.time() - start_time:.2f} seconds", flush=True)
  except Exception as e:
    print("[ERROR] loading model or processor:", str(e), flush=True)

def classify_image(image_path):
  print(f"[DEBUG] classify_image called with path: {image_path}", flush=True)
  try:
    if (not os.path.exists(image_path)):
      print("[ERROR] image file not found", flush=True)
      return []

    start_time = time.time()
    image      = Image.open(image_path)
    print("[DEBUG] image loaded", flush=True)

    inputs           = processor(text = l_categories, images=image, return_tensors="pt", padding=True)
    outputs          = model    (**inputs)
    logits_per_image = outputs.logits_per_image
    probs            = logits_per_image.softmax(dim=1)[0]

    d_prob_sums = {}
    for i in range(len(probs)):
      k = d_sub2main[l_categories[i]]
      if(k not in d_prob_sums):
        d_prob_sums[k] = 0
      d_prob_sums[k] += probs[i].item()
    d_prob_sums = dict(sorted(d_prob_sums.items(), key = lambda x: x[1], reverse = True))
    for k, v in d_prob_sums.items():
      print("'%-20s' = %6.3f; " % (k, v))

    print(f"[DEBUG] classification time: {time.time() - start_time:.2f} seconds", flush=True)
    return d_prob_sums
  except Exception as e:
    print("[ERROR] classification failed:", str(e), flush=True)
    return []

def classify_folder(folder_path):
  print(f"[DEBUG] classify_folder called with path: {folder_path}", flush=True)
  try:
    for filename in os.listdir(folder_path):
      if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
        full_path = os.path.join(folder_path, filename)
        d_probs   = classify_image(full_path)
  except Exception as e:
    print("[ERROR] folder classification failed:", str(e), flush=True)


if (__name__ == '__main__'):
  if (len(sys.argv) != 2):
    print("Usage: python3 -m image_analyzer.core <image_path>", flush=True)
    sys.exit(1)
  load_model()
  classify_image(sys.argv[1])
