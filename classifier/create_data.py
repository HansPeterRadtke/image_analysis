import os
import sys
import time
import json
import random
import traceback
import numpy     as     np
from   pathlib   import Path
from   PIL       import Image, ImageFilter

print("[DEBUG] Starting create_data.py", flush = True)

sys.path.append("../../image_generator")
import make_ai_image

folder_output = "data"
folder_web    = "/var/www/html/images"
os.makedirs(folder_output, exist_ok = True)
d_categories = json.load(open("../data/base_categories.json"))

for category, synonymes in d_categories.items():
  category_dir = os.path.join(folder_output, category)
  os.makedirs(category_dir, exist_ok = True)
  for i in range(10):
    filename  = f"{category}_{i:03d}.jpg"
    save_path = os.path.join(category_dir, filename)
    if(not os.path.exists(save_path)):
      print(f"[DEBUG] Generating {filename} in category {category}", flush=True)
      try:
        if(category == "empty"):
          width, height = 256, 256
          x             = np.linspace(0, 1, width)
          y             = np.linspace(0, 1, height)
          xv, yv        = np.meshgrid(x, y)
          angle         = random.uniform(0, 2 * np.pi)
          freq          = random.uniform(0.1, 2.0) * np.pi
          rotated       = np.cos(np.cos(angle) * xv * freq + np.sin(angle) * yv * freq)
          base_pattern  = rotated * random.uniform(30, 100)
          if random.random() < 0.8:
            angle2        = random.uniform(0, 2 * np.pi)
            freq2         = random.uniform(4.0, 8.0) * np.pi
            overlay       = np.sin(np.cos(angle2) * xv * freq2 + np.sin(angle2) * yv * freq2)
            base_pattern += overlay * random.uniform(10, 50)
          if(random.random() < 0.5):
            base_pattern += np.random.rand(height, width) * 255
          array = np.clip(base_pattern, 0, 255)
          image = Image.fromarray(array.astype(np.uint8)).convert("RGB")
          if(random.random() < 0.5):
            image = image.filter(ImageFilter.GaussianBlur(radius=3))
          image.save(save_path)
        else:
          print("[DEBUG] Sleeping 5 seconds", flush = True)
          time.sleep(5)
          prompt = category#random.choice(synonymes)
          print(f"[DEBUG] Calling make_ai_image.make_image with prompt: {prompt}", flush = True)
          ret = make_ai_image.make_image(prompt = prompt, output = save_path)
          if(ret is None):
            print("[ERROR] There was a problem while generating image!")
            continue
          print("[DEBUG] Image generation call finished", flush = True)
      except Exception:
        print("[ERROR] Failed to generate image", flush = True)
        print(traceback.format_exc(), flush = True)
    else:
      print(f"[DEBUG] Skipping existing file: {filename}", flush=True)
    cmd = ("cp %s %s" % (save_path, (folder_web + "/" + filename)))
    print("%s" % (cmd), flush = True)
    os.system(cmd)

print("[DEBUG] Finished create_data.py", flush = True)


