print("[DEBUG] test script started", flush=True)

import time
from image_analyzer import load_model, classify_folder

start_time = time.time()
model, processor = load_model()
print("[DEBUG] model loaded in %6.2f sec" % (time.time() - start_time), flush=True)

folder_path = "/var/www/html/images"
results = classify_folder(model, processor, folder_path)

for fname, label in results.items():
  print(f"[RESULT] {fname:20} => {label}", flush=True)

print("[DEBUG] test script finished", flush=True)