print("[DEBUG] test script started", flush=True)

import time
t1 = time.time()

import image_analyzer

t2 = time.time()
td = (t2 - t1)
print("[DEBUG] importing took %6.2f sec" % (td), flush=True)

start_time = time.time()
image_analyzer.load_model()
print(f"[DEBUG] model loaded in {time.time() - start_time:.2f} seconds", flush=True)

start_time  = time.time()
folder_path = "/var/www/html/images"
#folder_path = "/var/www/html/explorer/upload"
image_analyzer.classify_folder(folder_path)
print(f"[DEBUG] folder classified in {time.time() - start_time:.2f} seconds", flush=True)

print("[DEBUG] test script finished", flush=True)

