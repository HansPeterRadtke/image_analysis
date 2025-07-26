from image_analyzer import core as image_analyzer
import time

print("[DEBUG] test script starting")

start_time = time.time()
image_analyzer.load_model()
print(f"[DEBUG] model loaded in {time.time() - start_time:.2f} seconds")

start_time = time.time()
image_analyzer.classify_folder("/var/www/html/images")
print(f"[DEBUG] folder classified in {time.time() - start_time:.2f} seconds")

print("[DEBUG] test script finished")