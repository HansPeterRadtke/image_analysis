import os
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

def image_analyzer():
  print("[DEBUG] image_analyzer called")

  try:
    model_path = "/home/hans/dev/GPT/models"
    print("[DEBUG] model path:", model_path)
  except Exception as e:
    print("[ERROR] resolving model path:", str(e))
    return

  try:
    model_file = os.path.join(model_path, "clip_vit_base_patch32.bin")
    print("[DEBUG] model file:", model_file)
  except Exception as e:
    print("[ERROR] resolving model file path:", str(e))
    return

  try:
    image_path = "/var/www/html/images/test_make_ai_image_cli.jpg"
    print("[DEBUG] image path:", image_path)
  except Exception as e:
    print("[ERROR] resolving image path:", str(e))
    return

  try:
    if not os.path.exists(model_file):
      print("[ERROR] model file not found")
      return
  except Exception as e:
    print("[ERROR] checking model file existence:", str(e))
    return

  try:
    if not os.path.exists(image_path):
      print("[ERROR] image file not found")
      return
  except Exception as e:
    print("[ERROR] checking image file existence:", str(e))
    return

  try:
    image = Image.open(image_path)
    print("[DEBUG] image loaded")
  except Exception as e:
    print("[ERROR] loading image:", str(e))
    return

  try:
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    print("[DEBUG] model and processor loaded")
  except Exception as e:
    print("[ERROR] loading model or processor:", str(e))
    return

  try:
    text_inputs = ["a photo of a cat", "a photo of a dog", "a meme", "text over image"]
    print("[DEBUG] text inputs:", text_inputs)
  except Exception as e:
    print("[ERROR] preparing text inputs:", str(e))
    return

  try:
    inputs = processor(text=text_inputs, images=image, return_tensors="pt", padding=True)
    print("[DEBUG] inputs processed")
  except Exception as e:
    print("[ERROR] processing inputs:", str(e))
    return

  try:
    outputs = model(**inputs)
    print("[DEBUG] model inference complete")
  except Exception as e:
    print("[ERROR] model inference failed:", str(e))
    return

  try:
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    print("[DEBUG] output probabilities:", probs)
  except Exception as e:
    print("[ERROR] processing output probabilities:", str(e))
    return

  try:
    max_index = probs.argmax().item()
    print(f"[DEBUG] most likely label: {text_inputs[max_index]} (confidence: {probs.max().item():.4f})")
  except Exception as e:
    print("[ERROR] extracting prediction:", str(e))
    return

  print("[DEBUG] image_analyzer finished")