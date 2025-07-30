# Image Analysis Project

This project provides a toolkit for image classification using transformer models. It supports generating categorized datasets, running classification with pretrained models, and analyzing image folders. It uses models like CLIP and LayoutLMv3 for classification tasks.

---

## Project Structure
```
image_analysis/
├── classifier/
│   ├── classify.py         # Classify dataset using CLIP
│   ├── create_data.py      # Generate sample data
├── image_analyzer/
│   ├── __main__.py         # CLI entry point
│   ├── core.py             # Classification logic
│   ├── setup.py            # Install entry point
├── data/
│   ├── base_categories.json
│   ├── azure_categories.txt
├── tools/
│   ├── download_model.py   # (not yet used)
├── test.py                 # Manual test script
```

---

## 1. classifier/create_data.py
Generates image datasets per category.

### Usage:
```bash
python3 classifier/create_data.py
```

### Output:
- Saves 10 images per category to `classifier/data/<category>`
- Images are either AI-generated (via `make_ai_image`) or noise patterns
- Also copies each image to `/var/www/html/images`

---

## 2. classifier/classify.py
Classifies images by similarity using a local CLIP model.

### Usage:
```bash
python3 classifier/classify.py
```

### Description:
- Loads a CLIP model from `models/laion-clip-vit-b-32-laion2B`
- Picks one example image per category, computes features
- Classifies remaining images by comparing embedding distances

---

## 3. image_analyzer
Main classification module using LayoutLMv3.

### Installation:
```bash
cd image_analyzer
pip install .
```

### CLI usage:
```bash
image-analyzer
```

Or from project root:
```bash
python3 -m image_analyzer
```

### Import usage:
```python
from image_analyzer import load_model, classify_image, classify_folder

model, processor = load_model()
label = classify_image(model, processor, "/path/to/image.jpg")
```

### Folder classification:
```python
results = classify_folder(model, processor, "/var/www/html/images")
for fname, label in results.items():
  print(fname, "=>", label)
```

### Model Path:
- Defaults to: `../../models/microsoft/layoutlmv3-base`
- Modify in `load_model(model_path=...)` as needed

---

## 4. test.py
Manual test script to validate setup.

### Usage:
```bash
python3 test.py
```
- Loads the LayoutLMv3 model
- Classifies images in `/var/www/html/images`
- Prints predictions

---

## Notes
- Models are loaded locally (`local_files_only=True`)
- Ensure model directories exist before running
- Generated images are used both for training/testing and web serving

---

## License
MIT