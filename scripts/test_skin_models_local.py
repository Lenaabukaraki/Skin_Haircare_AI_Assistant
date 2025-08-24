#!/usr/bin/env python3
import sys
import os
from typing import List
from transformers import pipeline
import torch


# Default local models (image-classification)
DEFAULT_MODELS: List[str] = [
"abdlh/ResNet34_finetuned_for_skin_diseases_by-abdlh",
"WahajRaza/finetuned-dermnet",
"Eraly-ml/Skin-AI",
]


def main():
if len(sys.argv) < 2:
print("Usage: python test_skin_models_local.py path/to/image.jpg [model_id]")
sys.exit(1)


image_path = sys.argv[1]
if not os.path.isfile(image_path):
print(f"❌ File not found: {image_path}")
sys.exit(3)


models = DEFAULT_MODELS
if len(sys.argv) == 3:
models = [sys.argv[2]]


print(f"\n📸 Testing local skin-condition models on {image_path}\n")
device = 0 if torch.cuda.is_available() else -1


for model_id in models:
print(f"── Model: {model_id}")
try:
classifier = pipeline("image-classification", model=model_id, device=device)
except Exception as e:
print(f" ❌ Error loading model: {e}\n")
continue


try:
results = classifier(image_path, top_k=3)
except Exception as e:
print(f" ❌ Inference failed: {e}\n")
continue


for i, res in enumerate(results, start=1):
label = res.get("label", "<unknown>")
score = float(res.get("score", 0.0)) * 100
print(f" {i}. {label:<30} {score:5.2f}%")
print()


if __name__ == "__main__":
main()