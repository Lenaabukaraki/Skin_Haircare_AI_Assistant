#!/usr/bin/env python3
import sys
import os
from huggingface_hub import InferenceClient


# ------------------------------------------------------------
# Auth: Hugging Face token via environment only
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
if not HF_API_TOKEN or not HF_API_TOKEN.startswith("hf_"):
print("ERROR: Please set HF_API_TOKEN in your environment (export/setx).")
sys.exit(2)


client = InferenceClient(api_key=HF_API_TOKEN)


# ------------------------------------------------------------
# Shortlist of inference‑ready skin lesion classifiers on HF (remote)
MODELS = [
"actavkid/vit-large-patch32-384-finetuned-skin-lesion-classification",
"Atoany/convnext-tiny-fituned-skin-lesions",
"nateraw/skin-lesion-mobile-v2-classifier",
]


def usage_exit():
print(f"Usage: python {os.path.basename(__file__)} path/to/image.jpg")
sys.exit(1)


if len(sys.argv) != 2:
usage_exit()


image_path = sys.argv[1]
if not os.path.isfile(image_path):
print(f"ERROR: file not found: {image_path}")
sys.exit(3)


with open(image_path, "rb") as f:
img_bytes = f.read()


print(f"\n▶️ Classifying {image_path!r} with {len(MODELS)} remote model(s):\n")


for model_id in MODELS:
print(f"── Model: {model_id}")
try:
preds = client.image_classification(model=model_id, image=img_bytes)
if not preds:
print(" ⚠️ No predictions returned.")
continue
for idx, c in enumerate(preds[:3], 1):
label = c.get("label") or c.get("class_name") or "<unknown>"
score = float(c.get("score", 0.0))
print(f" {idx}. {label}: {score:.2%}")
except Exception as e:
print(f" ❌ inference failed: {e}")


print("\nDone.\n")