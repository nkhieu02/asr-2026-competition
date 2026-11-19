"""Entry points (python -m scripts.<stage>.<name>). Hugging Face libraries run offline by default, from the local
cache (~/.cache/huggingface); set HF_HUB_OFFLINE=0 to allow downloads, e.g. for a model not cached yet."""
import os

for _k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "HF_EVALUATE_OFFLINE"):
    os.environ.setdefault(_k, os.environ.get("HF_HUB_OFFLINE", "1"))
