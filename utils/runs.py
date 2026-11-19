import argparse
import json
from pathlib import Path

import torch

from utils.data import CharVocab
from utils.device import resolve_device


def load_run_args(run_dir):
    return argparse.Namespace(**json.load(open(Path(run_dir) / "args.json")))


def load_e2e_model(run_dir, device=None, overrides=()):
    """Rebuild an AED run from args.json + vocab.json and load best.pt.
    overrides: ["name=value", ...] applied to the saved args (e.g. decode settings).
    Returns (model in eval mode, vocab, args)."""
    run_dir = Path(run_dir)
    args = load_run_args(run_dir)
    for kv in overrides:
        k, v = kv.split("=")
        setattr(args, k, type(getattr(args, k))(v))
    device = resolve_device(device or args.device)
    args.device = device
    # best.pt holds every weight, so only the encoder's architecture (its config) is needed, not the hub weights.
    from transformers import Wav2Vec2Config
    args.encoder_config = Wav2Vec2Config.from_pretrained(args.encoder)
    vocab = CharVocab.from_json(run_dir / "vocab.json")
    from models.aed import AED
    model = AED(args, vocab).to(device).eval()
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    return model, vocab, args
