import torch


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_device(device):
    """A device saved in a run's args.json (e.g. "cuda") falls back to this machine's default if unavailable."""
    d = str(device or "")
    if not d or (d.startswith("cuda") and not torch.cuda.is_available()) or (d == "mps" and not torch.backends.mps.is_available()):
        return default_device()
    return d


def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda"))
