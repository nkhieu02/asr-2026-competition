import math
import os
import sys
import subprocess
from pathlib import Path

def _kenlm_bin():
    if os.environ.get("KENLM_BIN"):
        return os.environ["KENLM_BIN"]
    for d in (Path(sys.executable).parent, *(Path(x) for x in os.environ.get("PATH", "").split(os.pathsep) if x)):
        if (d / "lmplz").exists() and (d / "build_binary").exists():
            return str(d)
    return "/opt/kenlm/build/bin"


KENLM_BIN = _kenlm_bin()
LN10 = math.log(10)


def word_vocab(texts):
    """Sorted word list of the LM's training text (pyctcdecode unigrams / OOV test)."""
    return sorted({w for t in texts for w in t.split()})


def lm_bin(lm_dir, split):
    return str(Path(lm_dir) / ("train.bin" if split == "dev" else "train_dev.bin"))


def build_kenlm(texts, out_dir, name, order, fallback_dir=None):
    """Train <out_dir>/<name>.bin; returns (bin path, unigrams). Without KenLM binaries (e.g. a GPU box without the
    build), reuse <fallback_dir>/<name>.bin, an identical LM trained on the same text."""
    out_dir = Path(out_dir)
    txt, arpa, binf = out_dir / f"{name}.txt", out_dir / f"{name}.arpa", out_dir / f"{name}.bin"
    if not Path(f"{KENLM_BIN}/lmplz").exists():
        src = Path(fallback_dir) / f"{name}.bin" if fallback_dir else None
        assert src and src.exists(), f"no lmplz and no prebuilt {src}"
        print(f"lmplz not found: reusing {src}", flush=True)
        return str(src), word_vocab(texts)
    txt.write_text("\n".join(texts) + "\n")
    subprocess.run(f"{KENLM_BIN}/lmplz -o {order} --discount_fallback < {txt} > {arpa}",
                   shell=True, check=True, stderr=subprocess.DEVNULL)
    subprocess.run(f"{KENLM_BIN}/build_binary {arpa} {binf}", shell=True, check=True, stdout=subprocess.DEVNULL)
    return str(binf), word_vocab(texts)


def load_kenlm(path):
    import kenlm
    return kenlm.Model(str(path))


def lm_logprob(lm, text):
    """Natural-log sentence probability (KenLM returns log10)."""
    return lm.score(text, bos=True, eos=True) * LN10
