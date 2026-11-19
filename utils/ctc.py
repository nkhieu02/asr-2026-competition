from multiprocessing import get_context

import torch

from utils.data import read_audio
from utils.device import autocast


def load_ctc(model_dir, device):
    from transformers import AutoModelForCTC, Wav2Vec2Processor
    proc = Wav2Vec2Processor.from_pretrained(model_dir)
    return proc, AutoModelForCTC.from_pretrained(model_dir).to(device).eval()


def ctc_log_probs(model, proc, paths, device):
    """List of (T, V) CPU log-prob tensors, one per audio file."""
    out = []
    with torch.no_grad():
        for path in paths:
            inp = proc(read_audio(path), sampling_rate=16_000, return_tensors="pt").to(device)
            with autocast(device):
                lg = model(**inp).logits[0].float()
            out.append(torch.log_softmax(lg, -1).cpu())
    return out


def greedy_decode(proc, log_probs):
    return proc.batch_decode([lp.argmax(-1) for lp in log_probs])


def pyctcdecode_labels(proc):
    """Tokenizer vocab in id order for pyctcdecode: "" = blank ([PAD]), " " = word delimiter, other multi-char
    specials -> unique dummy symbols."""
    vocab = proc.tokenizer.get_vocab()
    labels = [None] * len(vocab)
    for tok, i in vocab.items():
        labels[i] = tok
    pad = proc.tokenizer.pad_token_id
    return ["" if i == pad else " " if t == "|" else (f"␀{i}" if len(t) > 1 else t) for i, t in enumerate(labels)]


def lm_decoder(proc, lm_path, unigrams, alpha, beta):
    from pyctcdecode import build_ctcdecoder
    return build_ctcdecoder(pyctcdecode_labels(proc), kenlm_model_path=lm_path, unigrams=unigrams,
                            alpha=alpha, beta=beta)


def decode_text(decoder, log_probs, beam_width=100, processes=6):
    with get_context("fork").Pool(processes) as pool:
        return decoder.decode_batch(pool, [lp.numpy() for lp in log_probs], beam_width=beam_width)
