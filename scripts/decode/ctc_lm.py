import argparse
import itertools
import json
from pathlib import Path

import jiwer
import torch

from utils.ctc import ctc_log_probs, decode_text, greedy_decode, lm_decoder, load_ctc
from utils.data import load_split, write_predictions
from utils.lm import build_kenlm
from utils.device import default_device, empty_cache


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True, help="<ctc run>/best")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--order", type=int, default=4)
    p.add_argument("--lm_from", default="runs/A_xlsr300m_lr3e-4/lm4",
                   help="if KenLM binaries are not installed, reuse <dir>/train.bin and <dir>/train_dev.bin")
    p.add_argument("--beam", type=int, default=100)
    p.add_argument("--alphas", default="0.3,0.4,0.5,0.6,0.8")
    p.add_argument("--betas", default="1.0,2.0,3.0,4.0,5.0")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    model_dir = Path(args.model_dir)
    out = model_dir.parent / f"lm{args.order}"
    out.mkdir(exist_ok=True)
    train, dev, test = (load_split(args.data_dir, s, durations=False) for s in ("train", "dev", "test"))
    refs = dev["transcript"].tolist()

    device = default_device()
    proc, model = load_ctc(model_dir, device)
    dev_lp = ctc_log_probs(model, proc, dev["path"], device)
    test_lp = ctc_log_probs(model, proc, test["path"], device)
    del model
    empty_cache()

    greedy = greedy_decode(proc, dev_lp)
    res = {"greedy_dev_wer": jiwer.wer(refs, greedy), "greedy_dev_cer": jiwer.cer(refs, greedy)}
    print("greedy", res, flush=True)

    lm_train, uni_train = build_kenlm(train["transcript"].tolist(), out, "train", args.order, args.lm_from)
    grid = []
    for a, b in itertools.product(map(float, args.alphas.split(",")), map(float, args.betas.split(","))):
        hyp = decode_text(lm_decoder(proc, lm_train, uni_train, a, b), dev_lp, args.beam)
        w, c = jiwer.wer(refs, hyp), jiwer.cer(refs, hyp)
        grid.append({"alpha": a, "beta": b, "wer": w, "cer": c})
        print(f"alpha={a} beta={b} WER={w:.4f} CER={c:.4f}", flush=True)
    best = min(grid, key=lambda g: g["wer"])
    res.update(best_lm=best, grid=grid)
    json.dump(res, open(out / "lm_results.json", "w"), indent=2)
    print("BEST", best, flush=True)

    lm_all, uni_all = build_kenlm(train["transcript"].tolist() + refs, out, "train_dev", args.order, args.lm_from)
    test["transcript"] = decode_text(lm_decoder(proc, lm_all, uni_all, best["alpha"], best["beta"]), test_lp, args.beam)
    write_predictions(test, out / "test_predictions_lm.csv")
    test["transcript"] = greedy_decode(proc, test_lp)
    write_predictions(test, out / "test_predictions_greedy.csv")
    print("DONE")


if __name__ == "__main__":
    main()
