import argparse
import itertools
import json
from pathlib import Path

import jiwer
import torch

from utils.data import load_split, loader, to_device, write_predictions
from utils.lm import lm_bin, lm_logprob, load_kenlm
from utils.device import autocast
from utils.runs import load_e2e_model, load_run_args


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--lm_dir", required=True, help="dir with train.bin / train_dev.bin (from scripts/decode/ctc_lm.py)")
    p.add_argument("--beam", type=int, default=16)
    p.add_argument("--lams", default="0.1,0.3,0.5")
    p.add_argument("--alphas", default="0,0.1,0.2,0.3,0.5,0.7,1.0")
    p.add_argument("--betas", default="0,0.5,1,2,3,4")
    return p


def compute_nbest(run, splits, beam):
    """{split: {utt index (str): [(text, model score, ctc score), ...]}}"""
    model, vocab, args = load_e2e_model(run)
    nbest = {}
    for split, df in splits.items():
        out = {}
        with torch.no_grad():
            for batch in loader(df, vocab, 160, False):
                batch = to_device(batch, args.device)
                with autocast(args.device):
                    for i, hyps in zip(batch["idx"], model.nbest_batch(batch, beam)):
                        out[i] = [(vocab.decode(t), s_model, s_ctc) for t, s_model, s_ctc in hyps]
        nbest[split] = {str(k): v for k, v in out.items()}
        print(f"{split}: n-best for {len(out)} utterances", flush=True)
    return nbest


def lm_scores(nbest, lm_path):
    lm = load_kenlm(lm_path)
    return {u: [lm_logprob(lm, t) for t, _, _ in hs] for u, hs in nbest.items()}


def rescore(nbest, lms, lam, alpha, beta):
    out = {}
    for u, hs in nbest.items():
        sc = [(1 - lam) * s_model + lam * s_ctc + alpha * l + beta * len(t.split())
              for (t, s_model, s_ctc), l in zip(hs, lms[u])]
        out[int(u)] = hs[max(range(len(hs)), key=sc.__getitem__)][0]
    return [out[i] for i in range(len(out))]


def main(argv=None):
    a = parser().parse_args(argv)
    run = Path(a.run)
    data_dir = load_run_args(run).data_dir
    dev_df, test_df = load_split(data_dir, "dev"), load_split(data_dir, "test")
    cache = run / f"nbest_beam{a.beam}.json"
    if not cache.exists():
        json.dump(compute_nbest(run, {"dev": dev_df, "test": test_df.assign(transcript="")}, a.beam),
                  open(cache, "w"))
    nbest = json.load(open(cache))

    refs = dev_df["transcript"].tolist()
    lm_dev = lm_scores(nbest["dev"], lm_bin(a.lm_dir, "dev"))
    grid = []
    for lam, alpha, beta in itertools.product(*(map(float, s.split(",")) for s in (a.lams, a.alphas, a.betas))):
        hyp = rescore(nbest["dev"], lm_dev, lam, alpha, beta)
        grid.append({"lam": lam, "alpha": alpha, "beta": beta, "wer": jiwer.wer(refs, hyp), "cer": jiwer.cer(refs, hyp)})
    best = min(grid, key=lambda g: g["wer"])
    no_lm = min(({"lam": lam, "wer": jiwer.wer(refs, h), "cer": jiwer.cer(refs, h)}  # baseline: same n-best, no LM
                 for lam in map(float, a.lams.split(",")) for h in [rescore(nbest["dev"], lm_dev, lam, 0.0, 0.0)]),
                key=lambda g: g["wer"])
    print("no LM (best λ)", no_lm)
    print("BEST", best)
    out = run / f"lm_rescore_beam{a.beam}"
    out.mkdir(exist_ok=True)
    json.dump({"best": best, "no_lm": no_lm, "grid": grid}, open(out / "dev_grid.json", "w"), indent=2)

    lm_test = lm_scores(nbest["test"], lm_bin(a.lm_dir, "test"))
    test_df["transcript"] = rescore(nbest["test"], lm_test, best["lam"], best["alpha"], best["beta"])
    write_predictions(test_df, out / "test_predictions.csv")
    print("DONE", out)


if __name__ == "__main__":
    main()
