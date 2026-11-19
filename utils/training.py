import argparse
import json
import math
import random
import time
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd
import torch

from utils.data import CharVocab, load_split, loader, to_device, write_predictions
from utils.device import autocast, default_device


def base_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", required=True)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--encoder", default="facebook/wav2vec2-xls-r-300m")
    p.add_argument("--enc_lr", type=float, default=5e-5)
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_steps", type=int, default=8000)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--freeze_enc_steps", type=int, default=0, help="train only the heads for the first N steps")
    p.add_argument("--max_batch_sec", type=float, default=120, help="dynamic batch size (seconds of audio)")
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--clip", type=float, default=5.0)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--log_steps", type=int, default=25)
    p.add_argument("--beam", type=int, default=8, help="beam for the final dev/test decode")
    p.add_argument("--mask_time_prob", type=float, default=0.05)
    p.add_argument("--mask_feature_prob", type=float, default=0.0)
    p.add_argument("--layerdrop", type=float, default=0.05)
    p.add_argument("--enc_dropout", type=float, default=0.05)
    p.add_argument("--train_on_dev", action="store_true")
    p.add_argument("--train_split", default="train", help="training CSV name, e.g. train_sp (speed-perturbed)")
    p.add_argument("--quick", action="store_true", help="tiny smoke test")
    p.add_argument("--device", default=default_device())
    p.add_argument("--seed", type=int, default=42)
    return p


class JsonlLog:
    def __init__(self, path):
        self.f, self.t0 = open(path, "a"), time.time()

    def __call__(self, **rec):
        rec = {"elapsed_min": round((time.time() - self.t0) / 60, 2), **rec}
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()
        print(rec, flush=True)


def evaluate(model, dl, vocab, beam, device, limit=None):
    """Loss, WER, CER over a loader; returns (metrics, {utterance index: hypothesis})."""
    model.eval()
    hyps, refs, losses = {}, {}, []
    with torch.no_grad():
        for n, batch in enumerate(dl):
            if limit and n >= limit:
                break
            batch = to_device(batch, device)
            with autocast(device):
                losses.append(float(model(batch)["loss"]))
                out = model.decode(batch, beam=beam)
            for b, (i, h) in enumerate(zip(batch["idx"], out)):
                hyps[i] = h
                refs[i] = vocab.decode(batch["y"][b, : batch["ylen"][b]].tolist())
    model.train()
    keys = sorted(hyps)
    R, H = [refs[k] for k in keys], [hyps[k] for k in keys]
    return {"loss": float(np.mean(losses)), "wer": jiwer.wer(R, H), "cer": jiwer.cer(R, H)}, hyps


def predict_test(model, test_df, vocab, beam, device, path):
    test_df = test_df.assign(transcript="")
    _, hyps = evaluate(model, loader(test_df, vocab, 160, False), vocab, beam=beam, device=device)
    test_df["transcript"] = [hyps[i] for i in range(len(test_df))]
    write_predictions(test_df, path)


def cosine_with_warmup(warmup, max_steps):
    def f(s):
        if s < warmup:
            return s / max(1, warmup)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, max_steps - warmup))))
    return f


def train(build_model, args):
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    out = Path(args.runs_dir) / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out / "args.json", "w"), indent=2)
    device = args.device

    train_df, dev_df, test_df = (load_split(args.data_dir, s) for s in (args.train_split, "dev", "test"))
    vocab = CharVocab(train_df["transcript"].tolist() + dev_df["transcript"].tolist())
    vocab.to_json(out / "vocab.json")
    if args.quick:
        train_df, dev_df, test_df = train_df.head(32), dev_df.head(8), test_df.head(4)
    tr_df = pd.concat([train_df, dev_df]) if args.train_on_dev else train_df

    train_dl = loader(tr_df, vocab, args.max_batch_sec, True, args.seed)
    dev_dl = loader(dev_df, vocab, 160, False)

    model = build_model(args, vocab).to(device)
    enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.") and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
    print(f"params: encoder {sum(p.numel() for p in enc_params)/1e6:.1f}M  heads {sum(p.numel() for p in head_params)/1e6:.1f}M")
    opt = torch.optim.AdamW([{"params": enc_params, "lr": args.enc_lr},
                             {"params": head_params, "lr": args.head_lr}], weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine_with_warmup(args.warmup, args.max_steps))

    step, best_wer = 0, float("inf")
    ck = out / "last.pt"
    if ck.exists():  # resume after a crash
        st = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        step, best_wer = st["step"], st["best_wer"]
        train_dl.batch_sampler.epoch = st.get("epoch", 0)
        print(f"resumed from step {step}")
    log = JsonlLog(out / "metrics.jsonl")

    model.train()
    acc, agg = 0, {}
    while step < args.max_steps:
        for batch in train_dl:
            freeze = step < args.freeze_enc_steps
            for p in enc_params:
                p.requires_grad_(not freeze)
            batch = to_device(batch, device)
            with autocast(device):
                res = model(batch)
            (res["loss"] / args.grad_accum).backward()
            for k, v in res.items():
                agg.setdefault(k, []).append(float(v.detach()))
            acc += 1
            if acc < args.grad_accum:
                continue
            acc = 0
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            if torch.isfinite(gn):
                opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            if step % args.log_steps == 0:
                log(step=step, epoch=train_dl.batch_sampler.epoch,
                    **{k: round(float(np.mean(v)), 4) for k, v in agg.items()},
                    grad_norm=round(float(gn), 3), enc_lr=sched.get_last_lr()[0], head_lr=sched.get_last_lr()[1])
                agg = {}
            if step % args.eval_steps == 0 or step == args.max_steps:
                m, _ = evaluate(model, dev_dl, vocab, beam=1, device=device)
                log(step=step, **{f"eval_{k}": v for k, v in m.items()})
                if m["wer"] < best_wer:
                    best_wer = m["wer"]
                    torch.save(model.state_dict(), out / "best.pt")
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "step": step, "best_wer": best_wer, "epoch": train_dl.batch_sampler.epoch}, ck)
            if step >= args.max_steps:
                break

    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    final = {}
    for beam in sorted({1, args.beam}):
        m, _ = evaluate(model, dev_dl, vocab, beam=beam, device=device)
        final[f"beam{beam}"] = m
        print(f"FINAL DEV beam={beam}", m, flush=True)
    json.dump(final, open(out / "final_dev.json", "w"), indent=2)
    predict_test(model, test_df, vocab, args.beam, device, out / "test_predictions.csv")
    print("DONE", flush=True)
