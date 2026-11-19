import argparse
import json
from pathlib import Path

from utils.data import load_split, loader
from utils.runs import load_e2e_model
from utils.training import evaluate, predict_test


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--beam", type=int, default=8)
    p.add_argument("--set", action="append", default=[], help="override a decode arg, e.g. ctc_weight_decode=0.5")
    p.add_argument("--splits", default="dev,test")
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    run = Path(a.run)
    model, vocab, args = load_e2e_model(run, overrides=a.set)
    out = run / (f"decode_beam{a.beam}" + "".join(f"_{kv}" for kv in a.set))
    out.mkdir(exist_ok=True)
    if "dev" in a.splits:
        m, _ = evaluate(model, loader(load_split(args.data_dir, "dev"), vocab, 160, False), vocab, a.beam, args.device)
        print("DEV", m, flush=True)
        json.dump(m, open(out / "dev.json", "w"), indent=2)
    if "test" in a.splits:
        predict_test(model, load_split(args.data_dir, "test"), vocab, a.beam, args.device,
                     out / "test_predictions.csv")
    print("DONE", out)


if __name__ == "__main__":
    main()
