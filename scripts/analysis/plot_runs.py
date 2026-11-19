import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    runs = Path("runs")
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    rows = []
    for d in sorted(p for p in runs.iterdir() if (p / "metrics.jsonl").exists() and not p.name.startswith("smoke")):
        recs = [json.loads(l) for l in open(d / "metrics.jsonl")]
        tr = [(r["step"], r["loss"]) for r in recs if "loss" in r and "eval_loss" not in r]
        ev = {r["step"]: r for r in recs if "eval_wer" in r}
        ev = [ev[s] for s in sorted(ev)]
        if tr:
            s, l = zip(*tr)
            axes[0].plot(s, l, label=d.name, lw=1)
        if ev:
            axes[1].plot([r["step"] for r in ev], [r["eval_loss"] for r in ev], "o-", label=d.name, ms=3)
            axes[2].plot([r["step"] for r in ev], [100 * r["eval_wer"] for r in ev], "o-", label=d.name, ms=3)
            best = min(ev, key=lambda r: r["eval_wer"])
            lm = ""
            for f in sorted(d.glob("lm*/lm_results.json")):
                b = json.load(open(f))["best_lm"]
                lm += f"  {f.parent.name}: WER {100*b['wer']:.2f} CER {100*b['cer']:.2f} (a={b['alpha']},b={b['beta']})"
            rows.append(f"{d.name:28s} step {best['step']:>6}  WER {100*best['eval_wer']:.2f}  CER {100*best['eval_cer']:.2f}{lm}")

    axes[0].set(title="train loss", xlabel="step", yscale="log")
    axes[1].set(title="dev loss", xlabel="step")
    axes[2].set(title="dev WER % (greedy)", xlabel="step")
    for a in axes:
        a.grid(alpha=0.3)
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(runs / "loss_curves.png", dpi=110)
    print("\n".join(rows))


if __name__ == "__main__":
    main()
