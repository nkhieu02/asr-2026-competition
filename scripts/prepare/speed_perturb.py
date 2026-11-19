import argparse
import csv
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--factors", default="0.9,1.1")
    p.add_argument("--jobs", type=int, default=2, help="parallel sox processes (the GPU box has a ~6.5-core quota)")
    p.add_argument("--split", default="train", choices=["train", "dev"])
    return p


def perturb(job):
    src, dst, f = job
    if not dst.exists():
        subprocess.run(["nice", "-n", "15", "sox", str(src), "-b", "16", str(dst), "speed", f, "rate", "16000"],
                       check=True, capture_output=True)


def main(argv=None):
    a = parser().parse_args(argv)
    root = Path(a.data_dir)
    rows = list(csv.DictReader(open(root / "geo" / f"{a.split}.csv", encoding="utf-8")))
    (root / "geo" / "clips_sp").mkdir(exist_ok=True)
    jobs, out_rows = [], list(rows)
    for f in a.factors.split(","):
        for r in rows:
            rel = f"geo/clips_sp/{Path(r['file']).stem}_sp{f}.wav"
            out_rows.append({"file": rel, "transcript": r["transcript"]})
            jobs.append((root / r["file"], root / rel, f))
    with ThreadPoolExecutor(a.jobs) as ex:
        for i, _ in enumerate(ex.map(perturb, jobs), 1):
            if i % 2000 == 0:
                print(f"{i}/{len(jobs)}", flush=True)
    with open(root / "geo" / f"{a.split}_sp.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "transcript"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"DONE: {len(out_rows)} rows in {a.split}_sp.csv")


if __name__ == "__main__":
    main()
