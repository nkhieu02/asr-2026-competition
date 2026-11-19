#!/bin/bash
# Pull logs/metrics/predictions (not checkpoints) from the GPU box, refresh the loss plot, print the summary.
# usage: utils/sync_runs.sh   (from anywhere)
set -e
cd "$(dirname "$0")/.."   # project root
HOST="-p 28383 root@151.237.25.16"
ssh -n $HOST 'mkdir -p /workspace/asr/scripts/analysis' 2>/dev/null
scp -q -P 28383 scripts/analysis/plot_runs.py root@151.237.25.16:/workspace/asr/scripts/analysis/ 2>/dev/null
ssh -n $HOST 'cd /workspace/asr && source /venv/main/bin/activate >/dev/null && python -m scripts.analysis.plot_runs' 2>/dev/null
mkdir -p runs
ssh -n $HOST 'cd /workspace/asr && tar cf - --exclude=ckpt --exclude=best --exclude=processor \
  --exclude="*.arpa" --exclude="*.bin" --exclude="*.txt" --exclude="*.pt" runs' 2>/dev/null | tar xf -
echo "plot: $(pwd)/runs/loss_curves.png"
