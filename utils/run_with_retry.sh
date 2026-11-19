#!/bin/bash
# Run a script on the GPU box with retries; train scripts resume from their latest checkpoint in runs/<run_name>.
# usage: utils/run_with_retry.sh <scripts.<stage>.<name> | scripts/<stage>/<name>.py> <run_name> [script args...]
#   e.g. utils/run_with_retry.sh scripts.train.ctc D_ctc_sp_8k --train_split train_sp
source /venv/main/bin/activate
cd /workspace/asr
module=${1%.py}; module=${module//\//.}; name=$2; shift 2
mkdir -p runs/$name
for attempt in 1 2 3; do
  echo "=== attempt $attempt $(date)" >> runs/$name/train.log
  python -m $module --run_name $name "$@" >> runs/$name/train.log 2>&1
  code=$?
  if [ $code -eq 0 ] && grep -q "^DONE" runs/$name/train.log; then echo "=== SUCCESS $(date)" >> runs/$name/train.log; exit 0; fi
  echo "=== FAILED attempt $attempt (exit $code) $(date)" >> runs/$name/train.log
  sleep 30
done
echo "=== GAVE UP" >> runs/$name/train.log
