#!/usr/bin/env bash
# Full M7.2 run: every model in config.sh, all 2,603 banking20 test items, rotation counts
# 1,2,4,5,10,20. About 15 min per model on an A100; 1-2.5 h per model on a T4/V100 (float32).
# Models that already have results are skipped, so a resubmit continues where it stopped.
#     sbatch hpc/run_rotations.slurm
#SBATCH --job-name=brier-m72
#SBATCH --partition=geospark-gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/ceph/home/moh77867/brier/brier/hpc/logs/brier-m72-%j.out
#SBATCH --error=/ceph/home/moh77867/brier/brier/hpc/logs/brier-m72-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source hpc/common.sh

for entry in "${MODELS[@]}"; do
  read -r key model revision batch <<<"$entry"
  out="$RESULTS/$key"
  if compgen -G "$out/*_rotations.json" >/dev/null; then
    echo "== $key: results exist, skipping"
    continue
  fi
  echo "== $key: $model @ ${revision:0:7}  started $(date +%T)"
  python -m brier.bench rotations --model "$model" --revision "$revision" --dtype "$DTYPE" \
    --batch-size "$batch" --counts "$COUNTS" --out "$out"
  echo "== $key: finished $(date +%T)"
done

python hpc/summarize.py "$RESULTS"
echo "== done. Send: $RESULTS (the *_rotations.json files; the .npz files are optional)"
