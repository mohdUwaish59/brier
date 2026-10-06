#!/bin/bash
#SBATCH --job-name=brier-gpu-tests
#SBATCH --partition=geospark-gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/ceph/home/moh77867/brier/brier/hpc/logs/gpu_test_%j.out
#SBATCH --error=/ceph/home/moh77867/brier/brier/hpc/logs/gpu_test_%j.err

set -uo pipefail   # no -e: run every step and report all results at the end
cd "${SLURM_SUBMIT_DIR:-.}"
source hpc/common.sh
cd "$REPO_DIR"
status=()

step() {  # step <name> <command...>
  local name="$1"; shift
  echo; echo "========== $name =========="
  if "$@"; then status+=("PASS  $name"); else status+=("FAIL  $name"); fi
}

for entry in "${MODELS[@]}"; do
  read -r key model revision batch <<<"$entry"
  step "brier check $key" brier check "$model" --revision "$revision" --dtype "$DTYPE"
done

# Integration suite on the GPU (fp32 by design); its default model is Qwen3-0.6B.
step "integration tests (qwen3-0.6B)" env BRIER_TEST_MODELS=qwen3 \
  python -m pytest -m integration -q -p no:cacheprovider tests/integration

read -r key model revision batch <<<"${MODELS[0]}"
step "rotations smoke ($key, 20 items)" python -m brier.bench rotations --model "$model" \
  --revision "$revision" --dtype "$DTYPE" --batch-size "$batch" --counts 1,4,20 --limit 20 \
  --n-resamples 100 --out "$RESULTS/smoke_$key"

echo; echo "========== summary =========="
printf '%s\n' "${status[@]}"
if printf '%s\n' "${status[@]}" | grep -q '^FAIL'; then exit 1; fi
echo "All GPU tests passed. Full run: sbatch hpc/run_rotations.slurm"
