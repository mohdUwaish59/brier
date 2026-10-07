#!/usr/bin/env bash
# One-time setup. Run on a LOGIN node (it needs internet), from the repository root:
#     bash hpc/setup.sh
# Checks out the M7.2 branch, creates a virtual environment, installs the checkout in editable
# mode with test tools, runs the unit tests (CPU), and downloads the models and BANKING77 into
# WORKDIR so the GPU jobs can run offline.
set -euo pipefail
source "$(dirname "$0")/config.sh"

mkdir -p "$WORKDIR" "$HF_HOME" "$BRIER_CACHE_DIR" "$RESULTS"
echo "== repo: $REPO_DIR   workdir: $WORKDIR"

cd "$REPO_DIR"
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "!! $REPO_DIR has uncommitted changes to tracked files; commit or stash them first" >&2
  exit 1
fi
git fetch -q origin
git checkout -q "$BRIER_BRANCH"
git pull -q --ff-only origin "$BRIER_BRANCH"
echo "== branch $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"

if [ ! -x "$VENV/bin/python" ]; then
  echo "== creating virtual environment with $($PYTHON --version)"
  "$PYTHON" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -q --upgrade pip
if [ -n "$TORCH_INDEX_URL" ]; then
  echo "== installing torch from $TORCH_INDEX_URL"
  python -m pip install -q torch --index-url "$TORCH_INDEX_URL"
fi
echo "== installing the checkout (editable) with test tools"
python -m pip install -q -e "$REPO_DIR[hf]" pytest pytest-cov hypothesis
python -c "import brier, torch, transformers; print('brier', brier.__version__, '| torch', torch.__version__, '| transformers', transformers.__version__)"

echo "== unit tests (CPU, no downloads)"
python -m pytest -q -p no:cacheprovider

echo "== downloading BANKING77 (pinned, SHA-256 verified)"
python -c "from brier.bench.tasks import fetch_banking77; print(len(fetch_banking77()), 'bytes')"

echo "== downloading models (pinned revisions)"
download() {
  python - "$1" "$2" <<'PY'
import sys
from huggingface_hub import snapshot_download
path = snapshot_download(
    sys.argv[1],
    revision=sys.argv[2],
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "*.tiktoken"],
)
print("     ->", path)
PY
}
read -r tmodel trev <<<"$TEST_MODEL"
echo "   test model: $tmodel @ ${trev:0:7}"
download "$tmodel" "$trev"
for entry in "${MODELS[@]}"; do
  read -r key model revision batch <<<"$entry"
  echo "   $key: $model @ ${revision:0:7}"
  download "$model" "$revision"
done
echo "== setup done. Next: sbatch hpc/gpu_tests.slurm   then   sbatch hpc/run_rotations.slurm"
