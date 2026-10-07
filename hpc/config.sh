#!/usr/bin/env bash
# Shared settings for the M7.2 tests and run. Edit these for your cluster.

# The brier repository checkout on the cluster: by default the folder that contains hpc/.
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export REPO_DIR

# Where the virtual environment, model cache, data cache and results live.
# Use a filesystem with ~10 GB free (Qwen3-1.7B ~3.5 GB, Falcon3-1B-Base ~3.3 GB, Qwen3-0.6B ~1.5 GB).
export WORKDIR="${WORKDIR:-$HOME/brier_m72}"

# Python 3.10-3.13. On many clusters: `module load python/3.11` first, or set the full path.
export PYTHON="${PYTHON:-python3}"

# Branch (or tag) to test. setup.sh checks it out in REPO_DIR.
export BRIER_BRANCH="${BRIER_BRANCH:-main}"

# Optional: a PyTorch wheel index matching your cluster's CUDA driver, e.g.
#   https://download.pytorch.org/whl/cu121
# Leave empty to use PyPI's default torch build.
export TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

# Benchmark models: key, Hugging Face id, pinned revision, batch size.
MODELS=(
  "falcon3base tiiuae/Falcon3-1B-Base cb37ef3559b157b5c9d9226296ba01a5162da1f7 64"
  "qwen3 Qwen/Qwen3-1.7B 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e 64"
)
# The integration test suite's default model (tests/integration/test_hf_backend.py, "qwen3").
TEST_MODEL="Qwen/Qwen3-0.6B c1899de289a04d12100db370d81485cdf75e47ca"
export COUNTS="1,2,4,5,10,20"

# Caches inside WORKDIR, so jobs on compute nodes find everything offline.
export VENV="$WORKDIR/venv"
export HF_HOME="$WORKDIR/hf_home"
export BRIER_CACHE_DIR="$WORKDIR/brier_cache"
export RESULTS="$WORKDIR/results_m72"
