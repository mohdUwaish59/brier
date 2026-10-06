#!/usr/bin/env bash
# Sourced by the SLURM scripts: environment for an offline GPU job.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

# Uncomment / adapt if your cluster needs modules on compute nodes:
# module load cuda/12.1
# module load python/3.11

# shellcheck disable=SC1091
source "$VENV/bin/activate"
export HF_HUB_OFFLINE=1          # everything was downloaded by setup.sh
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python - <<'PY'
import torch
assert torch.cuda.is_available(), "no CUDA device visible to PyTorch"
print("torch", torch.__version__, "| cuda", torch.version.cuda, "|", torch.cuda.get_device_name(0))
PY

# bfloat16 where the GPU supports it (A100, H100, L4, ...); float32 otherwise (V100, T4),
# because float16 can overflow on some models.
DTYPE="$(python -c 'import torch; print("bfloat16" if torch.cuda.is_bf16_supported() else "float32")')"
export DTYPE
echo "dtype: $DTYPE"
