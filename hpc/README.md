# Testing brier and running M7.2 on a SLURM cluster

Uses your brier repository checkout on the cluster (branch `task/M7.2-rotations`). Four rungs, cheapest first:

| Step | Where | Time | What it proves |
|---|---|---|---|
| 1. `bash hpc/setup.sh` | login node | ~5 min + downloads | install works; **unit tests** pass (CPU) |
| 2. `sbatch hpc/gpu_tests.sbatch` | GPU node | ~15 min | `brier check` on both models, **integration tests** on the GPU, a 20-item rotations run |
| 3. `sbatch hpc/run_rotations.sbatch` | GPU node | ~15 min/model (A100), 1–2.5 h/model (T4/V100) | the M7.2 results |
| 4. `python hpc/summarize.py ~/brier_m72/results_m72` | anywhere | seconds | prints the tables |

## Before you start

1. **Get this folder:** it is on the `task/M7.2-rotations` branch.
   ```bash
   cd ~/brier                                # your checkout
   git fetch origin && git checkout task/M7.2-rotations && git pull
   ```
2. **Edit `hpc/config.sh`** if needed: `WORKDIR` (~10 GB free), `PYTHON` (3.10–3.13), and
   `TORCH_INDEX_URL` if PyTorch can't see the GPU with the default build.
3. **Edit the `#SBATCH` lines** in `gpu_tests.sbatch` and `run_rotations.sbatch`
   (`--partition`, `--account`), and add `module load` lines in `common.sh` if compute nodes
   need them.

## Run (from the repository root, e.g. `cd ~/brier`)

```bash
bash hpc/setup.sh                      # login node: branch, venv, unit tests, downloads
sbatch hpc/gpu_tests.sbatch            # GPU: checks, integration tests, smoke run
tail -f brier-gpu-tests-*.out          # ends with a PASS/FAIL summary
sbatch hpc/run_rotations.sbatch        # GPU: the full M7.2 run
```

`setup.sh` checks out `task/M7.2-rotations` and refuses to run if tracked files have
uncommitted changes. Jobs run offline (`HF_HUB_OFFLINE=1`); everything is downloaded by
`setup.sh`. Precision is picked automatically: bfloat16 where supported, else float32.

## Send back

- the summary at the end of `brier-gpu-tests-<job>.out`;
- the two `*_rotations.json` files from `$WORKDIR/results_m72/` (no text inside).
