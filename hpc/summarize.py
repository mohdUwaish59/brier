"""Print the rotation tables from every *_rotations.json under a results directory."""

import glob
import json
import sys
from pathlib import Path


def fmt(v: list[float]) -> str:
    """Format a metric as value [low, high]."""
    return f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"


root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
files = sorted(glob.glob(str(root / "**" / "*_rotations.json"), recursive=True))
if not files:
    sys.exit(f"no *_rotations.json under {root}")
for path in files:
    res = json.loads(Path(path).read_text(encoding="utf-8"))
    print(
        f"\n{res['model']['id']} @ {res['model']['revision']}  dtype={res['model']['dtype']}"
        f"  n_test={res['n_test']}  brier {res['brier_version']} ({res['git_commit']})"
    )
    print("| rotations | accuracy | ece | nll | flip rate |")
    print("|---|---|---|---|---|")
    for m, e in sorted(res["rotation_curve"].items(), key=lambda kv: int(kv[0])):
        mt = e["metrics"]
        print(
            f"| {m} | {fmt(mt['accuracy'])} | {fmt(mt['ece'])} | {fmt(mt['nll'])} "
            f"| {fmt(mt['flip_rate'])} |"
        )
