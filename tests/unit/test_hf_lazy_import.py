import subprocess
import sys


def test_importing_brier_and_hf_module_does_not_import_torch() -> None:
    # CLAUDE.md: torch/transformers are imported lazily, only inside backends/hf.py.
    code = (
        "import sys, brier, brier.backends.hf; "
        "bad = [m for m in ('torch', 'transformers') if m in sys.modules]; "
        "print(bad); sys.exit(1 if bad else 0)"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)  # noqa: S603
    assert proc.returncode == 0, proc.stdout + proc.stderr
