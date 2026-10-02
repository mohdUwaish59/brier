"""The README's Python examples run as written (FakeBackend stands in for HFBackend)."""

import re
import runpy
from pathlib import Path

import pytest

import brier.backends.hf as hf
from brier.backends.fake import FakeBackend

README = Path(__file__).resolve().parents[2] / "README.md"
BLOCKS = re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), re.S)


def test_readme_has_python_examples() -> None:
    assert len(BLOCKS) >= 2


def test_readme_examples_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeBackend(hidden_size=16, num_layers=8)
    monkeypatch.setattr(hf, "HFBackend", lambda *a, **k: fake)
    options = ["billing", "technical", "sales"]
    namespace: dict[str, object] = {
        "unlabelled_states": [f"unlabelled {i}" for i in range(20)],
        "labelled_states": [f"message {i}" for i in range(90)],
        "route_labels": [options[i % 3] for i in range(90)],
        "state": "where is my invoice",
    }
    monkeypatch.chdir(tmp_path)  # the examples save to a relative "calibration/" directory
    for i, block in enumerate(BLOCKS):  # later blocks use names from earlier ones
        script = tmp_path / f"readme_block_{i}.py"
        script.write_text(block, encoding="utf-8")
        namespace = runpy.run_path(str(script), init_globals=namespace)
    assert namespace["d"].decide("x", [namespace["route"]], level="L2")["route"].level == "L2"
