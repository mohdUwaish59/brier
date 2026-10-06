"""M7.2: how many L0 rotations are needed (benchmark `rotations` command)."""

import json
from pathlib import Path

import numpy as np
import pytest

from brier.backends.fake import FakeBackend
from brier.bench.__main__ import main
from brier.bench.rotations import evenly_spaced, per_rotation, run_rotations, subset_probs
from brier.bench.run import run_task
from brier.bench.tasks import Task, make_banking20
from brier.errors import BrierError

BIASED = FakeBackend(position_bias=(2.0, 0.5, 0.0, -0.5))


def _task() -> Task:
    rows = [(f"intent_{c} message {i}", f"intent_{c}") for c in "abcd" for i in range(30)]
    return make_banking20(rows, top_k=4, n_pool=12, n_cal=8)


@pytest.mark.parametrize(
    ("k", "m", "want"),
    [(20, 1, [0]), (20, 4, [0, 5, 10, 15]), (20, 3, [0, 6, 13]), (4, 4, [0, 1, 2, 3])],
)
def test_evenly_spaced(k: int, m: int, want: list[int]) -> None:
    assert evenly_spaced(k, m) == want


@pytest.mark.parametrize(("k", "m"), [(4, 0), (4, 5), (4, -1)])
def test_evenly_spaced_rejects_bad_counts(k: int, m: int) -> None:
    with pytest.raises(BrierError):
        evenly_spaced(k, m)


def test_per_rotation_shape_and_normalisation() -> None:
    t = _task()
    rot = per_rotation(BIASED, list(t.test_states[:5]), t.question)
    assert rot.shape == (4, 5, 4)  # (rotation, item, option)
    np.testing.assert_allclose(np.exp(rot).sum(axis=-1), 1.0)


def test_all_rotations_reproduce_the_l0_benchmark(tmp_path: Path) -> None:
    # The full set of K rotations must give exactly what `run_task` reports for L0
    # (prior fitted on the pool, option order and reversed order).
    t = _task()
    run_task(BIASED, t, ["L0"], tmp_path / "run", n_resamples=10)
    with np.load(tmp_path / "run" / "banking20_fake_seed0.npz", allow_pickle=False) as z:
        want, want_rev = z["L0_probs"], z["L0_probs_reversed"]
    run_rotations(BIASED, t, tmp_path / "rot", ms=[4], n_resamples=10)
    with np.load(tmp_path / "rot" / "banking20_fake_seed0_rotations.npz", allow_pickle=False) as z:
        p = subset_probs(z["test_rot"], z["pool_rot"], [0, 1, 2, 3])
        p_rev = subset_probs(z["test_rot_reversed"], z["pool_rot_reversed"], [0, 1, 2, 3])
    np.testing.assert_allclose(p, want, atol=1e-12)
    np.testing.assert_allclose(p_rev[:, ::-1], want_rev, atol=1e-12)


def test_run_rotations_writes_the_curve(tmp_path: Path) -> None:
    s = run_rotations(BIASED, _task(), tmp_path, ms=[1, 2, 4], n_resamples=20)
    saved = json.loads((tmp_path / "banking20_fake_seed0_rotations.json").read_text())
    assert saved == s
    curve = s["rotation_curve"]
    assert set(curve) == {"1", "2", "4"}
    assert curve["2"]["shifts"] == [0, 2]
    assert curve["2"]["forward_passes"] == 2
    for entry in curve.values():
        for key in ("accuracy", "ece", "nll", "flip_rate"):
            v, lo, hi = entry["metrics"][key]
            assert lo <= v <= hi
    # with a strong position bias, more rotations flip less
    assert curve["4"]["metrics"]["flip_rate"][0] < curve["1"]["metrics"]["flip_rate"][0]
    assert s["n_test"] == len(_task().test_states)
    assert s["model"]["id"] == "fake"


def test_rotations_npz_has_no_text(tmp_path: Path) -> None:
    run_rotations(BIASED, _task(), tmp_path, ms=[1, 4], n_resamples=10)
    with np.load(tmp_path / "banking20_fake_seed0_rotations.npz", allow_pickle=False) as z:
        assert {"labels", "test_rot", "test_rot_reversed", "pool_rot", "pool_rot_reversed"} <= set(
            z.files
        )
        assert not any(z[k].dtype.kind in "OU" for k in z.files if k != "options")


def test_run_rotations_rejects_counts_above_k(tmp_path: Path) -> None:
    with pytest.raises(BrierError):
        run_rotations(BIASED, _task(), tmp_path, ms=[1, 5], n_resamples=10)


def test_cli_rotations_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from brier.bench import __main__ as cli

    monkeypatch.setattr(cli, "_load_task", lambda name, seed: _task())
    monkeypatch.setattr(cli, "_make_backend", lambda args: BIASED)
    argv = ["rotations", "--model", "fake", "--out", str(tmp_path), "--counts", "1,2,4"]
    assert main([*argv, "--n-resamples", "10"]) == 0
    assert (tmp_path / "banking20_fake_seed0_rotations.json").exists()
