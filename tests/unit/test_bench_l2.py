import json
from pathlib import Path

import numpy as np
import pytest

from brier.backends.fake import FakeBackend
from brier.bench.features import cached_hidden_states, stratified_budget
from brier.bench.run import check_levels, run_task
from brier.bench.tasks import Task, make_banking20
from brier.errors import BrierError

# ---------- feature cache ----------


class CountingBackend(FakeBackend):
    """FakeBackend that counts hidden-state calls (frozen dataclass: count in a list)."""

    calls: list[int]

    def __init__(self, **kw) -> None:  # type: ignore[no-untyped-def]
        super().__init__(**kw)
        object.__setattr__(self, "calls", [])

    def hidden_states(self, prompts, layers):  # type: ignore[no-untyped-def,override]
        self.calls.append(len(prompts))
        return super().hidden_states(prompts, layers)


PROMPTS = [f"prompt {i}" for i in range(7)]


def test_cache_hit_skips_the_backend_and_returns_identical_features(tmp_path: Path) -> None:
    b = CountingBackend(hidden_size=5, num_layers=6)
    first = cached_hidden_states(b, PROMPTS, [1, 3], tmp_path)
    second = cached_hidden_states(b, PROMPTS, [1, 3], tmp_path)
    assert b.calls == [7]
    np.testing.assert_array_equal(first, second)
    assert first.shape == (7, 2, 5)
    assert first.dtype == np.float32


@pytest.mark.parametrize(
    "change",
    [
        lambda: (
            CountingBackend(hidden_size=5, num_layers=6, seed=9),
            PROMPTS,
            [1, 3],
        ),  # test-only seed: not in the key
        lambda: (CountingBackend(hidden_size=5, num_layers=6, model_id="other"), PROMPTS, [1, 3]),
        lambda: (CountingBackend(hidden_size=5, num_layers=6), PROMPTS[:-1], [1, 3]),
        lambda: (CountingBackend(hidden_size=5, num_layers=6), PROMPTS, [1, 4]),
    ],
    ids=["same-key", "model", "prompts", "layers"],
)
def test_cache_key_covers_model_prompts_and_layers(tmp_path: Path, change) -> None:  # type: ignore[no-untyped-def]
    cached_hidden_states(CountingBackend(hidden_size=5, num_layers=6), PROMPTS, [1, 3], tmp_path)
    b, prompts, layers = change()
    cached_hidden_states(b, prompts, layers, tmp_path)
    expected_calls = (
        [] if (prompts is PROMPTS and layers == [1, 3] and b.model_id == "fake") else [len(prompts)]
    )
    assert b.calls == expected_calls


def test_cache_files_hold_no_prompt_text(tmp_path: Path) -> None:
    cached_hidden_states(
        FakeBackend(hidden_size=3, num_layers=4), ["secret customer text"], [0], tmp_path
    )
    for f in tmp_path.rglob("*"):
        if f.is_file():
            assert b"secret customer text" not in f.read_bytes()


def test_corrupt_cache_is_recomputed(tmp_path: Path) -> None:
    b = CountingBackend(hidden_size=5, num_layers=6)
    cached_hidden_states(b, PROMPTS, [1], tmp_path)
    for f in tmp_path.glob("*.npz"):
        f.write_bytes(b"garbage")
    out = cached_hidden_states(b, PROMPTS, [1], tmp_path)
    assert b.calls == [7, 7]
    assert out.shape == (7, 1, 5)


def test_no_cache_dir_always_computes() -> None:
    b = CountingBackend(hidden_size=5, num_layers=6)
    cached_hidden_states(b, PROMPTS, [1], None)
    cached_hidden_states(b, PROMPTS, [1], None)
    assert b.calls == [7, 7]


# ---------- stratified budget ----------


def test_stratified_budget_balances_classes_and_is_seeded() -> None:
    y = np.repeat(np.arange(4), [30, 20, 10, 8])
    idx = stratified_budget(y, 4, budget=24, seed=0)
    assert len(idx) == 24
    assert len(set(idx.tolist())) == 24
    np.testing.assert_array_equal(np.bincount(y[idx], minlength=4), [6, 6, 6, 6])
    np.testing.assert_array_equal(idx, stratified_budget(y, 4, budget=24, seed=0))
    assert not np.array_equal(idx, stratified_budget(y, 4, budget=24, seed=1))


def test_stratified_budget_spills_over_when_a_class_runs_out() -> None:
    y = np.repeat(np.arange(3), [30, 30, 4])
    idx = stratified_budget(y, 3, budget=30, seed=0)
    counts = np.bincount(y[idx], minlength=3)
    assert counts[2] == 4  # all of the small class
    assert counts.sum() == 30


def test_stratified_budget_validation() -> None:
    y = np.repeat(np.arange(2), 10)
    with pytest.raises(BrierError):
        stratified_budget(y, 2, budget=21, seed=0)
    with pytest.raises(BrierError):
        stratified_budget(y, 2, budget=0, seed=0)


# ---------- runner L2 ----------


def _task() -> Task:
    rows = [(f"intent_{c} message {i}", f"intent_{c}") for c in "abcd" for i in range(60)]
    return make_banking20(rows, top_k=4, n_pool=10, n_cal=80)  # 150 test items


def test_levels_accept_l2_alone_and_with_others() -> None:
    assert check_levels(["L2"]) == ["L2"]
    assert check_levels(["L2", "raw", "L0", "L1"]) == ["raw", "L0", "L1", "L2"]


def test_run_task_l2_curve(tmp_path: Path) -> None:
    b = FakeBackend(hidden_size=8, num_layers=10, position_bias=(1.0,))
    s = run_task(
        b,
        _task(),
        ["raw", "L0", "L1", "L2"],
        tmp_path / "out",
        n_resamples=20,
        l2_budgets=(60, 80),
        cache_dir=tmp_path / "cache",
    )
    assert set(s["l2_curve"]) == {"60", "80"}
    for entry in s["l2_curve"].values():
        assert entry["layer"] in (4, 6, 8)
        assert entry["solver"] in ("ridge", "lda")
        assert entry["temperature"] > 0
        assert set(entry["metrics"]) >= {"accuracy", "nll", "ece", "flip_rate"}
    assert s["levels"]["L2"] == s["l2_curve"]["80"]["metrics"]  # largest budget
    assert {"L2_minus_raw", "L2_minus_L1"} <= set(s["comparisons"])
    assert s["timing"]["forward_passes_per_decision"]["L2"] == 1
    with np.load(tmp_path / "out" / "banking20_fake_seed0.npz", allow_pickle=False) as z:
        assert z["L2_probs"].shape == (150, 4)
        np.testing.assert_allclose(z["L2_probs"].sum(axis=1), 1.0)
    assert any((tmp_path / "cache").rglob("*.npz"))
    json.dumps(s)  # serialisable


def test_run_task_l2_reuses_cached_features(tmp_path: Path) -> None:
    b = CountingBackend(hidden_size=8, num_layers=10)
    kw = {"n_resamples": 5, "l2_budgets": (60,), "cache_dir": tmp_path / "cache"}
    run_task(b, _task(), ["L2"], tmp_path / "o1", **kw)
    first = len(b.calls)
    run_task(b, _task(), ["L2"], tmp_path / "o2", **kw)
    assert first == 4  # calib + test, forward + reversed
    assert len(b.calls) == first  # second run: all from cache


def test_run_task_l2_budget_validation(tmp_path: Path) -> None:
    with pytest.raises(BrierError):
        run_task(FakeBackend(), _task(), ["L2"], tmp_path, l2_budgets=(500,))  # > 80 calib items
    with pytest.raises(BrierError):
        run_task(FakeBackend(), _task(), ["L2"], tmp_path, l2_budgets=())
