import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from brier import Choice
from brier.backends.fake import FakeBackend
from brier.bench import tasks
from brier.bench.__main__ import build_parser, main
from brier.bench.run import run_task
from brier.bench.tasks import Task, make_banking20, parse_csv
from brier.errors import BrierError, InsufficientDataError

# ---------- tasks ----------

CSV = 'text,category\n"hello, there",card_arrival\n"lost my card",lost_or_stolen_card\n'


def test_parse_csv_handles_quotes_and_header() -> None:
    assert parse_csv(CSV.encode()) == [
        ("hello, there", "card_arrival"),
        ("lost my card", "lost_or_stolen_card"),
    ]


def _rows(counts: dict[str, int]) -> list[tuple[str, str]]:
    return [(f"{name} message {i}", name) for name, n in counts.items() for i in range(n)]


def _task(seed: int = 0) -> Task:
    counts = {f"intent_{c}": 30 - i for i, c in enumerate("abcdef")}  # 30,29,...,25
    return make_banking20(_rows(counts), seed=seed, top_k=4, n_pool=10, n_cal=8)


def test_make_banking20_top_k_names_and_sizes() -> None:
    t = _task()
    assert t.name == "banking20"
    assert isinstance(t.question, Choice)
    # 4 most frequent intents, shown with "_" -> " ", sorted by name
    assert t.question.options == ("intent a", "intent b", "intent c", "intent d")
    total = 30 + 29 + 28 + 27
    assert len(t.pool) == 10
    assert len(t.calib_states) == len(t.calib_labels) == 8
    assert len(t.test_states) == len(t.test_labels) == total - 18
    assert set(t.test_labels) <= set(range(4))


def test_make_banking20_label_matches_text() -> None:
    t = _task()
    for s, y in zip(t.test_states, t.test_labels, strict=True):
        assert s.startswith(t.question.options[y].replace(" ", "_"))


def test_make_banking20_splits_disjoint_and_seeded() -> None:
    t = _task()
    parts = [set(t.pool), set(t.calib_states), set(t.test_states)]
    assert sum(map(len, parts)) == len(set().union(*parts))
    assert _task(seed=0) == t
    assert _task(seed=1).test_states != t.test_states


def test_make_banking20_tie_break_is_by_name() -> None:
    rows = _rows({"zeta": 5, "alpha": 5, "mid": 9})
    t = make_banking20(rows, top_k=2, n_pool=1, n_cal=1)
    assert t.question.options == ("alpha", "mid")


def test_make_banking20_needs_enough_items() -> None:
    with pytest.raises(InsufficientDataError):
        make_banking20(_rows({"a": 3, "b": 3}), top_k=2, n_pool=5, n_cal=5)


def test_fetch_uses_verified_cache_without_network(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    data = CSV.encode()
    monkeypatch.setattr(tasks, "BANKING77_SHA256", hashlib.sha256(data).hexdigest())
    (tmp_path / tasks.BANKING77_FILE).write_bytes(data)

    def no_network(url: str) -> bytes:
        raise AssertionError("network used")

    monkeypatch.setattr(tasks, "_download", no_network)
    assert tasks.fetch_banking77(tmp_path) == data


def test_fetch_downloads_verifies_and_caches(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    data = CSV.encode()
    monkeypatch.setattr(tasks, "BANKING77_SHA256", hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(tasks, "_download", lambda url: data)
    assert tasks.fetch_banking77(tmp_path) == data
    assert (tmp_path / tasks.BANKING77_FILE).read_bytes() == data


def test_fetch_rejects_hash_mismatch_and_does_not_cache(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(tasks, "_download", lambda url: b"tampered")
    with pytest.raises(BrierError):
        tasks.fetch_banking77(tmp_path)
    assert not (tmp_path / tasks.BANKING77_FILE).exists()


def test_fetch_redownloads_when_cache_is_corrupt(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    data = CSV.encode()
    monkeypatch.setattr(tasks, "BANKING77_SHA256", hashlib.sha256(data).hexdigest())
    (tmp_path / tasks.BANKING77_FILE).write_bytes(b"corrupt")
    monkeypatch.setattr(tasks, "_download", lambda url: data)
    assert tasks.fetch_banking77(tmp_path) == data


def test_source_metadata_records_licence() -> None:
    assert tasks.BANKING77_SOURCE["licence"] == "CC-BY-4.0"
    assert tasks.BANKING77_SOURCE["sha256"] == tasks.BANKING77_SHA256
    assert tasks.BANKING77_COMMIT in tasks.BANKING77_SOURCE["url"]


# ---------- runner ----------

BIASED = FakeBackend(position_bias=(2.0, 0.5, 0.0, -0.5))


def test_run_task_writes_schema(tmp_path: Path) -> None:
    summary = run_task(BIASED, _task(), ["raw", "L0"], tmp_path, n_resamples=50)
    json_path = tmp_path / "banking20_fake_seed0.json"
    assert json.loads(json_path.read_text()) == summary
    assert summary["schema_version"] == 1
    assert summary["task"] == "banking20"
    assert summary["split_seed"] == 0
    assert summary["n_test"] == len(_task().test_states)
    assert summary["model"] == {"id": "fake", "revision": None, "dtype": None}
    assert set(summary["env"]) == {"python", "torch", "transformers", "device"}
    assert summary["dataset"] == _task().source
    for level in ("raw", "L0"):
        m = summary["levels"][level]
        for key in ("accuracy", "nll", "brier", "ece", "ece_width", "aurc", "flip_rate"):
            v, lo, hi = m[key]
            assert lo <= v <= hi
        assert "coverage_at_risk_0.05_in_sample" in m
    assert summary["timing"]["forward_passes_per_decision"] == {"raw": 1, "L0": 4}
    assert set(summary["timing"]["wall_seconds"]) == {"raw", "L0"}
    assert set(summary["comparisons"]["L0_minus_raw"]) >= {"accuracy", "flip_rate", "ece"}


def test_run_task_l0_flip_rate_below_raw(tmp_path: Path) -> None:
    s = run_task(BIASED, _task(), ["raw", "L0"], tmp_path, n_resamples=50)
    assert s["levels"]["raw"]["flip_rate"][0] > 0.2
    assert s["levels"]["L0"]["flip_rate"][0] < s["levels"]["raw"]["flip_rate"][0]


def test_run_task_npz_reloads_without_pickle_and_has_no_text(tmp_path: Path) -> None:
    run_task(BIASED, _task(), ["raw", "L0"], tmp_path, n_resamples=20)
    with np.load(tmp_path / "banking20_fake_seed0.npz", allow_pickle=False) as z:
        keys = set(z.files)
        assert {"labels", "options", "raw_probs", "raw_probs_reversed", "L0_probs"} <= keys
        n = len(_task().test_states)
        assert z["raw_probs"].shape == (n, 4)
        np.testing.assert_allclose(z["L0_probs"].sum(axis=1), 1.0)
        assert list(z["options"]) == list(_task().question.options)
        assert not any(z[k].dtype.kind in "OU" for k in keys if k != "options")


def test_run_task_limit(tmp_path: Path) -> None:
    s = run_task(BIASED, _task(), ["raw"], tmp_path, limit=5, n_resamples=20)
    assert s["n_test"] == 5
    assert s["n_pool"] == 0  # raw needs no prior pool


def test_run_task_writes_after_each_level(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from brier.bench import run as run_mod

    calls: list[str] = []
    real = run_mod._write

    def spy(out, stem, summary, arrays):  # type: ignore[no-untyped-def]
        calls.append(",".join(summary["levels"]))
        real(out, stem, summary, arrays)

    monkeypatch.setattr(run_mod, "_write", spy)
    run_task(BIASED, _task(), ["raw", "L0"], tmp_path, n_resamples=10)
    assert calls == ["raw", "raw,L0"]


@pytest.mark.parametrize("levels", [["L1"], ["L2"], ["bogus"], []])
def test_run_task_rejects_unsupported_levels(tmp_path: Path, levels: list[str]) -> None:
    with pytest.raises(BrierError):
        run_task(BIASED, _task(), levels, tmp_path)


# ---------- CLI ----------


def test_parser_defaults_and_levels() -> None:
    args = build_parser().parse_args(["run", "--model", "m", "--out", "r"])
    assert args.task == "banking20"
    assert args.levels == "raw,L0"
    assert args.seed == 0
    assert args.limit is None


def test_main_runs_with_injected_backend(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from brier.bench import __main__ as cli

    monkeypatch.setattr(cli, "_load_task", lambda name, seed: _task(seed))
    monkeypatch.setattr(cli, "_make_backend", lambda args: BIASED)
    argv = ["run", "--model", "fake", "--out", str(tmp_path), "--levels", "raw"]
    assert main([*argv, "--n-resamples", "10"]) == 0
    assert (tmp_path / "banking20_fake_seed0.json").exists()


def test_main_rejects_unknown_task(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["run", "--model", "m", "--out", str(tmp_path), "--task", "nope"])


def test_git_commit_in_this_checkout_is_a_sha() -> None:
    from brier.bench.__main__ import _git_commit

    sha = _git_commit()
    assert sha is None or (len(sha) == 40 and all(c in "0123456789abcdef" for c in sha))


def test_no_temp_files_left_behind(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    data = CSV.encode()
    monkeypatch.setattr(tasks, "BANKING77_SHA256", hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(tasks, "_download", lambda url: data)
    tasks.fetch_banking77(tmp_path / "cache")
    run_task(BIASED, _task(), ["raw"], tmp_path / "out", n_resamples=5)
    leftovers = [p.name for p in tmp_path.rglob("*.tmp")]
    assert leftovers == []
