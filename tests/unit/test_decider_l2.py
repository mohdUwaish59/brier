import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from brier import Choice, Decider, Noul, Score
from brier.artifacts import (
    ARRAYS_FILE,
    JSON_FILE,
    SCHEMA_VERSION,
    Artifact,
    Calibration,
    load_artifact,
    save_artifact,
)
from brier.backends.fake import FakeBackend
from brier.errors import ArtifactError, BrierError, InsufficientDataError, NotFittedError
from brier.heads.fitted import MAX_HIDDEN, L2Head
from brier.heads.select import select_head

ROUTE = Choice("Which team?", ["billing", "technical", "sales"], name="route")
REFUND = Noul("Is this a refund request?", name="refund")
URGENCY = Score("How urgent?", levels=3, name="urgency")
STATES = [f"customer message {i}" for i in range(75)]
ROUTE_LABELS = [ROUTE.options[i % 3] for i in range(75)]
PROBE = [f"new message {i}" for i in range(6)]


def _backend(**kw) -> FakeBackend:  # type: ignore[no-untyped-def]
    return FakeBackend(hidden_size=12, num_layers=10, **kw)


def _fitted() -> Decider:
    d = Decider(_backend())
    d.fit_head(STATES, ROUTE, ROUTE_LABELS)
    d.fit_head(STATES, REFUND, [i % 2 == 0 for i in range(75)])
    d.fit_head(STATES, URGENCY, [i % 3 + 1 for i in range(75)])
    return d


# ---------- L2Head ----------


def _head(**kw) -> L2Head:  # type: ignore[no-untyped-def]
    args = {
        "layer": 4,
        "solver": "ridge",
        "alpha": 1.0,
        "temperature": 0.2,
        "temperature_at_bound": False,
        "oof_nll": 0.9,
        "oof_accuracy": 0.6,
        "mean": np.zeros(5),
        "scale": np.ones(5),
        "weights": np.ones((5, 3)),
        "bias": np.zeros(3),
    }
    args.update(kw)
    return L2Head(**args)  # type: ignore[arg-type]


def test_l2head_from_selection_matches_selection_log_probs() -> None:
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(3), 25)
    hidden = rng.normal(size=(75, 2, 8))
    hidden[:, 1] += 3 * np.eye(3, 8)[y]
    for solvers in (("ridge",), ("lda",)):
        sel = select_head(hidden, [3, 5], y, 3, solvers=solvers)
        head = L2Head.from_selection(sel)
        assert (head.layer, head.solver, head.alpha) == (sel.layer, sel.solver, sel.alpha)
        np.testing.assert_allclose(head.log_probs(hidden[:, 1]), sel.log_probs(hidden[:, 1]))


@pytest.mark.parametrize(
    "kw",
    [
        {"layer": -1},
        {"layer": True},
        {"solver": "svm"},
        {"alpha": None},  # ridge needs alpha
        {"solver": "lda", "alpha": 1.0},  # lda has none
        {"alpha": -1.0},
        {"temperature": 0.0},
        {"temperature": math.exp(8)},  # outside L2's range
        {"temperature_at_bound": 1},
        {"oof_nll": math.inf},
        {"oof_accuracy": 1.5},
        {"mean": np.zeros(4)},  # d mismatch
        {"scale": np.zeros(5)},  # scale must be > 0
        {"weights": np.ones((5, 1))},  # C < 2
        {"bias": np.zeros(2)},  # C mismatch
        {"weights": np.full((5, 3), np.nan)},
        {
            "mean": np.zeros(MAX_HIDDEN + 1),
            "scale": np.ones(MAX_HIDDEN + 1),
            "weights": np.ones((MAX_HIDDEN + 1, 3)),
        },
    ],
)
def test_l2head_validation(kw: dict) -> None:  # type: ignore[type-arg]
    with pytest.raises(BrierError):
        _head(**kw)


def test_l2head_arrays_read_only() -> None:
    h = _head()
    with pytest.raises(ValueError, match="read-only"):
        h.weights[0, 0] = 2.0


# ---------- Decider.fit_head / level L2 ----------


def test_l2_requires_fit() -> None:
    with pytest.raises(NotFittedError):
        Decider(_backend()).decide("s", [ROUTE], level="L2")


def test_fit_head_and_decide_l2() -> None:
    d = _fitted()
    res = d.decide_batch(PROBE, [ROUTE, REFUND, URGENCY], level="L2")
    for r in res:
        assert r["route"].level == "L2"
        assert sum(r["route"].probs.values()) == pytest.approx(1.0)
        meta = r["route"].meta
        assert meta["n_forward"] == 1
        assert meta["solver"] in ("ridge", "lda")
        assert 0 <= meta["layer"] < 10
        assert meta["temperature"] > 0
    assert isinstance(res[0]["refund"].answer, bool)
    assert res[0]["urgency"].answer in (1, 2, 3)


def test_fit_head_default_layers_and_custom_layers() -> None:
    d = Decider(_backend())
    d.fit_head(STATES, ROUTE, ROUTE_LABELS)
    assert d.decide("s", [ROUTE], level="L2")["route"].meta["layer"] in (4, 6, 8)  # 40-90 % of 10
    d.fit_head(STATES, ROUTE, ROUTE_LABELS, layers=[1])
    assert d.decide("s", [ROUTE], level="L2")["route"].meta["layer"] == 1


@pytest.mark.parametrize("layers", [[], [10], [-1], [1, 1], ["2"]])
def test_fit_head_validates_layers(layers: list) -> None:  # type: ignore[type-arg]
    with pytest.raises(BrierError):
        Decider(_backend()).fit_head(STATES, ROUTE, ROUTE_LABELS, layers=layers)


def test_fit_head_label_and_size_checks() -> None:
    d = Decider(_backend())
    with pytest.raises(BrierError):
        d.fit_head(STATES, ROUTE, ["refunds"] * 75)
    with pytest.raises(BrierError):
        d.fit_head(STATES, ROUTE, ROUTE_LABELS[:-1])
    with pytest.raises(InsufficientDataError):
        d.fit_head(STATES[:59], ROUTE, ROUTE_LABELS[:59])


def test_l2_independent_of_prior_and_temperature() -> None:
    d = _fitted()
    before = d.decide_batch(PROBE, [REFUND], "L2")
    d.fit_prior(STATES, [REFUND])
    d.fit_temperature(STATES, REFUND, [i % 2 == 0 for i in range(75)])
    after = d.decide_batch(PROBE, [REFUND], "L2")
    assert [r["refund"].probs for r in before] == [r["refund"].probs for r in after]


def test_all_requested_questions_need_a_head() -> None:
    d = Decider(_backend())
    d.fit_head(STATES, ROUTE, ROUTE_LABELS)
    with pytest.raises(NotFittedError):
        d.decide("s", [ROUTE, REFUND], level="L2")


# ---------- save / load (schema 2) ----------


def test_round_trip_l2_decisions_identical(tmp_path: Path) -> None:
    d = _fitted()
    d.fit_prior(STATES, [ROUTE])
    d.save(tmp_path / "c")
    loaded = Decider.load(tmp_path / "c", _backend())
    qs = [ROUTE, REFUND, URGENCY]
    for level in ("L0", "L2"):
        a = [
            {k: (dict(v.probs), dict(v.meta)) for k, v in r.items()}
            for r in d.decide_batch(PROBE, qs, level)
        ]  # type: ignore[arg-type]
        b = [
            {k: (dict(v.probs), dict(v.meta)) for k, v in r.items()}
            for r in loaded.decide_batch(PROBE, qs, level)
        ]  # type: ignore[arg-type]
        assert a == b


def test_schema_version_is_2_and_head_json(tmp_path: Path) -> None:
    _fitted().save(tmp_path / "c")
    j = json.loads((tmp_path / "c" / JSON_FILE).read_text(encoding="utf-8"))
    assert j["schema_version"] == SCHEMA_VERSION == 2
    head = j["questions"][0]["head"]
    assert set(head) == {
        "layer",
        "solver",
        "alpha",
        "temperature",
        "temperature_at_bound",
        "oof_nll",
        "oof_accuracy",
    }


def test_load_rejects_head_layer_beyond_model(tmp_path: Path) -> None:
    _fitted().save(tmp_path / "c")
    with pytest.raises(ArtifactError, match="layer"):
        Decider.load(tmp_path / "c", FakeBackend(hidden_size=12, num_layers=3))


def test_version_1_artifact_still_loads(tmp_path: Path) -> None:
    d = Decider(_backend())
    d.fit_temperature(STATES[:60], REFUND, [i % 2 == 0 for i in range(60)])
    d.save(tmp_path / "c")
    p = tmp_path / "c" / JSON_FILE
    j = json.loads(p.read_text(encoding="utf-8"))
    j["schema_version"] = 1
    for q in j["questions"]:
        q.pop("head")
    p.write_text(json.dumps(j), encoding="utf-8")
    loaded = Decider.load(tmp_path / "c", _backend())
    assert loaded.decide("s", [REFUND], "L1")["refund"].meta["temperature"] > 0


def test_version_1_with_head_key_rejected(tmp_path: Path) -> None:
    _fitted().save(tmp_path / "c")
    p = tmp_path / "c" / JSON_FILE
    j = json.loads(p.read_text(encoding="utf-8"))
    j["schema_version"] = 1
    p.write_text(json.dumps(j), encoding="utf-8")
    with pytest.raises(ArtifactError):
        Decider.load(tmp_path / "c", _backend())


def _save_one(tmp_path: Path) -> Path:
    d = tmp_path / "a"
    save_artifact(d, Artifact("fake", None, 1.0, (Calibration(ROUTE, head=_head()),)))
    return d


def _edit(d: Path, fn) -> None:  # type: ignore[no-untyped-def]
    p = d / JSON_FILE
    j = json.loads(p.read_text(encoding="utf-8"))
    fn(j)
    p.write_text(json.dumps(j), encoding="utf-8")


@pytest.mark.parametrize(
    "edit",
    [
        lambda j: j["questions"][0]["head"].update(solver="svm"),
        lambda j: j["questions"][0]["head"].update(layer="4"),
        lambda j: j["questions"][0]["head"].update(extra=1),
        lambda j: j["questions"][0]["head"].pop("oof_nll"),
        lambda j: j["questions"][0].update(head=[1, 2]),
        lambda j: j["questions"][0]["head"].update(temperature=1e9),
    ],
)
def test_malformed_head_json_rejected(tmp_path: Path, edit) -> None:  # type: ignore[no-untyped-def]
    d = _save_one(tmp_path)
    _edit(d, edit)
    with pytest.raises(ArtifactError):
        load_artifact(d, model_id="fake", revision=None)


def test_head_arrays_validated_with_hardened_loader(tmp_path: Path) -> None:
    import io
    import zipfile

    d = _save_one(tmp_path)
    with zipfile.ZipFile(d / ARRAYS_FILE) as z:
        members = {n: z.read(n) for n in z.namelist()}
    assert set(members) == {
        "head_0_mean.npy",
        "head_0_scale.npy",
        "head_0_weights.npy",
        "head_0_bias.npy",
    }
    # replace weights with a header that declares a gigantic shape
    f = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        f, {"descr": "<f8", "fortran_order": False, "shape": (10**9, 3)}
    )
    members["head_0_weights.npy"] = f.getvalue()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for n, b in members.items():
            z.writestr(n, b)
    (d / ARRAYS_FILE).write_bytes(buf.getvalue())
    _edit(d, lambda j: j.update(arrays_sha256=hashlib.sha256(buf.getvalue()).hexdigest()))
    with pytest.raises(ArtifactError):
        load_artifact(d, model_id="fake", revision=None)


def test_head_class_count_must_match_question(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError):
        Calibration(REFUND, head=_head())  # REFUND has 2 answers, head has 3 classes


def test_temperature_at_lower_bound_warns(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import dataclasses
    import math as _math

    from brier import decider as decider_mod

    real = decider_mod.select_head

    def at_lower_bound(*a, **k):  # type: ignore[no-untyped-def]
        sel = real(*a, **k)
        return dataclasses.replace(sel, temperature=_math.exp(-7), temperature_at_bound=True)

    monkeypatch.setattr(decider_mod, "select_head", at_lower_bound)
    with pytest.warns(UserWarning, match="bound"):
        Decider(_backend()).fit_head(STATES, ROUTE, ROUTE_LABELS)


def test_temperature_at_upper_bound_does_not_warn(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import dataclasses
    import math as _math
    import warnings

    from brier import decider as decider_mod

    real = decider_mod.select_head

    def at_upper_bound(*a, **k):  # type: ignore[no-untyped-def]
        sel = real(*a, **k)
        return dataclasses.replace(sel, temperature=_math.exp(7), temperature_at_bound=True)

    monkeypatch.setattr(decider_mod, "select_head", at_upper_bound)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Decider(_backend()).fit_head(
            STATES, ROUTE, ROUTE_LABELS
        )  # uninformative, not overconfident


@pytest.mark.parametrize(
    "edit",
    [
        lambda j: j["questions"][0]["head"].update(alpha="1.0"),
        lambda j: j["questions"][0]["head"].update(oof_nll="0.9"),
        lambda j: j["questions"][0]["head"].update(temperature_at_bound="no"),
    ],
    ids=["alpha-type", "metric-type", "flag-type"],
)
def test_head_json_field_types(tmp_path: Path, edit) -> None:  # type: ignore[no-untyped-def]
    d = _save_one(tmp_path)
    _edit(d, edit)
    with pytest.raises(ArtifactError):
        load_artifact(d, model_id="fake", revision=None)


def test_calibration_head_must_be_l2head() -> None:
    with pytest.raises(ArtifactError):
        Calibration(ROUTE, head="not a head")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kw",
    [
        {"alpha": 10**400},
        {"oof_nll": 10**400},
        {"temperature": 10**400},
        {"scale": np.full(5, 1e-7)},  # below the fitting floor
    ],
    ids=["alpha-overflow", "nll-overflow", "temp-overflow", "scale-below-floor"],
)
def test_l2head_rejects_overflow_and_tiny_scale_as_brier_error(kw: dict) -> None:  # type: ignore[type-arg]
    with pytest.raises(BrierError):
        _head(**kw)


def test_artifact_question_count_is_capped(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from brier import artifacts

    d = tmp_path / "a"
    qs = tuple(Calibration(Noul(f"q{i}?", name=f"q{i}"), temperature=1.0) for i in range(3))
    save_artifact(d, Artifact("fake", None, 1.0, qs))
    monkeypatch.setattr(artifacts, "MAX_ARTIFACT_QUESTIONS", 2)
    with pytest.raises(ArtifactError, match="at most 2"):
        load_artifact(d, model_id="fake", revision=None)


def test_npy_shape_dimensions_must_be_exact_ints(tmp_path: Path) -> None:
    import io
    import zipfile

    d = _save_one(tmp_path)
    with zipfile.ZipFile(d / ARRAYS_FILE) as z:
        members = {n: z.read(n) for n in z.namelist()}
    raw = members["head_0_mean.npy"]
    members["head_0_mean.npy"] = raw.replace(b"(5,)", b"(True,)", 1)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for n, b in members.items():
            z.writestr(n, b)
    (d / ARRAYS_FILE).write_bytes(buf.getvalue())
    _edit(d, lambda j: j.update(arrays_sha256=hashlib.sha256(buf.getvalue()).hexdigest()))
    with pytest.raises(ArtifactError, match="expected shape"):
        load_artifact(d, model_id="fake", revision=None)
