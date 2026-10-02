"""Conformance check: can this backend run brier, and at which levels? (`brier check`).

Runs before any calibration, on built-in inputs only (no user data): label tokens per
question type, batched-vs-single consistency, hidden-state access, and a small sanity task.
Every check catches any exception from the backend and reports it as a failed item: a check
tool must describe a broken model, not crash on it.
"""

from __future__ import annotations

import string
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from brier.backends.base import Backend
from brier.decider import Decider
from brier.heads.select import default_layers
from brier.prompts import labels, render
from brier.questions import Choice, Noul, Question, Score
from brier.readout import resolve_labels

Status = Literal["pass", "warn", "fail"]

# Max probability difference between a batched and a single-prompt forward pass. Padding or
# masking bugs move probabilities by far more; bf16 noise stays well below.
BATCH_TOLERANCE = 2e-2
# Below this L0 accuracy on the sanity task the model is probably too weak to be useful.
SANITY_WARN_BELOW = 0.6

_ROUTE = Choice(
    "Which team should handle this message?",
    ["billing", "technical support", "sales"],
    name="route",
)
_ANGRY = Noul("Is the customer angry?", name="angry")
_URGENCY = Score(
    "How urgent is this message?",
    levels=5,
    name="urgency",
    labels=["not urgent", "slightly urgent", "moderately urgent", "very urgent", "critical"],
)

# Built-in sanity items: (question, [(state, answer), ...]). Clear cases a capable
# instruction-tuned model gets right; Score counts as right within one level.
SANITY: tuple[tuple[Question, tuple[tuple[str, object], ...]], ...] = (
    (
        _ROUTE,
        (
            ("I was charged twice for my subscription this month, please refund one.", "billing"),
            ("The app crashes every time I open the settings page.", "technical support"),
            ("I'd like a quote for 50 licences for my company.", "sales"),
            ("My invoice shows the wrong VAT number, please correct it.", "billing"),
            ("I can't log in: the password reset email never arrives.", "technical support"),
            ("Do you offer a discount on annual plans? We are thinking of buying.", "sales"),
        ),
    ),
    (
        _ANGRY,
        (
            ("This is the third time I've asked. Unacceptable, fix it NOW!", True),
            ("Thanks so much, your team was wonderful!", False),
            ("Worst service ever. I'm furious and I'm cancelling.", True),
            ("Just wondering when the new version comes out, no rush.", False),
        ),
    ),
    (
        _URGENCY,
        (
            ("Production is down for all our customers and we lose money every minute!", 5),
            ("Whenever you have time, could you update my mailing address? No hurry.", 1),
            ("Our payment system is broken and no customer can check out right now!", 5),
            ("Just a small idea for some future release, no rush at all.", 1),
        ),
    ),
)


@dataclass(frozen=True)
class CheckItem:
    """One check: ``name``, ``status`` (pass / warn / fail) and a human-readable detail."""

    name: str
    status: Status
    detail: str


@dataclass(frozen=True)
class CheckReport:
    """Result of :func:`check_backend`.

    Attributes
    ----------
    model_id, revision : str, str or None
        The checked backend.
    items : tuple of CheckItem
        Every check, in run order.
    levels : Mapping[str, bool]
        Whether ``raw``, ``L0``, ``L1`` and ``L2`` can run on this backend.
    sanity : Mapping[str, float]
        Sanity-task numbers (empty if the task could not run).
    """

    model_id: str
    revision: str | None
    items: tuple[CheckItem, ...]
    levels: Mapping[str, bool]
    sanity: Mapping[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True if no check failed (warnings allowed)."""
        return all(i.status != "fail" for i in self.items)

    def to_dict(self) -> dict[str, object]:
        """JSON-ready form."""
        return {
            "model": {"id": self.model_id, "revision": self.revision},
            "ok": self.ok,
            "levels": dict(self.levels),
            "checks": [
                {"name": i.name, "status": i.status, "detail": i.detail} for i in self.items
            ],
            "sanity": dict(self.sanity),
        }


def _error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:300]


def _check_labels(backend: Backend) -> list[CheckItem]:
    out = []
    letters = tuple(string.ascii_uppercase)
    for name, check_labels, shown in (
        ("labels.choice", letters, "A-Z"),
        ("labels.noul", ("Yes", "No"), "Yes, No"),
    ):
        try:
            backend.label_token_ids(check_labels)
            out.append(CheckItem(name, "pass", f"{shown}: single, distinct tokens"))
        except Exception as exc:
            out.append(CheckItem(name, "fail", _error(exc)))
    try:
        letters_used, _ = resolve_labels(backend, Score("q", levels=10, name="s"))
        how = "lettered options (digit labels are not single tokens)" if letters_used else "digits"
        out.append(CheckItem("labels.score", "pass", f"10 levels as {how}"))
    except Exception as exc:
        out.append(CheckItem("labels.score", "fail", _error(exc)))
    return out


def _check_batch(backend: Backend) -> CheckItem:
    states = [s for _, items in SANITY[:1] for s, _ in items[:4]]
    prompts = [render(s, _ROUTE) for s in states]
    try:
        ids = backend.label_token_ids(labels(_ROUTE))
        batched = backend.label_logprobs(prompts, ids)
        single = np.concatenate([backend.label_logprobs([p], ids) for p in prompts])
        if not (np.all(np.isfinite(batched)) and np.all(np.isfinite(single))):
            return CheckItem("batch_consistency", "fail", "non-finite log-probabilities")
        diff = float(np.max(np.abs(np.exp(batched) - np.exp(single))))
    except Exception as exc:
        return CheckItem("batch_consistency", "fail", _error(exc))
    status: Status = "pass" if diff <= BATCH_TOLERANCE else "fail"
    return CheckItem(
        "batch_consistency",
        status,
        f"max |p_batched - p_single| = {diff:.1e} (limit {BATCH_TOLERANCE:.0e})",
    )


def _check_hidden(backend: Backend) -> CheckItem:
    try:
        layers = default_layers(backend.num_layers)
        h = backend.hidden_states([render(SANITY[0][1][0][0], _ROUTE)], layers)
        if h.shape[:2] != (1, len(layers)) or not np.all(np.isfinite(h)):
            return CheckItem("hidden_states", "fail", f"unexpected output, shape {h.shape}")
    except Exception as exc:
        return CheckItem("hidden_states", "fail", _error(exc))
    return CheckItem(
        "hidden_states",
        "pass",
        f"{len(layers)} candidate layers ({layers[0]}-{layers[-1]} of {backend.num_layers}), "
        f"d = {h.shape[2]}",
    )


def _correct(question: Question, answer: object, gold: object) -> bool:
    if isinstance(question, Score):
        return abs(int(str(answer)) - int(str(gold))) <= 1
    return bool(answer == gold)


def _run_sanity(backend: Backend) -> tuple[CheckItem, dict[str, float]]:
    decider = Decider(backend)
    reversed_route = Choice(_ROUTE.text, list(reversed(_ROUTE.options)), name=_ROUTE.name)
    try:
        result: dict[str, float] = {}
        for level in ("raw", "L0"):
            right = total = 0
            t0 = time.perf_counter()
            for q, items in SANITY:
                got = decider.decide_batch([s for s, _ in items], [q], level=level)
                answers = [g[q.name].answer for g in got]
                right += sum(_correct(q, a, y) for a, (_, y) in zip(answers, items, strict=True))
                total += len(items)
            if level == "raw":  # raw is one forward pass per prompt
                result["ms_per_prompt"] = 1000 * (time.perf_counter() - t0) / total
            route_states = [s for s, _ in SANITY[0][1]]
            a = decider.decide_batch(route_states, [_ROUTE], level=level)
            b = decider.decide_batch(route_states, [reversed_route], level=level)
            name = _ROUTE.name
            flips = [x[name].answer != y[name].answer for x, y in zip(a, b, strict=True)]
            result[f"{level}_accuracy"] = right / total
            result[f"{level}_flip_rate"] = sum(flips) / len(flips)
    except Exception as exc:
        return CheckItem("sanity", "fail", _error(exc)), {}
    acc = result["L0_accuracy"]
    detail = (
        f"accuracy raw {result['raw_accuracy']:.0%} / L0 {acc:.0%} on {total} items, "
        f"order flips raw {result['raw_flip_rate']:.0%} / L0 {result['L0_flip_rate']:.0%}, "
        f"{result['ms_per_prompt']:.0f} ms per prompt"
    )
    if acc < SANITY_WARN_BELOW:
        detail += "; low accuracy: the model may be too weak for useful decisions"
        return CheckItem("sanity", "warn", detail), result
    return CheckItem("sanity", "pass", detail), result


def check_backend(backend: Backend) -> CheckReport:
    """Run every conformance check on ``backend``.

    Parameters
    ----------
    backend : Backend
        A loaded backend (e.g. :class:`brier.backends.hf.HFBackend`).

    Returns
    -------
    CheckReport
        Per-check status, supported levels and sanity-task numbers. Never raises for a
        failing check: failures are reported as ``fail`` items.
    """
    items: list[CheckItem] = []
    if backend.revision is None:
        items.append(
            CheckItem("revision", "warn", "not pinned: pass a commit SHA to fix the weights")
        )
    else:
        items.append(CheckItem("revision", "pass", str(backend.revision)))
    prompt_format = getattr(backend, "prompt_format", None)
    if prompt_format is not None:
        detail = {
            "chat": "the model's chat template",
            "plain": "plain text: the model has no chat template (base model, ADR-0008)",
        }.get(str(prompt_format), str(prompt_format))
        items.append(CheckItem("prompt_format", "pass", detail))
    label_items = _check_labels(backend)
    items += label_items
    items.append(_check_batch(backend))
    hidden = _check_hidden(backend)
    items.append(hidden)
    sanity_item, sanity = _run_sanity(backend)
    items.append(sanity_item)

    readout_ok = all(i.status == "pass" for i in label_items)
    readout_ok = readout_ok and all(i.status != "fail" for i in items if i.name != "hidden_states")
    levels = {"raw": readout_ok, "L0": readout_ok, "L1": readout_ok, "L2": hidden.status == "pass"}
    return CheckReport(backend.model_id, backend.revision, tuple(items), levels, sanity)


def format_report(report: CheckReport) -> str:
    """Plain-text table of a :class:`CheckReport`."""
    lines = [f"brier check: {report.model_id} @ {report.revision or '(unpinned)'}", ""]
    width = max(len(i.name) for i in report.items)
    lines += [f"  {i.status.upper():<4}  {i.name:<{width}}  {i.detail}" for i in report.items]
    supported = [lv for lv, ok in report.levels.items() if ok]
    lines += ["", f"Supported levels: {', '.join(supported) if supported else 'none'}"]
    lines.append("Result: OK" if report.ok else "Result: FAILED")
    return "\n".join(lines)


__all__: Sequence[str] = ["SANITY", "CheckItem", "CheckReport", "check_backend", "format_report"]
