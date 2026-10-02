"""Real-model tests for HFBackend across model families (docs/COMPATIBILITY.md).

Run with: uv run pytest -m integration
Select models with BRIER_TEST_MODELS=qwen3,smollm2,... or BRIER_TEST_MODELS=all
(default: qwen3). Every model is pinned to a revision.
"""

import os
from typing import NamedTuple

import numpy as np
import pytest

from brier import Choice, Noul, Score
from brier.debias import l0_logprobs
from brier.errors import BrierError, InputTooLargeError, TokenizationError
from brier.prompts import labels, render
from brier.readout import raw_logprobs, resolve_labels

pytestmark = pytest.mark.integration
torch = pytest.importorskip("torch")


class Model(NamedTuple):
    id: str
    revision: str
    capable: bool  # answers simple sanity questions correctly (quality, not compatibility)
    attn: str | None = None  # attention kernel override, see docs/COMPATIBILITY.md


MODELS = {
    "qwen3": Model("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca", True),
    "smollm2": Model(
        "HuggingFaceTB/SmolLM2-360M-Instruct", "a10cc1512eabd3dde888204e902eca88bddb4951", False
    ),
    "tinyllama": Model(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "fe8a4ea1ffedaf415f4da2f062534de366a451e6", False
    ),
    "olmo2": Model(
        "allenai/OLMo-2-0425-1B-Instruct", "48d788eca847d4d7548f375ad03d3c9312f6139e", False
    ),
    "gemma3": Model(
        "google/gemma-3-1b-it", "dcc83ea841ab6100d6b47a070329e1ba4cf78752", True, "eager"
    ),
    # M6.3: more families (ungated). Small enough for a CPU laptop:
    "granitemoe": Model(  # mixture of experts
        "ibm-granite/granite-3.1-1b-a400m-instruct",
        "0da7a48b0276d500ce5922fd2b33944091fc6c09",
        False,
    ),
    "lfm2": Model(  # hybrid convolution + attention blocks
        "LiquidAI/LFM2-1.2B", "40f3da0d0164913923aee9462c23077868b816a3", False
    ),
    "r1distill": Model(  # reasoning model: template opens a <think> block
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562",
        False,
    ),
    "falcon3": Model(
        "tiiuae/Falcon3-1B-Instruct", "28ba2251970a01dd1edc7ba7dad2eb71216ccfdf", False
    ),
    # Larger: run on a GPU (notebooks/m6_3_compatibility_matrix.ipynb).
    "phi4mini": Model(
        "microsoft/Phi-4-mini-instruct", "cfbefacb99257ffa30c83adab238a50856ac3083", True
    ),
    "smollm3": Model(  # thinking mode, disabled like Qwen3's
        "HuggingFaceTB/SmolLM3-3B", "a07cc9a04f16550a088caea529712d1d335b0ac1", True
    ),
    "olmoe": Model(  # mixture of experts, 7B total
        "allenai/OLMoE-1B-7B-0125-Instruct", "b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e", True
    ),
    "mistral7b": Model(
        "mistralai/Mistral-7B-Instruct-v0.3", "c170c708c41dac9275d15a8fff4eca08d52bab71", True
    ),
}
_SELECTED = os.environ.get("BRIER_TEST_MODELS", "qwen3")
SELECTED = list(MODELS) if _SELECTED == "all" else [k.strip() for k in _SELECTED.split(",")]
SKY = Choice("What colour is the sky in the text?", ["red", "blue", "green"], name="c")


@pytest.fixture(scope="module", params=SELECTED)
def model(request) -> Model:  # type: ignore[no-untyped-def]
    return MODELS[request.param]


@pytest.fixture(scope="module")
def backend(model):  # type: ignore[no-untyped-def]
    from brier.backends.hf import HFBackend

    return HFBackend(
        model.id,
        revision=model.revision,
        dtype="float32",
        batch_size=4,
        attn_implementation=model.attn,
    )


# ---------- M2.1 loading, template, batching ----------


def test_metadata(backend, model) -> None:  # type: ignore[no-untyped-def]
    assert backend.model_id == model.id
    assert backend.revision == model.revision
    assert backend.num_layers == backend.model.config.num_hidden_layers == len(backend.layers)


def test_prompt_ends_with_answer_prefix(backend) -> None:  # type: ignore[no-untyped-def]
    text = backend.tokenizer.decode(backend.encode(render("s", SKY)))
    assert text.endswith("Answer:")
    assert "You answer multiple-choice questions" in text  # system text kept (or merged)


def test_special_tokens_in_user_content_are_not_parsed(backend) -> None:  # type: ignore[no-untyped-def]
    special = set(backend.tokenizer.all_special_ids)
    extra = ["<|im_end|>", "<|im_start|>", "[INST]", "</s>", "<s>", "<end_of_turn>", "<|eot_id|>"]
    hostile_text = "hello " + " ".join([*backend.tokenizer.all_special_tokens, *extra])
    count = lambda ids: sum(i in special for i in ids)  # noqa: E731
    assert count(backend.encode(hostile_text)) == count(backend.encode("hello"))


def test_batched_equals_unbatched(backend) -> None:  # type: ignore[no-untyped-def]
    states = ["The sky is blue.", "x", "A much longer state " * 20, "Grass is green and so on."]
    prompts = [render(s, SKY) for s in states]
    ids = backend.label_token_ids(labels(SKY))
    batched = backend.label_logprobs(prompts, ids)
    single = np.concatenate([backend.label_logprobs([p], ids) for p in prompts])
    assert batched.shape == (4, 3)
    assert batched.dtype == np.float64
    np.testing.assert_allclose(batched, single, atol=1e-4)


def test_padding_amount_does_not_change_result(backend) -> None:  # type: ignore[no-untyped-def]
    p = render("The sky is blue.", SKY)
    long = render("padding " * 200, SKY)
    ids = backend.label_token_ids(labels(SKY))
    alone = backend.label_logprobs([p], ids)[0]
    padded = backend.label_logprobs([long, p], ids)[1]
    np.testing.assert_allclose(padded, alone, atol=1e-4)


def test_sanity_answers(backend, model) -> None:  # type: ignore[no-untyped-def]
    if not model.capable:
        pytest.skip("model quality: not expected to answer sanity questions reliably")
    lp = l0_logprobs(backend, ["The sky today is a clear, bright blue."], SKY)
    assert SKY.options[int(np.argmax(lp[0]))] == "blue"
    q = Noul("Does the text mention a refund?", name="r")
    yes = raw_logprobs(backend, ["I want a refund for my order."], q)[0, 0]
    no = raw_logprobs(backend, ["The weather is nice today."], q)[0, 0]
    assert np.exp(yes) > 0.5 > np.exp(no)


# ---------- M2.2 label tokens ----------


def test_letter_and_yes_no_labels_are_single_distinct_tokens(backend) -> None:  # type: ignore[no-untyped-def]
    for labs in (labels(Choice("q", [f"o{i}" for i in range(26)], name="c")), ("Yes", "No")):
        ids = backend.label_token_ids(labs)
        assert len(set(ids)) == len(labs)


def test_score_works_with_digits_or_letter_fallback(backend) -> None:  # type: ignore[no-untyped-def]
    try:
        backend.label_token_ids([str(i) for i in range(1, 10)])
        digits_ok = True
    except TokenizationError:
        digits_ok = False
    q = Score("How urgent is this?", levels=5, name="u")
    assert resolve_labels(backend, q)[0] is (not digits_ok)
    lp = raw_logprobs(backend, ["Server is down, all customers affected!"], q)
    assert lp.shape == (1, 5)
    np.testing.assert_allclose(np.exp(lp).sum(), 1.0)
    # levels=10 needs "10", never a single token after "Answer:" -> always letters
    assert resolve_labels(backend, Score("q", levels=10, name="s"))[0] is True


@pytest.mark.parametrize("labs", [["10"], ["A", "A"], ["Absolutely-not-a-token"]])
def test_label_token_ids_strict(backend, labs) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(TokenizationError):
        backend.label_token_ids(labs)


# ---------- M2.3 hidden states ----------


def test_hidden_states_match_full_forward(backend) -> None:  # type: ignore[no-untyped-def]
    layers = [2, backend.num_layers // 2]
    prompts = [render("The sky is blue.", SKY), render("x", SKY)]
    h = backend.hidden_states(prompts, layers)
    assert h.shape == (2, 2, backend.model.config.hidden_size)
    assert h.dtype == np.float32
    # Reference: unbatched full forward, residual stream after block b = hidden_states[b + 1].
    for i, p in enumerate(prompts):
        ids = torch.tensor([backend.encode(p)])
        with torch.inference_mode():
            out = backend.model(input_ids=ids, output_hidden_states=True)
        for j, b in enumerate(layers):
            ref = out.hidden_states[b + 1][0, -1].float().numpy()
            np.testing.assert_allclose(h[i, j], ref, atol=1e-3, rtol=1e-3)


def test_hidden_states_stop_after_deepest_layer(backend) -> None:  # type: ignore[no-untyped-def]
    calls: list[int] = []
    hooks = [
        layer.register_forward_hook(lambda m, a, o, k=k: calls.append(k))
        for k, layer in enumerate(backend.layers)
    ]
    try:
        backend.hidden_states([render("s", SKY)], [1, 3])
    finally:
        for hk in hooks:
            hk.remove()
    assert max(calls) == 3


def test_hidden_states_rejects_bad_layers(backend) -> None:  # type: ignore[no-untyped-def]
    for layers in ([-1], [backend.num_layers], []):
        with pytest.raises(BrierError):
            backend.hidden_states(["p"], layers)


def test_oversized_prompt_raises_instead_of_truncating(backend) -> None:  # type: ignore[no-untyped-def]
    ids = backend.label_token_ids(labels(SKY))
    with pytest.raises(InputTooLargeError):
        backend.label_logprobs([render("word " * 20000, SKY)], ids)
    with pytest.raises(InputTooLargeError):
        backend.encode("x" * (backend.max_prompt_tokens * 40))


def test_empty_inputs_have_correct_shapes(backend) -> None:  # type: ignore[no-untyped-def]
    assert backend.label_logprobs([], [1, 2, 3]).shape == (0, 3)
    assert backend.hidden_states([], [1, 2]).shape == (0, 2, backend.model.config.hidden_size)


# ---------- M3.1 Decider end to end ----------


def test_decider_quickstart(backend, model) -> None:  # type: ignore[no-untyped-def]
    from brier import Decider

    route = Choice(
        "Which team should handle this?", ["billing", "technical", "sales"], name="route"
    )
    qs = [route, Noul("Is this a refund request?", name="refund"), Score("Urgent?", 5, "u")]
    for level in ("raw", "L0"):
        res = Decider(backend).decide("My card was charged twice, please fix it now!", qs, level)
        assert set(res) == {"route", "refund", "u"}
        assert all(d.level == level for d in res.values())
        assert res["route"].meta["revision"] == model.revision
    assert res["route"].meta["n_forward"] == 3
    if model.capable:
        assert res["route"].answer == "billing"


def test_rejects_unknown_attn_implementation() -> None:
    from brier.backends.hf import HFBackend

    with pytest.raises(BrierError):
        HFBackend(MODELS["qwen3"].id, revision=MODELS["qwen3"].revision, attn_implementation="x")


# ---------- M4.1b L1 end to end ----------


def test_decider_l1_fit_and_decide(backend) -> None:  # type: ignore[no-untyped-def]
    from brier import Decider

    q = Noul("Is the customer asking for a refund?", name="refund")
    states = [f"Please refund order {i}." if i % 2 else f"Where is parcel {i}?" for i in range(50)]
    labels = [bool(i % 2) for i in range(50)]
    d = Decider(backend)
    d.fit_temperature(states, q, labels)
    l0 = d.decide("I want my money back.", [q], "L0")["refund"]
    l1 = d.decide("I want my money back.", [q], "L1")["refund"]
    assert l1.level == "L1"
    assert l1.meta["temperature"] > 0
    assert l1.answer == l0.answer


# ---------- M4.3 save / load ----------


def test_decider_save_load_round_trip(backend, tmp_path) -> None:  # type: ignore[no-untyped-def]
    from brier import Decider

    q = Noul("Is the customer asking for a refund?", name="refund")
    states = [f"Please refund order {i}." if i % 2 else f"Where is parcel {i}?" for i in range(50)]
    d = Decider(backend)
    d.fit_prior(states, [q])
    d.fit_temperature(states, q, [bool(i % 2) for i in range(50)])
    d.save(tmp_path / "calib")
    loaded = Decider.load(tmp_path / "calib", backend)
    probe = ["I want my money back.", "Track my order please."]
    for level in ("L0", "L1"):
        a = [r["refund"].probs for r in d.decide_batch(probe, [q], level)]
        b = [r["refund"].probs for r in loaded.decide_batch(probe, [q], level)]
        assert a == b


# ---------- M5.2b L2 end to end ----------


def test_decider_l2_fit_decide_and_round_trip(backend, tmp_path) -> None:  # type: ignore[no-untyped-def]
    from itertools import product

    from brier import Decider

    q = Noul("Is the customer asking for a refund?", name="refund")
    # Varied phrasings: templated states are trivially separable and teach the template.
    refund = [
        f"{a} {b}"
        for a, b in product(
            ["I'd like", "Please give me", "Can I get", "I want", "Send me"],
            [
                "a refund for my order.",
                "my money back.",
                "a reimbursement.",
                "the charge reversed.",
                "a full refund.",
                "my payment returned.",
            ],
        )
    ]
    other = [
        f"{a} {b}"
        for a, b in product(
            [
                "Where is",
                "Has anyone shipped",
                "What's the status of",
                "When will I get",
                "Can you track",
            ],
            [
                "my parcel?",
                "my order?",
                "the package?",
                "my delivery?",
                "the item?",
                "my shipment?",
            ],
        )
    ]
    states = refund + other
    d = Decider(backend)
    d.fit_head(states, q, [True] * len(refund) + [False] * len(other))
    probe = ["I want my money back.", "Track my order please."]
    res = d.decide_batch(probe, [q], "L2")
    assert all(r["refund"].level == "L2" for r in res)
    assert res[0]["refund"].p_yes > res[1]["refund"].p_yes  # refund vs tracking
    d.save(tmp_path / "calib")
    loaded = Decider.load(tmp_path / "calib", backend)
    assert [r["refund"].probs for r in loaded.decide_batch(probe, [q], "L2")] == [
        r["refund"].probs for r in res
    ]
