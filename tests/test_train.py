import random

import pytest

from verdict import Choice, Noul, Score
from verdict.data import Example
from verdict.train import batches, evaluate, train

DATA = [
    Example("the light is red", Choice("Light colour?", ["red", "green", "blue"]), 0),
    Example("the light is green", Choice("Light colour?", ["red", "green", "blue"]), 1),
    Example("the user is furious", Noul("Is the user angry?"), 0),
    Example("the user is happy", Noul("Is the user angry?"), 1),
    Example("terrible, never again", Score("Stars?"), 0),
    Example("perfect in every way", Score("Stars?"), 4),
] * 2


def test_batches_cover_every_example_once(tok):
    seen = [g for b in batches(DATA, tok, 4, random.Random(0)) for g in b["gold"].tolist()]
    assert sorted(seen) == sorted(e.gold for e in DATA)


def test_evaluate_reports_every_kind(tiny, tok):
    m = evaluate(tiny, tok, DATA, device="cpu", batch_size=4)
    assert set(m) == {"Noul", "Choice", "Score"}
    assert set(m["Score"]) == {"n", "accuracy", "nll", "confidence", "ece", "mae"}
    assert all(0 <= v["accuracy"] <= 1 and v["nll"] > 0 for v in m.values())


def test_train_improves_nll(tiny, tok):
    before = evaluate(tiny, tok, DATA, device="cpu")
    train(tiny, tok, DATA, device="cpu", epochs=15, batch_size=4, lr=3e-3, log_every=10**9)
    after = evaluate(tiny, tok, DATA, device="cpu")
    for kind in before:
        assert after[kind]["nll"] < before[kind]["nll"], kind


def test_metrics_group_by_family_and_report_ece():
    import torch
    from verdict.train import metrics
    rows = [Example("s", Noul("q"), 0, "a"), Example("s", Noul("q"), 1, "a"), Example("s", Score("q"), 2, "b")]
    logps = [torch.tensor([0.9, 0.1]).log(), torch.tensor([0.9, 0.1]).log(), torch.tensor([0, 0, 1.0, 0, 0]).clamp(min=1e-9).log()]
    m = metrics(rows, logps, key=lambda e: e.family)
    assert m["a"]["accuracy"] == 0.5 and abs(m["a"]["ece"] - 0.4) < 1e-6   # 90% confident, 50% right
    assert m["b"]["accuracy"] == 1.0 and m["b"]["mae"] < 1e-6


def test_fit_temperatures_cools_an_overconfident_model(tiny, tok):
    import torch
    from verdict.train import fit_temperatures
    # always 99% sure of option 0, right only half the time -> T should grow
    rows = [Example("s", Noul("q"), g, "x") for g in (0, 1) * 20]
    with torch.inference_mode():  # predictions may come from inference code
        logps = [torch.tensor([0.99, 0.01]).log()] * len(rows)
    temps = fit_temperatures(tiny, rows, logps)
    assert temps["Noul"] > 3 and tiny.log_T[0] > 1


def test_fit_temperatures_leaves_small_groups_alone(tiny):
    import torch
    from verdict.train import fit_temperatures
    rows = [Example("s", Noul("q"), 0, "x")] * 4
    assert fit_temperatures(tiny, rows, [torch.tensor([0.5, 0.5]).log()] * 4) == {}
    assert (tiny.log_T == 0).all()


def test_token_budget_batches(tok):
    import random
    from verdict.synthetic import long_log
    from verdict.train import plan
    rng = random.Random(0)
    data = DATA + [long_log(rng, 1000) for _ in range(3)]
    enc, groups = plan(data, tok, size=8, rng=random.Random(0), max_tokens=3000)
    assert sorted(i for g in groups for i in g) == list(range(len(data)))         # every example once
    assert all(len(g) <= 8 and max(len(enc[i].ids) for i in g) * len(g) <= 3000 or len(g) == 1 for g in groups)


def test_long_log_labels_are_right():
    import random
    from verdict.synthetic import long_log
    rng = random.Random(0)
    for _ in range(50):
        e = long_log(rng, 1000)
        failing = next(l for l in e.state.splitlines() if " ERROR " in l).split("[")[1].split("]")[0]
        answer = e.question.options[e.gold]
        if isinstance(e.question, Choice):
            assert answer == failing
        else:
            assert (answer == "yes") == (f"the {failing} service" in e.question.text)


def test_gradient_checkpointing_trains(tok):
    import torch
    import torch.nn.functional as F
    from verdict import collate, encode
    from tests.conftest import tiny_model
    m = tiny_model(gradient_checkpointing=True).train()
    b = collate([encode(tok, e.state, e.question) for e in DATA[:4]], tok.pad_token_id)
    F.nll_loss(m(**b), torch.tensor([e.gold for e in DATA[:4]])).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in m.named_parameters() if "lora_" in n)


def test_policy_answers_follow_the_rules():
    import random
    from verdict.synthetic import policy
    rng = random.Random(0)
    for _ in range(300):
        e = policy(rng)
        answer = e.question.options[e.gold]
        facts = e.state.split("Facts: ")[1]
        if "final sale" in facts:
            assert answer == "no"
        elif not any(k in facts for k in ("delivered on", "total is", "born in")):
            assert answer == "cannot tell"
        elif "total is" in facts:
            total = float(facts.split("$")[1].split(" ")[0].rstrip("."))
            threshold = int(e.state.split("over $")[1].split(" ")[0])
            assert answer == ("yes" if total > threshold else "no")
