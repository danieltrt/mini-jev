import torch
import torch.nn.functional as F

from verdict import Choice, Noul, Score, collate, encode

STATE = "ticket #812: customer says the app crashes on login since yesterday's update"
QS = [Noul("Is this a bug report?"), Choice("Area?", ["auth", "billing", "ui", "other"]), Score("Urgency")]


def batch(tok, qs=QS, state=STATE):
    return collate([encode(tok, state, q) for q in qs], tok.pad_token_id)


def test_head_matches_eos_parameter_count():
    from verdict.model import CandidateHead
    assert sum(p.numel() for p in CandidateHead(1024).parameters()) == 1_053_184


def test_untrained_output_is_near_uniform(tiny, tok):
    p = tiny(**batch(tok)).exp()
    for row, k in zip(p, (2, 4, 5)):
        assert (row[:k] - 1 / k).abs().max() < 0.05


def test_output_is_a_distribution_over_real_options(tiny, tok):
    logp = tiny(**batch(tok))
    assert logp.shape == (3, 5)
    assert torch.allclose(logp.exp().sum(-1), torch.ones(3))
    assert torch.isinf(logp[0, 2:]).all() and torch.isfinite(logp[0, :2]).all()   # Noul: 2 options
    assert torch.isinf(logp[1, 4:]).all() and torch.isfinite(logp[1, :4]).all()   # Choice: 4 options


def test_padding_does_not_change_answers(tiny, tok):
    """A question answered alone == the same question in a batch with a longer sequence."""
    tiny.eval()
    with torch.no_grad():
        alone = tiny(**batch(tok, QS[:1]))[0, :2]
        mixed = tiny(**collate([encode(tok, STATE, QS[0]), encode(tok, STATE * 5, QS[2])], tok.pad_token_id))[0, :2]
    assert torch.allclose(alone, mixed, atol=1e-5)


def test_only_lora_and_head_are_trainable(tiny):
    names = [n for n, p in tiny.named_parameters() if p.requires_grad]
    assert names and all("lora_" in n or n.startswith("head.") for n in names)
    assert "log_T" not in dict(tiny.named_parameters())


def test_temperature_flattens_the_distribution(tiny, tok):
    tiny.eval()
    b = batch(tok)
    with torch.no_grad():
        tiny.head.scalar.weight.normal_()  # make the untrained output non-uniform
        sharp = tiny(**b).exp()
        tiny.log_T[:] = torch.log(torch.tensor(100.0))
        flat = tiny(**b).exp()
    assert (flat.max(-1).values <= sharp.max(-1).values + 1e-6).all()


def test_can_learn_a_toy_task(tiny, tok):
    """Overfit a tiny dataset: gradients reach the right places and the loss can go to ~0."""
    data = [
        ("the light is red", Choice("Light colour?", ["red", "green"]), 0),
        ("the light is green", Choice("Light colour?", ["red", "green"]), 1),
        ("the user is furious", Noul("Is the user angry?"), 0),
        ("the user is happy", Noul("Is the user angry?"), 1),
    ]
    b = collate([encode(tok, s, q) for s, q, _ in data], tok.pad_token_id)
    gold = torch.tensor([g for *_, g in data])
    opt = torch.optim.AdamW([p for p in tiny.parameters() if p.requires_grad], lr=3e-3)
    first = None
    for _ in range(60):
        loss = F.nll_loss(tiny(**b), gold)
        first = first if first is not None else loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.1 * first
