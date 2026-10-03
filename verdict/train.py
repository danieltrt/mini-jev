"""Training loop and evaluation."""

from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterator, Sequence

import torch
import torch.nn.functional as F

from .data import Example
from .encoding import Encoded, collate, encode
from .model import VerdictModel
from .questions import KINDS, Choice, Score


def plan(examples: Sequence[Example], tok, size: int, rng: random.Random | None = None,
         max_tokens: int | None = None) -> tuple[list[Encoded], list[list[int]]]:
    """Encode every example and group them into batches of at most `size` examples and,
    if `max_tokens` is set, at most that many padded tokens (so one batch can be 16 short
    examples or one long one). With `rng`: shuffled, Choice options shuffled in the text,
    similar lengths grouped together. Without: in order of length (for evaluation)."""
    def order(e: Example):
        n = len(e.question.options)
        return rng.sample(range(n), n) if rng and isinstance(e.question, Choice) else None

    enc = [encode(tok, e.state, e.question, order(e)) for e in examples]
    idx = list(range(len(examples)))
    if rng:
        rng.shuffle(idx)
        window = size * 8
        idx = [i for w in range(0, len(idx), window) for i in sorted(idx[w : w + window], key=lambda i: len(enc[i].ids))]
    else:
        idx.sort(key=lambda i: len(enc[i].ids))
    groups, current, longest = [], [], 0
    for i in idx:
        grown = max(longest, len(enc[i].ids))
        if current and (len(current) == size or (max_tokens and grown * (len(current) + 1) > max_tokens)):
            groups.append(current)
            current, grown = [], len(enc[i].ids)
        current.append(i)
        longest = grown
    if current:
        groups.append(current)
    if rng:
        rng.shuffle(groups)
    return enc, groups


def batches(examples: Sequence[Example], tok, size: int, rng: random.Random | None = None,
            max_tokens: int | None = None) -> Iterator[dict]:
    enc, groups = plan(examples, tok, size, rng, max_tokens)
    for g in groups:
        b = collate([enc[i] for i in g], tok.pad_token_id)
        b["gold"] = torch.tensor([examples[i].gold for i in g])
        b["index"] = torch.tensor(g)
        yield b


def _split(b: dict, device) -> tuple[dict, torch.Tensor]:
    b = {k: v.to(device) for k, v in b.items() if k != "index"}
    return b, b.pop("gold")


def train(model: VerdictModel, tok, examples: Sequence[Example], *, device, epochs: int = 1, batch_size: int = 8,
          lr: float = 2e-4, warmup: float = 0.1, log_every: int = 10, seed: int = 0,
          on_epoch_end: Callable[[int], None] | None = None, log_file: Path | None = None,
          max_tokens: int | None = None) -> None:
    rng = random.Random(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    total = epochs * len(plan(examples, tok, batch_size, random.Random(seed), max_tokens)[1])
    n_warm = max(1, int(warmup * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, min((s + 1) / n_warm, (total - s) / max(1, total - n_warm))))

    model.train()
    step, losses, tokens, t0 = 0, [], 0, time.perf_counter()
    for epoch in range(epochs):
        for b in batches(examples, tok, batch_size, rng, max_tokens):
            inputs, gold = _split(b, device)
            loss = F.nll_loss(model(**inputs), gold)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            losses.append(loss.item())
            tokens += int(inputs["attention_mask"].sum())
            if step % log_every == 0 or step == total:
                dt = time.perf_counter() - t0
                row = {"step": step, "total": total, "epoch": epoch + 1, "loss": sum(losses[-log_every:]) / len(losses[-log_every:]),
                       "lr": sched.get_last_lr()[0], "tok_s": tokens / dt, "minutes": dt / 60}
                print(f"step {step:4}/{total}  loss {row['loss']:.3f}  lr {row['lr']:.1e}  {row['tok_s']:.0f} tok/s  {row['minutes']:.1f} min",
                      flush=True)
                if log_file:
                    with open(log_file, "a") as f:
                        f.write(json.dumps(row) + "\n")
        if on_epoch_end:
            on_epoch_end(epoch + 1)
            model.train()


@torch.no_grad()  # not inference_mode: the outputs are reused to fit temperatures
def predict(model: VerdictModel, tok, examples: Sequence[Example], *, device, batch_size: int = 16,
            max_tokens: int | None = None) -> list[torch.Tensor]:
    """log p over each example's own options, in example order."""
    model.eval()
    out: list[torch.Tensor | None] = [None] * len(examples)
    for b in batches(examples, tok, batch_size, max_tokens=max_tokens):
        index = b["index"].tolist()
        inputs, _ = _split(b, device)
        for i, row in zip(index, model(**inputs).float().cpu()):
            out[i] = row[: len(examples[i].question.options)]
    return out


def by_kind(e: Example) -> str:
    return type(e.question).__name__


def metrics(examples: Sequence[Example], logps: Sequence[torch.Tensor], key: Callable[[Example], str] = by_kind,
            bins: int = 10) -> dict[str, dict[str, float]]:
    """Per group: accuracy, NLL, mean confidence, ECE, and (Score) mean |E[s] - gold|."""
    groups = defaultdict(list)
    for e, lp in zip(examples, logps):
        groups[key(e)].append((e, lp))
    report = {}
    for name, rows in groups.items():
        conf = torch.tensor([lp.exp().max().item() for _, lp in rows])
        hit = torch.tensor([float(lp.argmax() == e.gold) for e, lp in rows])
        ece = sum(
            (m.sum() / len(rows)) * (conf[m].mean() - hit[m].mean()).abs()
            for lo in torch.linspace(0, 1, bins + 1)[:-1]
            if (m := (conf > lo) & (conf <= lo + 1 / bins)).any()
        )
        r = {"n": len(rows), "accuracy": hit.mean().item(), "nll": sum(-lp[e.gold].item() for e, lp in rows) / len(rows),
             "confidence": conf.mean().item(), "ece": float(ece)}
        if all(isinstance(e.question, Score) for e, _ in rows):
            r["mae"] = sum(abs((lp.exp() * torch.arange(len(lp))).sum().item() - e.gold) for e, lp in rows) / len(rows)
        report[name] = r
    return dict(sorted(report.items()))


def evaluate(model: VerdictModel, tok, examples: Sequence[Example], *, device, batch_size: int = 16,
             key: Callable[[Example], str] = by_kind) -> dict[str, dict[str, float]]:
    return metrics(examples, predict(model, tok, examples, device=device, batch_size=batch_size), key)


def fit_temperatures(model: VerdictModel, examples: Sequence[Example], logps: Sequence[torch.Tensor],
                     min_examples: int = 30, bounds: tuple[float, float] = (0.25, 10.0)) -> dict[str, float]:
    """One temperature per question type, minimizing NLL on held-out predictions made at T = 1.
    log_softmax(logp / T) == log_softmax(z / T), so the stored log-probs are enough.
    Types with fewer than `min_examples` keep T = 1; T is clamped to `bounds`."""
    lo, hi = math.log(bounds[0]), math.log(bounds[1])
    fitted = {}
    for k, kind in enumerate(KINDS):
        rows = [(lp.detach().clone(), e.gold) for e, lp in zip(examples, logps) if isinstance(e.question, kind)]
        if len(rows) < min_examples:
            continue
        log_t = torch.zeros((), requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

        def nll():
            opt.zero_grad()
            loss = sum(-(lp / log_t.clamp(lo, hi).exp()).log_softmax(-1)[g] for lp, g in rows) / len(rows)
            loss.backward()
            return loss

        opt.step(nll)
        model.log_T[k] = log_t.detach().clamp(lo, hi)
        fitted[kind.__name__] = model.log_T[k].exp().item()
    return fitted
