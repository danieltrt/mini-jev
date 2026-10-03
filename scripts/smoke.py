"""Real-model smoke test: load Qwen3.5-0.8B on this machine, answer typed
questions (untrained, so answers are near-random), time inference and one
training step.

    uv run python scripts/smoke.py
"""

import time

import torch
import torch.nn.functional as F

from verdict import Choice, Noul, Score, Verdict, collate, encode

STATE = """{"ticket": 812, "user": "ana@example.com", "plan": "pro",
 "messages": ["The app crashes every time I log in since yesterday's update.",
              "I have a client demo in 2 hours, this is really bad!!"]}"""
QUESTIONS = [
    Noul("Is the user angry?"),
    Choice("Which area is affected?", ["auth", "billing", "ui", "other"]),
    Score("How urgent is this ticket?"),
]


def sync(device):
    if device == "mps":
        torch.mps.synchronize()


def timed(fn, device, n=5):
    fn(); sync(device)                       # warm-up
    t = time.perf_counter()
    for _ in range(n):
        out = fn()
    sync(device)
    return out, (time.perf_counter() - t) / n * 1000


def main():
    t = time.perf_counter()
    vd = Verdict.load()
    dev = vd.device
    print(f"loaded on {dev} in {time.perf_counter() - t:.1f}s")
    trainable = sum(p.numel() for p in vd.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in vd.model.parameters())
    print(f"params: {total / 1e6:.0f}M total, {trainable / 1e6:.1f}M trainable")

    answers, ms = timed(lambda: vd.ask(STATE, QUESTIONS), dev)
    print(f"\nask(): {len(QUESTIONS)} questions in {ms:.0f} ms (untrained, expect near-uniform)")
    for q, a in zip(QUESTIONS, answers):
        probs = ", ".join(f"{k}: {v:.2f}" for k, v in a.probs.items())
        print(f"  {type(q).__name__:6} {q.text:30} -> {a.value!r:8} conf {a.confidence:.2f}  [{probs}]")

    vd.model.train()
    b = {k: v.to(dev) for k, v in collate([encode(vd.tok, STATE, q) for q in QUESTIONS], vd.tok.pad_token_id).items()}
    gold = torch.tensor([0, 0, 4], device=dev)
    opt = torch.optim.AdamW([p for p in vd.model.parameters() if p.requires_grad], lr=1e-4)

    def step():
        loss = F.nll_loss(vd.model(**b), gold)
        opt.zero_grad(); loss.backward(); opt.step()
        return loss.item()

    losses = [step() for _ in range(3)]
    _, ms = timed(step, dev, n=3)
    print(f"\ntrain step (batch 3, {b['input_ids'].shape[1]} tokens): {ms:.0f} ms, loss {losses[0]:.3f} -> {losses[-1]:.3f}")
    if dev == "mps":
        print(f"mps memory: {torch.mps.driver_allocated_memory() / 2**30:.1f} GB")


if __name__ == "__main__":
    main()
