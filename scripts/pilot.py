"""Tiny pilot: does the model learn at all? ~5-10 minutes on an M3 Pro.

    caffeinate -i uv run python scripts/pilot.py
"""

import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from verdict.api import default_device
from verdict.data import mix
from verdict.model import BASE, VerdictModel
from verdict.train import evaluate, train

N_TRAIN, N_VAL = 200, 100  # per question type
CHANCE = {"Noul": 1 / 2, "Choice": 1 / 4, "Score": 1 / 5}


def report(title: str, metrics: dict) -> None:
    print(f"\n{title}")
    print(f"  {'type':7} {'acc':>6} {'chance':>7} {'nll':>6} {'conf':>6} {'mae':>6}")
    for kind in ("Noul", "Choice", "Score"):
        m = metrics[kind]
        mae = f"{m['mae']:.2f}" if "mae" in m else ""
        print(f"  {kind:7} {m['accuracy']:6.1%} {CHANCE[kind]:7.0%} {m['nll']:6.3f} {m['confidence']:6.2f} {mae:>6}")


def main():
    device = default_device()
    t = time.perf_counter()
    train_set, val_set = mix(N_TRAIN, train=True), mix(N_VAL, train=False)
    print(f"data: {len(train_set)} train / {len(val_set)} val  ({time.perf_counter() - t:.0f}s)")

    tok = AutoTokenizer.from_pretrained(BASE)
    model = VerdictModel.from_pretrained().to(device)

    report("before training", evaluate(model, tok, val_set, device=device))
    train(model, tok, train_set, device=device, epochs=1, batch_size=8, lr=2e-4, log_every=5)
    report("after training", evaluate(model, tok, val_set, device=device))

    out = Path("checkpoints/pilot.pt")
    out.parent.mkdir(exist_ok=True)
    torch.save(model.trainable_state(), out)
    print(f"\nsaved trainable weights -> {out}  (total {(time.perf_counter() - t) / 60:.1f} min)")


if __name__ == "__main__":
    main()
