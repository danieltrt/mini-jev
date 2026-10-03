"""Train on the full mix, evaluate per family, fit temperatures.

    uv run python scripts/train.py --scale 0.1 --n-eval 50 --epochs 1 --out checkpoints/pilot          # ~5 min pilot
    uv run python scripts/train.py --long 1000 --max-tokens 16384 --grad-ckpt --epochs 2 --out checkpoints/verdict

Families: 14 public + 4 synthetic for training (see MIX), optional long logs
(1k-16k tokens); 2 synthetic families (time, routing) are never trained on
and only evaluated.
Every family gets separate calibration and test sets.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from verdict.api import default_device
from verdict.data import PUBLIC
from verdict.model import BASE, VerdictModel
from verdict.synthetic import HELD_OUT_FAMILIES, LONG_BUCKETS, TRAIN_FAMILIES, generate, long_log
from verdict.train import by_kind, fit_temperatures, metrics, predict, train


def build_long(n_train: int, n_eval: int):
    """Long service logs (1k-16k tokens): mixed lengths for training, n_eval per length for testing."""
    import random
    rng = random.Random(4000)
    train_set = [long_log(rng) for _ in range(n_train)]
    test = [long_log(rng, k) for k in LONG_BUCKETS for _ in range(n_eval)]
    return train_set, test


MIX = {  # training examples per family at --scale 1 (about 21k)
    "boolq": 1250, "sciq": 1000, "race": 1250,                      # reading a passage
    "squad_v2": 1500,                                               # is it answerable at all?
    "banking77": 1250, "massive": 1250, "ag_news": 1000,            # intent and topic routing
    "multi_nli": 1250, "paws": 1000,                                # entailment, paraphrase
    "yelp": 1250, "emotion": 1000, "stsb": 1000,                    # sentiment, emotion, similarity (ordinal)
    "arc": 1250, "commonsense_qa": 1250,                            # knowledge
    "arithmetic_choice": 800, "arithmetic_noul": 800, "urgency": 600, "policy": 1500,   # synthetic
}


def build(scale: float, n_eval: int, seed: int):
    train_set, calib, test = [], [], []
    for i, (family, count) in enumerate(MIX.items()):
        n = max(1, round(count * scale))
        if family in PUBLIC:
            loader, train_split, eval_split = PUBLIC[family]
            train_set += loader(train_split, n, seed)
            held = loader(eval_split, 2 * n_eval, seed)
        else:
            gen = TRAIN_FAMILIES[family]
            train_set += generate(gen, n, seed=1000 + i)
            held = generate(gen, 2 * n_eval, seed=2000 + i)
        calib, test = calib + held[:n_eval], test + held[n_eval : 2 * n_eval]
    for i, (family, gen) in enumerate(HELD_OUT_FAMILIES.items()):
        test += generate(gen, n_eval, seed=3000 + i)
    return train_set, calib, test


def table(title: str, report: dict) -> str:
    lines = [title, f"  {'group':18} {'n':>5} {'acc':>7} {'nll':>7} {'conf':>6} {'ece':>6} {'mae':>6}"]
    for name, m in report.items():
        mae = f"{m['mae']:.2f}" if "mae" in m else ""
        held = " (held out)" if name in HELD_OUT_FAMILIES else ""
        lines.append(f"  {name:18} {m['n']:5} {m['accuracy']:7.1%} {m['nll']:7.3f} {m['confidence']:6.2f} {m['ece']:6.3f} {mae:>6}{held}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0, help="multiplies the per-family counts in MIX (1.0 = ~21k examples)")
    ap.add_argument("--n-eval", type=int, default=200, help="calibration and test examples per family")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--long", type=int, default=0, help="long-log training examples (1k-16k tokens); 0 = none")
    ap.add_argument("--max-tokens", type=int, default=None, help="padded-token budget per batch (needed with --long)")
    ap.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing (needed with --long)")
    ap.add_argument("--init", type=Path, default=None, help="start from a saved checkpoint, e.g. a previous final.pt")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    device = default_device()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    train_set, calib, test = build(args.scale, args.n_eval, args.seed)
    if args.long:
        long_train, long_test = build_long(args.long, max(10, args.n_eval // 4))
        train_set, test = train_set + long_train, test + long_test
    print(f"device {device} | train {len(train_set)} | calib {len(calib)} | test {len(test)}  ({time.perf_counter() - t0:.0f}s)")

    tok = AutoTokenizer.from_pretrained(BASE)
    model = VerdictModel.from_pretrained(gradient_checkpointing=args.grad_ckpt).to(device)
    if args.init:
        missing, unexpected = model.load_state_dict(torch.load(args.init), strict=False)
        assert not unexpected, unexpected
        model.log_T.zero_()   # temperatures are refit at the end
        print(f"initialized from {args.init}")

    def checkpoint(epoch: int) -> None:
        torch.save(model.trainable_state(), args.out / f"epoch{epoch}.pt")
        print(f"saved {args.out / f'epoch{epoch}.pt'}")

    train(model, tok, train_set, device=device, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
          log_every=20, seed=args.seed, on_epoch_end=checkpoint, log_file=args.out / "train_log.jsonl",
          max_tokens=args.max_tokens)

    family = lambda e: e.family
    test_logp = predict(model, tok, test, device=device, batch_size=2 * args.batch_size, max_tokens=args.max_tokens)
    before = metrics(test, test_logp, family)
    print("\n" + table("test, T = 1", before))

    temps = fit_temperatures(model, calib, predict(model, tok, calib, device=device, batch_size=2 * args.batch_size, max_tokens=args.max_tokens))
    print("\nfitted temperatures: " + ", ".join(f"{k} {v:.2f}" for k, v in temps.items()))
    test_logp = predict(model, tok, test, device=device, batch_size=2 * args.batch_size, max_tokens=args.max_tokens)
    after, kinds = metrics(test, test_logp, family), metrics(test, test_logp, by_kind)
    print("\n" + table("test, calibrated", after))
    print("\n" + table("test by type, calibrated", kinds))

    torch.save(model.trainable_state(), args.out / "final.pt")
    (args.out / "metrics.json").write_text(json.dumps(
        {"args": {k: str(v) for k, v in vars(args).items()}, "temperatures": temps,
         "test_uncalibrated": before, "test_calibrated": after, "test_by_type": kinds}, indent=2))
    print(f"\nsaved {args.out / 'final.pt'} and metrics.json  (total {(time.perf_counter() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
