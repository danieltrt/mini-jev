from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from transformers import AutoTokenizer

from .encoding import collate, encode
from .model import BASE, VerdictModel
from .questions import Answer, Question


def load_checkpoint(checkpoint: str) -> dict[str, torch.Tensor]:
    path = Path(checkpoint)
    if not path.exists() and checkpoint.count("/") == 1:   # a Hugging Face repo id
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(checkpoint, "model.safetensors"))
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(path)
    return torch.load(path)


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


class Verdict:
    """vd.ask(state, [Noul(...), Choice(...), Score(...)]) -> typed answers."""

    def __init__(self, model: VerdictModel, tok, device: str | None = None):
        self.device = device or default_device()
        self.model = model.to(self.device).eval()
        self.tok = tok

    @classmethod
    def load(cls, name: str = BASE, checkpoint: str | None = None, device: str | None = None) -> Verdict:
        """`checkpoint`: trainable weights saved by training (LoRA adapters, head, temperatures):
        a local .pt or .safetensors file, or a Hugging Face repo id such as "user/verdict"."""
        model = VerdictModel.from_pretrained(name)
        if checkpoint:
            missing, unexpected = model.load_state_dict(load_checkpoint(checkpoint), strict=False)
            trainable = {n for n, p in model.named_parameters() if p.requires_grad}
            if unexpected or trainable & set(missing):
                raise ValueError(f"checkpoint does not match model: unexpected={unexpected[:3]}, missing={sorted(trainable & set(missing))[:3]}")
        return cls(model, AutoTokenizer.from_pretrained(name), device)

    @torch.inference_mode()
    def ask(self, state: str, questions: Sequence[Question]) -> list[Answer]:
        batch = collate([encode(self.tok, state, q) for q in questions], self.tok.pad_token_id)
        logp = self.model(**{k: v.to(self.device) for k, v in batch.items()}).cpu()
        return [q.decode(row[: len(q.options)].exp().tolist()) for q, row in zip(questions, logp)]
