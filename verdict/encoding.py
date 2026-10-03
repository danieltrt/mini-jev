"""Turn (state, question) into one token sequence, in the Eos / Decision prompt layout.

    Context:\\n{state}\\n\\nTask type: {kind}\\nQuestion:\\n{text}\\nOptions:
    \\n<option>\\n{"description":...,"key":...}\\n</option>      <- endpoint c_i: last token of each option
    ...
    \\n\\nSelect the single option best supported by the context and instructions.\\nDecision:
                                                                <- query g: last token

Parts are tokenized separately and concatenated, so every position is known exactly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

import torch

from .questions import Question

PREFIX = "Context:\n{state}\n\nTask type: {kind}\nQuestion:\n{text}\nOptions:"
OPTION = "\n<option>\n{}\n</option>"
SUFFIX = "\n\nSelect the single option best supported by the context and instructions.\nDecision:"


@dataclass(frozen=True)
class Encoded:
    ids: list[int]
    endpoints: list[int]   # last token of each option, in the question's option order
    kind: int

    @property
    def query(self) -> int:
        return len(self.ids) - 1


def _option(key: str) -> str:
    return OPTION.format(json.dumps({"description": key, "key": key}, ensure_ascii=False, separators=(",", ":")))


def encode(tok, state: str, q: Question, order: Sequence[int] | None = None) -> Encoded:
    """`order` sets the order options appear in the text (for shuffling in
    training); `endpoints` always follows `q.options`, so labels stay aligned."""
    ids: list[int] = []
    add = lambda text: ids.extend(tok(text, add_special_tokens=False).input_ids)

    add(PREFIX.format(state=state, kind=type(q).__name__.lower(), text=q.text))
    endpoints = [0] * len(q.options)
    for i in order if order is not None else range(len(q.options)):
        add(_option(q.options[i]))
        endpoints[i] = len(ids) - 1
    add(SUFFIX)
    return Encoded(ids, endpoints, q.kind)


def collate(items: Sequence[Encoded], pad_id: int) -> dict[str, torch.Tensor]:
    """Right-pad sequences and options into a batch."""
    B, L, K = len(items), max(len(e.ids) for e in items), max(len(e.endpoints) for e in items)
    batch = {
        "input_ids": torch.full((B, L), pad_id),
        "attention_mask": torch.zeros(B, L, dtype=torch.long),
        "endpoints": torch.zeros(B, K, dtype=torch.long),
        "valid": torch.zeros(B, K, dtype=torch.bool),
        "query": torch.tensor([e.query for e in items]),
        "kind": torch.tensor([e.kind for e in items]),
    }
    for b, e in enumerate(items):
        batch["input_ids"][b, : len(e.ids)] = torch.tensor(e.ids)
        batch["attention_mask"][b, : len(e.ids)] = 1
        batch["endpoints"][b, : len(e.endpoints)] = torch.tensor(e.endpoints)
        batch["valid"][b, : len(e.endpoints)] = True
    return batch
