"""Qwen3.5 backbone with its LM head replaced by the Eos decision head.

    g   = hidden state at the last token (sees the state, question and every option)
    c_i = hidden state at the last token of option i
    z_i = <W_k c_i, W_q g> / sqrt(256)  +  w . GELU(W_c c_i + W_g g)
    p(a_i | state, q) = softmax(z / T_kind)

Follows vllm-sr's Decision-1.0-Eos-0.8B (CandidateHead), 1,053,184 parameters.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from .questions import KINDS

BASE = "Qwen/Qwen3.5-0.8B-Base"
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",       # gated attention
    "in_proj_qkv", "in_proj_z", "out_proj",       # gated deltanet
    "gate_proj", "up_proj", "down_proj",          # swiglu
]


class CandidateHead(nn.Module):
    def __init__(self, d: int, width: int = 256):
        super().__init__()
        self.width = width
        self.candidate_norm = nn.LayerNorm(d)
        self.query_norm = nn.LayerNorm(d)
        self.key = nn.Linear(d, width, bias=False)
        self.query = nn.Linear(d, width, bias=False)
        self.candidate_mlp = nn.Linear(d, width)
        self.query_mlp = nn.Linear(d, width, bias=False)
        self.scalar = nn.Linear(width, 1, bias=False)
        nn.init.normal_(self.scalar.weight, std=0.01)

    def forward(self, c: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        # c: (B, K, d)   g: (B, d)  ->  logits (B, K)
        c, g = self.candidate_norm(c), self.query_norm(g)
        bilinear = (self.key(c) * self.query(g)[:, None]).sum(-1) / math.sqrt(self.width)
        nonlinear = self.scalar(F.gelu(self.candidate_mlp(c) + self.query_mlp(g)[:, None])).squeeze(-1)
        return bilinear + nonlinear


class VerdictModel(nn.Module):
    def __init__(self, lm: nn.Module):
        super().__init__()
        self.lm = lm  # decoder stack + final norm, no LM head
        self.head = CandidateHead(lm.config.hidden_size)
        self.register_buffer("log_T", torch.zeros(len(KINDS)))  # calibrated after training

    @classmethod
    def from_pretrained(cls, name: str = BASE, lora_rank: int = 16, dtype=torch.bfloat16,
                        gradient_checkpointing: bool = False) -> VerdictModel:
        """`gradient_checkpointing` recomputes activations in the backward pass: several
        times less memory for ~30% more compute, which is what makes long states fit."""
        lm = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).model
        if gradient_checkpointing:
            lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        lora = LoraConfig(r=lora_rank, lora_alpha=2 * lora_rank, target_modules=LORA_TARGETS)
        return cls(get_peft_model(lm, lora))

    def forward(self, input_ids, attention_mask, endpoints, valid, query, kind) -> torch.Tensor:
        H = self.lm(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
        H = H.float()                                          # head runs in FP32, as in Eos
        rows = torch.arange(H.shape[0], device=H.device)
        c = H[rows[:, None], endpoints]                        # (B, K, d)
        g = H[rows, query]                                     # (B, d)
        z = self.head(c, g) / self.log_T.exp()[kind, None]
        return z.masked_fill(~valid, float("-inf")).log_softmax(-1)  # log p(a_i | state, q)

    def trainable_state(self) -> dict[str, torch.Tensor]:
        """What training changes: LoRA adapters, the head, and the fitted temperatures."""
        keep = {n for n, p in self.named_parameters() if p.requires_grad} | {"log_T"}
        return {n: t.detach().cpu() for n, t in self.state_dict().items() if n in keep}
