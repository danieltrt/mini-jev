import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from verdict.model import BASE, LORA_TARGETS, VerdictModel


@pytest.fixture(scope="session")
def tok():
    return AutoTokenizer.from_pretrained(BASE)


@pytest.fixture
def tiny():
    return tiny_model()


def tiny_model(gradient_checkpointing: bool = False):
    """Same architecture as Qwen3.5-0.8B (3 DeltaNet + 1 attention), shrunk to d=64."""
    torch.manual_seed(0)
    cfg = AutoConfig.from_pretrained(BASE).get_text_config()
    cfg.update(dict(
        hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        linear_num_key_heads=4, linear_num_value_heads=4,
        linear_key_head_dim=16, linear_value_head_dim=16, mtp_num_hidden_layers=0,
    ))
    lm = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32).model
    if gradient_checkpointing:
        lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return VerdictModel(get_peft_model(lm, LoraConfig(r=4, lora_alpha=8, target_modules=LORA_TARGETS)))
