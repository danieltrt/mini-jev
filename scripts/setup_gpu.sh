#!/usr/bin/env bash
# One-time setup on a fresh Linux + NVIDIA GPU box (e.g. a Vast.ai "NVIDIA CUDA" instance).
#
#   git clone https://github.com/danieltrt/verdict.git && cd verdict && bash scripts/setup_gpu.sh
#
# Installs uv and the locked environment, adds flash-linear-attention (fast
# Gated DeltaNet kernels; pure Triton, no compiling), runs the tests, then
# checks that the fast kernels are actually used and measures training speed.
set -euo pipefail

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv sync

# Logic tests run a tiny model on the CPU, so they must run before flash-linear-attention
# is installed: once it is, transformers sends DeltaNet layers to its GPU-only Triton kernels.
uv pip uninstall -q flash-linear-attention fla-core 2>/dev/null || true
uv run pytest -q

uv pip install flash-linear-attention   # causal-conv1d is skipped: no wheels for torch 2.14, ~4 ms gain

uv run python - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not available"
print("torch", torch.__version__, "| cuda", torch.version.cuda, "|", torch.cuda.get_device_name())
EOF

# Fast-kernel check + speed: one real training step must not fall back to the torch DeltaNet path.
uv run python - <<'EOF' 2>&1 | tee /tmp/speed.log
import time, torch
from verdict import VerdictModel
m = VerdictModel.from_pretrained().to("cuda").train()
for B, L in [(16, 256), (16, 512)]:
    ids = torch.randint(0, 200000, (B, L), device="cuda")
    kw = dict(input_ids=ids, attention_mask=torch.ones_like(ids), endpoints=torch.tensor([[L - 30, L - 25, L - 20]] * B, device="cuda"),
              valid=torch.ones(B, 3, dtype=torch.bool, device="cuda"), query=torch.full((B,), L - 1, device="cuda"),
              kind=torch.ones(B, dtype=torch.long, device="cuda"))
    step = lambda: (m(**kw)[:, 0].sum().backward(), torch.cuda.synchronize())
    step(); t = time.perf_counter(); [step() for _ in range(5)]; dt = (time.perf_counter() - t) / 5
    print(f"train step B={B} L={L}: {dt * 1000:.0f} ms  ({B * L / dt:.0f} tok/s)  peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
EOF
if grep -q "chunk_gated_delta_rule.*falling back" /tmp/speed.log; then
  echo "WARNING: flash-linear-attention is NOT being used (slow torch fallback)"; exit 1
fi
echo "setup OK: fast DeltaNet kernels active"
