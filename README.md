# verdict

A small open decision model. Give it a state (text or JSON) and typed questions,
get back a calibrated probability for every allowed answer. Nothing is generated,
so the output is always one of the options you defined.

```python
from verdict import Verdict, Noul, Choice, Score

vd = Verdict.load(checkpoint="drramos/verdict")   # Qwen3.5-0.8B + trained weights from Hugging Face
angry, team, urgency = vd.ask(ticket_text, [
    Noul("Is the customer angry?"),                                   # yes / no
    Choice("Which team should handle this?", ["billing", "technical support", "sales"]),
    Score("How urgent is this ticket?", 1, 5),                        # a level on a scale
])
team.value, team.confidence       # 'billing', 0.91
urgency.value, urgency.expected   # 4, 3.8
```

Verdict follows the "System One" decision-model pattern popularized by TypeSafe
AI's Jev, and its question types mirror Jev's (Noul, Choice, Score). It is an
independent project, not affiliated with or endorsed by TypeSafe AI.

## How it works

Interactive architecture diagram, worked example and math:
[claude.ai/artifact/8ns3mSxQ6neq1y2Efagynq](https://claude.ai/artifact/8ns3mSxQ6neq1y2Efagynq)

The head follows vllm-sr's [Decision-1.0-Eos-0.8B](https://huggingface.co/vllm-sr/Decision-1.0-Eos-0.8B).
Every question type is a choice over text options (Noul is yes/no, Score is the
levels of the scale). The prompt is the state, the question type, the question,
each option, then a fixed suffix ending in `Decision:`. It runs through
Qwen3.5-0.8B with LoRA adapters; the language-model head is not used. Instead:

```
c_i = hidden state at the last token of option i
g   = hidden state at the last token of the prompt (it has read everything)
z_i = <W_k LN(c_i), W_q LN(g)> / sqrt(256)  +  w . GELU(W_c LN(c_i) + W_g LN(g))
p(a_i | state, q) = softmax(z / T_type)
```

Trainable: LoRA r=16 on every projection, plus the 1.05M-parameter head (11.3M in
total). One temperature per question type is fitted after training on a separate
calibration split.

## Layout

| Path | |
|---|---|
| `verdict/questions.py` | `Noul`, `Choice`, `Score` and their typed answers |
| `verdict/encoding.py` | builds the prompt, option endpoints and query position; batching |
| `verdict/model.py` | `VerdictModel`: Qwen3.5 backbone + decision head |
| `verdict/api.py` | `Verdict.load(...)`, `Verdict.ask(state, questions)` |
| `verdict/data.py`, `verdict/synthetic.py` | public datasets and synthetic families as typed examples |
| `verdict/train.py` | training loop, per-family metrics with ECE, temperature fitting |
| `scripts/train.py` | the full training run; `scripts/setup_gpu.sh` prepares a CUDA box |
| `scripts/real.py`, `scripts/ask.py` | try the trained model on realistic and single questions |
| `tests/` | logic tests on a tiny random Qwen3.5 (run in seconds) |

```bash
uv sync
uv run pytest
uv run python scripts/real.py        # realistic multi-question scenarios, downloads the weights
```

## Results

Weights: [huggingface.co/drramos/verdict](https://huggingface.co/drramos/verdict).
Trained on one RTX 5090 in 33 minutes: about 21k examples from 14 public datasets
and 4 synthetic families, plus 1,000 synthetic service logs of 1k-16k tokens,
2 epochs. Test accuracy after calibration (200 questions per family):

| Type | Accuracy | ECE |
|---|---|---|
| Choice | 84.0% | 0.018 |
| Noul | 87.5% | 0.029 |
| Score | 70.7% (mean error 0.37 levels) | 0.030 |

| Family | Acc | Family | Acc |
|---|---|---|---|
| banking77 (8 of 77 intents) | 95.0% | multi_nli | 84.5% |
| policy rules ("cannot tell") | 93.0% | boolq | 83.5% |
| massive | 92.5% | emotion | 78.0% |
| ag_news | 90.0% | squad_v2 answerability | 75.0% |
| arithmetic (choice / yes-no) | 99.5% / 98.0% | arc / commonsense_qa | 73.5% / 69.0% |
| paws | 86.5% | race / yelp / stsb | 66.5% / 61.5% / 50.5% |
| long logs, 1k-16k tokens | 100% at every length | sciq / urgency | 100% / 100% |
| **time** (never trained on) | 75.0% | **routing** (never trained on) | 64.0%, overconfident |

Known weaknesses: tone and implied intent in longer business text (it missed
"refund it today or I'm cancelling" as a cancellation threat), confusing who acts
with who is mentioned, team-routing questions pulled toward intent labels, and
date arithmetic over weeks.

Some training datasets carry their own terms, including non-commercial ones (for
example the Yelp review data); check them before commercial use.
