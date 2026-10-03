"""Ask the same questions to the untrained and the trained model.

    uv run python scripts/ask.py [drramos/verdict or a local .pt/.safetensors file]
"""

import random
import sys

from verdict import Choice, Noul, Score, Verdict
from verdict.synthetic import long_log

LOG = long_log(random.Random(11), 2_000)

CASES = [
    ("Answer is in the state",
     "Receipt: 17 apples + 25 apples = 42 apples total.",
     Choice("How many apples in total?", ["32", "42", "52", "45"])),
    ("Numbers only, model must add",
     "Alice has 17 apples. Bob gives her 25 more.",
     Choice("How many apples does Alice have now?", ["32", "42", "52", "45"])),
    ("Multiplication, no context",
     "",
     Choice("What is 7 times 8?", ["54", "56", "58", "64"])),
    ("Yes/no check (true)",
     "",
     Noul("Is 12 + 9 equal to 21?")),
    ("Yes/no check (false)",
     "",
     Noul("Is 12 + 9 equal to 24?")),
    ("Word problem",
     "A train leaves at 3:00 pm and the trip takes 2 hours.",
     Choice("When does the train arrive?", ["4:00 pm", "5:00 pm", "6:00 pm", "1:00 pm"])),
    ("Out-of-domain Score",
     "Ticket: production database is down, all customers affected, CEO is asking for updates.",
     Score("How urgent is this ticket?")),
    ("Policy, fact missing",
     "Policy: Orders over $50 ship for free.\nFacts: The order contains 3 items.",
     Choice("Does this order ship for free?", ["yes", "no", "cannot tell"])),
    ("Policy, dates",
     "Policy: Items can be returned within 30 days of delivery.\nFacts: The order was delivered on September 20. Today is October 3.",
     Choice("Can the customer still return this item?", ["yes", "no", "cannot tell"])),
    ("Intent routing",
     "Hi, I paid for my order twice and want one of the charges refunded.",
     Choice("Which team should handle this message?", ["billing", "technical support", "returns", "account"])),
    ("Emotion",
     "I finally got the job offer I've been waiting months for!!",
     Choice("Which emotion does the writer express?", ["sadness", "joy", "anger", "fear"])),
    ("Long log (~2k tokens)",
     LOG.state,
     LOG.question),
]


def show(answer) -> str:
    probs = "  ".join(f"{k}:{v:.2f}" for k, v in answer.probs.items())
    return f"{answer.value!r:>10}  conf {answer.confidence:.2f}   [{probs}]"


def main():
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else "drramos/verdict"
    models = {"untrained": Verdict.load(), "trained": Verdict.load(checkpoint=checkpoint)}
    for title, state, q in CASES:
        shown = state if len(state) < 200 else f"{state[:120]}... ({len(state):,} chars)"
        print(f"\n{title}\n  state: {shown or '(empty)'}\n  {type(q).__name__}: {q.text}")
        for name, vd in models.items():
            print(f"  {name:9} -> {show(vd.ask(state, [q])[0])}")


if __name__ == "__main__":
    main()
