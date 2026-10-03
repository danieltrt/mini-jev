"""Synthetic task families with exact labels.

Each generator takes a seeded `random.Random` and returns one Example. The
families cover what the pilot showed the public data does not teach:
arithmetic (with and without the answer in the state), true/false checks
with empty states, and Score questions other than review stars. `time` and
`routing` are held out from training to measure transfer.
"""

from __future__ import annotations

import random
from typing import Callable

from .data import Example
from .questions import Choice, Noul, Score

NAMES = ["Alice", "Bruno", "Chen", "Dana", "Emeka", "Fatima", "Goran", "Hana", "Ivan", "Julia"]
THINGS = ["apples", "books", "coins", "marbles", "stickers", "tickets", "pens", "cards"]


def _arith(rng: random.Random) -> tuple[str, int, int, int]:
    op = rng.choice("+-*")
    if op == "*":
        a, b = rng.randint(2, 12), rng.randint(2, 12)
        return op, a, b, a * b
    a, b = rng.randint(10, 99), rng.randint(2, 60)
    if op == "-":
        a, b = max(a, b), min(a, b)
        return op, a, b, a - b
    return op, a, b, a + b


def _distractors(rng: random.Random, answer: int, k: int = 3) -> list[int]:
    pool = {answer + d for d in (-10, -2, -1, 1, 2, 10)} | {answer * 2, abs(answer - 20)}
    pool.discard(answer)
    return rng.sample(sorted(x for x in pool if x >= 0), k)


def _word_problem(rng: random.Random, op: str, a: int, b: int) -> tuple[str, str]:
    name, thing = rng.choice(NAMES), rng.choice(THINGS)
    if op == "+":
        return f"{name} has {a} {thing}. A friend gives {name} {b} more.", f"How many {thing} does {name} have now?"
    if op == "-":
        return f"{name} has {a} {thing} and gives {b} of them away.", f"How many {thing} does {name} have left?"
    return f"There are {a} boxes with {b} {thing} in each box.", f"How many {thing} are there in total?"


SYMBOL = {"+": "+", "-": "-", "*": "×"}


def arithmetic_choice(rng: random.Random) -> Example:
    op, a, b, ans = _arith(rng)
    options = [ans] + _distractors(rng, ans)
    rng.shuffle(options)
    style = rng.choice(["word", "given", "bare"])
    if style == "word":
        state, text = _word_problem(rng, op, a, b)
    elif style == "given":
        state, text = f"Calculation log: {a} {SYMBOL[op]} {b} = {ans}.", "What is the result of the calculation?"
    else:
        state, text = "", f"What is {a} {SYMBOL[op]} {b}?"
    return Example(state, Choice(text, [str(o) for o in options]), options.index(ans), "arithmetic_choice")


def arithmetic_noul(rng: random.Random) -> Example:
    op, a, b, ans = _arith(rng)
    true = rng.random() < 0.5
    claim = ans if true else _distractors(rng, ans, 1)[0]
    if rng.random() < 0.5:
        state, text = "", f"Is {a} {SYMBOL[op]} {b} equal to {claim}?"
    else:
        state, question = _word_problem(rng, op, a, b)
        name, thing = state.split()[0], question.split()[2]
        text = {"+": f"Does {name} now have {claim} {thing}?", "-": f"Does {name} have {claim} {thing} left?",
                "*": f"Are there {claim} {thing} in total?"}[op]
    return Example(state, Noul(text), 0 if true else 1, "arithmetic_noul")


def _clock(h: int) -> str:
    h %= 24
    return f"{(h - 1) % 12 + 1}:00 {'am' if h < 12 else 'pm'}"


def time_choice(rng: random.Random) -> Example:
    start, dur = rng.randint(6, 20), rng.randint(1, 5)
    ans = start + dur
    wrong = rng.sample([ans - 1, ans + 1, start - dur, ans + 2], 3)
    options = [_clock(h) for h in [ans] + wrong]
    if len(set(options)) < 4:
        return time_choice(rng)
    order = rng.sample(range(4), 4)
    options = [options[i] for i in order]
    vehicle = rng.choice(["train", "bus", "ferry", "flight"])
    state = f"The {vehicle} departs at {_clock(start)} and the journey takes {dur} hour{'s' * (dur > 1)}."
    return Example(state, Choice(f"When does the {vehicle} arrive?", options), order.index(0), "time")


ISSUES = {  # urgency 1..5
    5: ["the production database is down", "no customer can log in", "payments are failing for every customer",
        "we suspect a data breach", "the whole site returns errors"],
    4: ["checkout fails for some customers", "a major feature is broken for an enterprise client",
        "emails to customers stopped sending", "the mobile app crashes on launch for many users"],
    3: ["report exports are very slow", "the dashboard shows an intermittent error",
        "search results are sometimes out of date", "one integration is failing for a single customer"],
    2: ["there is a typo on the pricing page", "a button is misaligned on mobile",
        "the help article links to an old page", "a customer asks how to change their avatar"],
    1: ["a customer says thanks for the great service", "someone asks about the newsletter schedule",
        "a feature request for a dark mode", "a question about next year's conference"],
}
URGENCY_QUESTIONS = ["How urgent is this ticket?", "How quickly does this ticket need attention?",
                     "Rate the urgency of this request."]


def urgency(rng: random.Random) -> Example:
    level = rng.randint(1, 5)
    who = rng.choice(NAMES)
    tone = rng.choice(["", " Please advise.", " Thanks.", " Can someone look?"])
    state = f"Support ticket from {who}: {rng.choice(ISSUES[level])}.{tone}"
    return Example(state, Score(rng.choice(URGENCY_QUESTIONS), 1, 5), level - 1, "urgency")


TEAMS = {
    "billing": ["I was charged twice this month", "my invoice shows the wrong amount", "I need a copy of my receipt",
                "my card payment was declined"],
    "technical": ["the app crashes when I open settings", "I get an error when uploading files",
                  "sync stopped working on my laptop", "the page never finishes loading"],
    "returns": ["the item arrived damaged", "I want to send back the shoes I ordered", "I received the wrong size",
                "how do I return a gift"],
    "account": ["I forgot my password", "I want to change the email on my account", "please delete my account",
                "I can't verify my phone number"],
}


def routing(rng: random.Random) -> Example:
    team = rng.choice(sorted(TEAMS))
    options = rng.sample(sorted(TEAMS), 4)
    state = f"Message from {rng.choice(NAMES)}: \"{rng.choice(TEAMS[team])}.\""
    return Example(state, Choice("Which team should handle this message?", options), options.index(team), "routing")


# ---------- rules over a state ----------

def _date(day_of_year: int) -> str:
    import datetime
    return (datetime.date(2026, 1, 1) + datetime.timedelta(days=day_of_year - 1)).strftime("%B %-d")


def policy(rng: random.Random) -> Example:
    """A policy plus facts; the answer is yes, no, or "cannot tell" when a needed fact is missing.
    Three kinds: a return window (with a final-sale exception), a free-shipping threshold, an age limit."""
    kind = rng.choice(["return", "shipping", "age"])
    missing = rng.random() < 0.25
    if kind == "return":
        window, today = rng.choice([14, 30, 60]), rng.randint(120, 300)
        delivered = today - rng.randint(1, 2 * window)
        final_sale = rng.random() < 0.2
        facts = [f"Today is {_date(today)}."]
        if not missing:
            facts.insert(0, f"The order was delivered on {_date(delivered)}.")
        if final_sale:
            facts.append("The item is marked final sale.")
        rules = f"Items can be returned within {window} days of delivery. Final-sale items cannot be returned."
        question = "Can the customer still return this item?"
        answer = "no" if final_sale else "cannot tell" if missing else "yes" if today - delivered <= window else "no"
    elif kind == "shipping":
        threshold = rng.choice([25, 50, 75, 100])
        total = round(rng.uniform(0.3, 2.0) * threshold, 2)
        facts = [] if missing else [f"The order total is ${total:.2f}."]
        facts.append(f"The order contains {rng.randint(1, 6)} items.")
        rules = f"Orders over ${threshold} ship for free."
        question = "Does this order ship for free?"
        answer = "cannot tell" if missing else "yes" if total > threshold else "no"
    else:
        limit, year = rng.choice([16, 18, 21]), rng.randint(1990, 2015)
        facts = [] if missing else [f"The applicant was born in {year}."]
        facts.append(f"The applicant lives in {rng.choice(['Lisbon', 'Seoul', 'Austin', 'Nairobi', 'Oslo'])}.")
        rules = f"Applicants must turn at least {limit} this year (2026) to open an account."
        question = "Can this applicant open an account?"
        answer = "cannot tell" if missing else "yes" if 2026 - year >= limit else "no"
    options = rng.sample(["yes", "no", "cannot tell"], 3)
    state = f"Policy: {rules}\nFacts: {' '.join(facts)}"
    return Example(state, Choice(question, options), options.index(answer), "policy")


TRAIN_FAMILIES: dict[str, Callable[[random.Random], Example]] = {
    "arithmetic_choice": arithmetic_choice,
    "arithmetic_noul": arithmetic_noul,
    "urgency": urgency,
    "policy": policy,
}
HELD_OUT_FAMILIES: dict[str, Callable[[random.Random], Example]] = {
    "time": time_choice,
    "routing": routing,
}


def generate(family: Callable[[random.Random], Example], n: int, seed: int) -> list[Example]:
    rng = random.Random(seed)
    return [family(rng) for _ in range(n)]


# ---------- long states ----------

SERVICES = ["auth", "billing", "search", "gateway", "storage", "mailer", "scheduler", "metrics"]
NOISE = ["request served in {ms} ms", "cache hit for key user:{n}", "health check ok", "worker {n} idle",
         "flushed {n} records", "connection pool size {n}", "rotated log file", "config reloaded"]
LONG_BUCKETS = (1_000, 2_000, 4_000, 8_000, 16_000)   # approximate state length in tokens
TOKENS_PER_LINE = 32   # measured with the Qwen3.5 tokenizer


def _log_line(rng: random.Random, t: int, service: str, level: str, message: str) -> str:
    return f"2026-10-03T{t // 3600 % 24:02d}:{t // 60 % 60:02d}:{t % 60:02d}Z {level:5} [{service}] {message}"


def long_log(rng: random.Random, tokens: int | None = None) -> Example:
    """A long service log with exactly one ERROR line, at a random depth.
    Asks either which service failed (Choice) or whether a given service failed (Noul)."""
    tokens = tokens or rng.choice(LONG_BUCKETS)
    n = max(10, tokens // TOKENS_PER_LINE)
    failing = rng.choice(SERVICES)
    error_at = rng.randrange(n)
    t, lines = rng.randrange(86_400), []
    for i in range(n):
        t += rng.randint(1, 20)
        if i == error_at:
            lines.append(_log_line(rng, t, failing, "ERROR", rng.choice(["disk full", "timeout talking to db", "out of memory",
                                                                        "certificate expired"])))
        else:
            msg = rng.choice(NOISE).format(ms=rng.randint(2, 900), n=rng.randint(1, 9999))
            lines.append(_log_line(rng, t, rng.choice(SERVICES), rng.choice(["INFO", "INFO", "DEBUG", "WARN"]), msg))
    state = "\n".join(lines)
    family = f"long_log_{tokens // 1000}k"
    if rng.random() < 0.5:
        options = rng.sample([s for s in SERVICES if s != failing], 3) + [failing]
        rng.shuffle(options)
        return Example(state, Choice("Which service logged an ERROR?", options), options.index(failing), family)
    asked = failing if rng.random() < 0.5 else rng.choice([s for s in SERVICES if s != failing])
    return Example(state, Noul(f"Did the {asked} service log an ERROR?"), 0 if asked == failing else 1, family)
