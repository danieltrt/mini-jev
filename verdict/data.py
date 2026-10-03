"""Public datasets as typed (state, question, gold) examples."""

from __future__ import annotations

import random
from dataclasses import dataclass

from datasets import load_dataset

from .questions import Choice, Noul, Question, Score

MAX_STATE_CHARS = 1000  # ~250 tokens


@dataclass(frozen=True)
class Example:
    state: str
    question: Question
    gold: int  # index into question.options
    family: str = ""


def _rows(name: str, split: str, n: int, seed: int, config: str | None = None, keep=lambda r: True):
    ds = load_dataset(name, config, split=split).shuffle(seed=seed)
    return [r for _, r in zip(range(n), filter(keep, ds))]


def boolq(split: str, n: int, seed: int = 0) -> list[Example]:
    return [
        Example(r["passage"][:MAX_STATE_CHARS], Noul(r["question"].capitalize() + "?"), 0 if r["answer"] else 1, "boolq")
        for r in _rows("google/boolq", split, n, seed)
    ]


def sciq(split: str, n: int, seed: int = 0) -> list[Example]:
    rng = random.Random(seed)
    out = []

    def options_of(r):
        return [r["correct_answer"], r["distractor1"], r["distractor2"], r["distractor3"]]

    def keep(r):  # some rows have no paragraph or repeat a distractor
        return r["support"].strip() and len(set(options_of(r))) == 4

    for r in _rows("allenai/sciq", split, n, seed, keep=keep):
        options = options_of(r)
        rng.shuffle(options)
        out.append(Example(r["support"][:MAX_STATE_CHARS], Choice(r["question"], options), options.index(r["correct_answer"]), "sciq"))
    return out


def yelp(split: str, n: int, seed: int = 0) -> list[Example]:
    q = Score("How many stars did the reviewer give?", 1, 5)
    return [Example(r["text"][:MAX_STATE_CHARS], q, r["label"], "yelp") for r in _rows("Yelp/yelp_review_full", split, n, seed)]


def _lettered(name: str, config: str | None, family: str, split: str, n: int, seed: int) -> list[Example]:
    """Multiple-choice sets with {"label": [...], "text": [...]} choices and an answer key; no state."""
    def keep(r):
        t = r["choices"]["text"]
        return r["answerKey"] in r["choices"]["label"] and len(set(t)) == len(t) >= 2

    return [
        Example("", Choice(r["question"], r["choices"]["text"]), r["choices"]["label"].index(r["answerKey"]), family)
        for r in _rows(name, split, n, seed, config, keep)
    ]


def commonsense_qa(split: str, n: int, seed: int = 0) -> list[Example]:
    return _lettered("tau/commonsense_qa", None, "commonsense_qa", split, n, seed)


def arc(split: str, n: int, seed: int = 0) -> list[Example]:
    half = n // 2
    return (_lettered("allenai/ai2_arc", "ARC-Easy", "arc", split, n - half, seed)
            + _lettered("allenai/ai2_arc", "ARC-Challenge", "arc", split, half, seed))


def squad_v2(split: str, n: int, seed: int = 0) -> list[Example]:
    """Is the question answerable from the passage? Half yes, half no."""
    want = {True: n - n // 2, False: n // 2}   # answerable / not
    picked = {True: [], False: []}
    for r in load_dataset("rajpurkar/squad_v2", split=split).shuffle(seed=seed):
        answerable = bool(r["answers"]["text"])
        if len(picked[answerable]) < want[answerable]:
            picked[answerable].append(r)
        if len(picked[True]) + len(picked[False]) == n:
            break
    return [
        Example(r["context"][:MAX_STATE_CHARS], Noul(f'Can this question be answered from the text: "{r["question"]}"?'),
                0 if answerable else 1, "squad_v2")
        for answerable, rows in picked.items() for r in rows
    ]


def _intents(name: str, config: str | None, family: str, split: str, n: int, seed: int, k: int = 8) -> list[Example]:
    """Intent classification with many labels: each question offers the right intent and k - 1 others."""
    ds = load_dataset(name, config, split=split)
    labels = sorted(set(ds["label_text"]))
    rng = random.Random(seed)
    out = []
    for r in ds.shuffle(seed=seed).select(range(min(n, len(ds)))):
        gold = r["label_text"].replace("_", " ")
        options = [gold] + [l.replace("_", " ") for l in rng.sample([l for l in labels if l != r["label_text"]], k - 1)]
        rng.shuffle(options)
        out.append(Example(r["text"], Choice("Which intent best matches this message?", options), options.index(gold), family))
    return out


def banking77(split: str, n: int, seed: int = 0) -> list[Example]:
    return _intents("mteb/banking77", None, "banking77", split, n, seed)


def massive(split: str, n: int, seed: int = 0) -> list[Example]:
    return _intents("mteb/amazon_massive_intent", "en", "massive", split, n, seed)


def ag_news(split: str, n: int, seed: int = 0) -> list[Example]:
    topics = ["world", "sports", "business", "science and technology"]
    q = Choice("What is this article about?", topics)
    return [Example(r["text"][:MAX_STATE_CHARS], q, r["label"], "ag_news") for r in _rows("fancyzhx/ag_news", split, n, seed)]


def multi_nli(split: str, n: int, seed: int = 0) -> list[Example]:
    return [
        Example(r["premise"][:MAX_STATE_CHARS],
                Choice(f'Based on the text, is this statement true: "{r["hypothesis"].strip()}"', ["true", "undetermined", "false"]),
                r["label"], "multi_nli")
        for r in _rows("nyu-mll/multi_nli", split, n, seed, keep=lambda r: r["label"] in (0, 1, 2))
    ]


def paws(split: str, n: int, seed: int = 0) -> list[Example]:
    return [
        Example(f"Sentence A: {r['sentence1']}\nSentence B: {r['sentence2']}", Noul("Do sentences A and B mean the same thing?"),
                0 if r["label"] == 1 else 1, "paws")
        for r in _rows("google-research-datasets/paws", split, n, seed, config="labeled_final")
    ]


def race(split: str, n: int, seed: int = 0) -> list[Example]:
    return [
        Example(r["article"][: 2 * MAX_STATE_CHARS], Choice(r["question"], r["options"]), "ABCD".index(r["answer"]), "race")
        for r in _rows("ehovy/race", split, n, seed, config="all", keep=lambda r: len(set(r["options"])) == 4)
    ]


def stsb(split: str, n: int, seed: int = 0) -> list[Example]:
    q = Score("How similar are the two sentences in meaning, from 0 (unrelated) to 5 (equivalent)?", 0, 5)
    return [
        Example(f"Sentence A: {r['sentence1']}\nSentence B: {r['sentence2']}", q, int(round(r["label"])), "stsb")
        for r in _rows("nyu-mll/glue", split, n, seed, config="stsb", keep=lambda r: r["label"] >= 0)
    ]


def emotion(split: str, n: int, seed: int = 0) -> list[Example]:
    q = Choice("Which emotion does the writer express?", ["sadness", "joy", "love", "anger", "fear", "surprise"])
    return [Example(r["text"], q, r["label"], "emotion") for r in _rows("dair-ai/emotion", split, n, seed, config="split")]


PUBLIC = {  # family -> (loader, train split, eval split)
    "boolq": (boolq, "train", "validation"),
    "sciq": (sciq, "train", "validation"),
    "race": (race, "train", "validation"),
    "squad_v2": (squad_v2, "train", "validation"),
    "banking77": (banking77, "train", "test"),
    "massive": (massive, "train", "validation"),
    "ag_news": (ag_news, "train", "test"),
    "multi_nli": (multi_nli, "train", "validation_matched"),
    "paws": (paws, "train", "validation"),
    "yelp": (yelp, "train", "test"),
    "emotion": (emotion, "train", "validation"),
    "stsb": (stsb, "train", "validation"),
    "arc": (arc, "train", "validation"),
    "commonsense_qa": (commonsense_qa, "train", "validation"),
}


def mix(n_per_kind: int, train: bool, seed: int = 0) -> list[Example]:
    """Pilot mix: equal parts Noul (BoolQ), Choice (SciQ) and Score (Yelp stars)."""
    out = []
    for family in ("boolq", "sciq", "yelp"):
        loader, train_split, eval_split = PUBLIC[family]
        out += loader(train_split if train else eval_split, n_per_kind, seed)
    return out
