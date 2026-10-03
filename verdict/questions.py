"""Typed questions and answers.

Every question type is a choice over text options; types differ only in
which options they offer and how the resulting distribution is reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Generic, Sequence, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Answer(Generic[T]):
    value: T
    probs: dict[T, float]

    @property
    def confidence(self) -> float:
        return self.probs[self.value]


@dataclass(frozen=True)
class ScoreAnswer(Answer[int]):
    @property
    def expected(self) -> float:
        return sum(v * p for v, p in self.probs.items())


@dataclass(frozen=True)
class Question:
    text: str
    kind: ClassVar[int]

    def values(self) -> tuple:
        """Typed values, aligned with `options`."""
        raise NotImplementedError

    @property
    def options(self) -> tuple[str, ...]:
        """The option strings the model reads."""
        return tuple(str(v) for v in self.values())

    def decode(self, probs: Sequence[float]) -> Answer:
        p = dict(zip(self.values(), probs))
        return Answer(max(p, key=p.__getitem__), p)


@dataclass(frozen=True)
class Noul(Question):
    kind: ClassVar[int] = 0

    def values(self) -> tuple[bool, bool]:
        return (True, False)

    @property
    def options(self) -> tuple[str, str]:
        return ("yes", "no")


@dataclass(frozen=True)
class Choice(Question):
    choices: tuple[str, ...] = ()
    kind: ClassVar[int] = 1

    def __post_init__(self):
        object.__setattr__(self, "choices", tuple(self.choices))
        if len(self.choices) < 2 or len(set(self.choices)) != len(self.choices):
            raise ValueError(f"Choice needs at least two distinct options, got {self.choices!r}")

    def values(self) -> tuple[str, ...]:
        return self.choices


@dataclass(frozen=True)
class Score(Question):
    lo: int = 1
    hi: int = 5
    kind: ClassVar[int] = 2

    def __post_init__(self):
        if self.hi <= self.lo:
            raise ValueError(f"Score needs lo < hi, got {self.lo}..{self.hi}")

    def values(self) -> tuple[int, ...]:
        return tuple(range(self.lo, self.hi + 1))

    def decode(self, probs: Sequence[float]) -> ScoreAnswer:
        a = super().decode(probs)
        return ScoreAnswer(a.value, a.probs)


KINDS = (Noul, Choice, Score)
