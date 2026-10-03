import pytest

from verdict import Choice, Noul, Score, ScoreAnswer


def test_noul_decodes_to_bool():
    a = Noul("is the user angry?").decode([0.8, 0.2])
    assert a.value is True and a.probs == {True: 0.8, False: 0.2} and a.confidence == 0.8


def test_choice_decodes_to_option():
    a = Choice("intent?", ["refund", "cancel", "other"]).decode([0.1, 0.7, 0.2])
    assert a.value == "cancel" and a.confidence == 0.7


def test_score_decodes_with_expectation():
    a = Score("urgency", 1, 3).decode([0.25, 0.25, 0.5])
    assert isinstance(a, ScoreAnswer) and a.value == 3
    assert a.expected == pytest.approx(2.25)


def test_options_are_text():
    assert Noul("q").options == ("yes", "no")
    assert Score("q", 0, 2).options == ("0", "1", "2")


@pytest.mark.parametrize("bad", [["only"], ["a", "a"]])
def test_choice_rejects_bad_options(bad):
    with pytest.raises(ValueError):
        Choice("q", bad)
