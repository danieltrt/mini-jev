from verdict import Choice, Noul, Score, collate, encode

STATE = '{"user": "ana", "msg": "I was charged twice, give me my money back!"}'
Q = Choice("What does the user want?", ["refund", "cancel", "other"])


def segment(tok, e, i):
    """Text of option i: from the previous option's endpoint up to its own."""
    start = max([p for p in e.endpoints if p < e.endpoints[i]], default=0)
    return tok.decode(e.ids[start + 1 : e.endpoints[i] + 1])


def test_prompt_layout(tok):
    text = tok.decode(encode(tok, STATE, Q).ids)
    assert text.startswith(f"Context:\n{STATE}\n\nTask type: choice\nQuestion:\nWhat does the user want?\nOptions:")
    assert text.endswith("Select the single option best supported by the context and instructions.\nDecision:")


def test_endpoints_close_each_option(tok):
    e = encode(tok, STATE, Q)
    for i, key in enumerate(Q.options):
        seg = segment(tok, e, i)
        assert f'"key":"{key}"' in seg and seg.endswith("</option>")
    assert e.query == len(e.ids) - 1 > max(e.endpoints)


def test_shuffled_order_keeps_endpoints_aligned(tok):
    e = encode(tok, STATE, Q, order=[2, 0, 1])
    assert all(f'"key":"{key}"' in segment(tok, e, i) for i, key in enumerate(Q.options))
    assert e.endpoints[2] < e.endpoints[0] < e.endpoints[1]  # "other" appears first in the text


def test_collate_right_pads(tok):
    items = [encode(tok, STATE, Noul("angry?")), encode(tok, STATE * 3, Score("urgency"))]
    b = collate(items, tok.pad_token_id)
    n0 = len(items[0].ids)
    assert b["input_ids"].shape == (2, len(items[1].ids))
    assert b["attention_mask"][0, :n0].all() and not b["attention_mask"][0, n0:].any()
    assert b["query"].tolist() == [n0 - 1, len(items[1].ids) - 1]
    assert b["valid"].tolist() == [[True, True, False, False, False], [True] * 5]
    assert b["kind"].tolist() == [0, 2]
