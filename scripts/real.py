"""Realistic states with several typed questions each, the way an app would call the model.

    uv run python scripts/real.py [drramos/verdict or a local .pt/.safetensors file]
"""

import sys

from verdict import Choice, Noul, Score, Verdict

SCENARIOS = [
    ("Support email",
     """From: maria.lopez@gmail.com
Subject: Charged TWICE again??
I just checked my bank statement and you charged my card twice for the March subscription.
This is the second time this happens. Refund the extra charge today or I'm cancelling my account.""",
     [Noul("Is the customer angry?"),
      Choice("What does the customer want?", ["a refund", "technical help", "to change their plan", "to give feedback"]),
      Choice("Which team should handle this?", ["billing", "technical support", "sales", "shipping"]),
      Score("How urgent is this ticket?", 1, 5),
      Noul("Does the customer threaten to cancel?")]),

    ("Pull request",
     """Title: Fix crash when cart is empty
This PR adds a null check in CheckoutService.computeTotal() so checkout no longer crashes
when the cart has no items. Added two unit tests for the empty-cart case. No database changes.""",
     [Noul("Does this PR include tests?"),
      Noul("Does this PR change the database?"),
      Choice("What kind of change is this?", ["bug fix", "new feature", "refactor", "documentation"]),
      Score("How risky is this change to deploy?", 1, 5)]),

    ("Server log",
     """14:02:11 INFO  [api] GET /orders 200 41ms
14:02:12 INFO  [api] GET /orders 200 39ms
14:02:13 ERROR [payments] connection refused: stripe.com:443
14:02:14 ERROR [payments] connection refused: stripe.com:443
14:02:15 INFO  [api] GET /health 200 3ms""",
     [Noul("Is any service failing?"),
      Choice("Which service has errors?", ["api", "payments", "auth", "search"]),
      Choice("What is the likely cause?", ["external provider unreachable", "bad user input", "out of memory", "disk full"])]),

    ("Chat moderation",
     "you are a complete idiot and nobody wants you here, just leave",
     [Noul("Is this message abusive?"),
      Choice("How should this message be handled?", ["allow", "warn the user", "remove the message"]),
      Choice("Which emotion does the writer express?", ["joy", "anger", "fear", "sadness"])]),

    ("Product review",
     "Battery lasts two days and the screen is gorgeous. Shipping took a week longer than promised though.",
     [Score("How many stars did the reviewer give?", 1, 5),
      Noul("Does the reviewer mention a problem with delivery?"),
      Choice("What does the reviewer like most?", ["battery and screen", "price", "customer service", "packaging"])]),

    ("Meeting notes",
     """Sync 3 Oct. Attendees: Ana, Ben, Chloe.
Ana will send the revised budget to finance by Friday. Ben raised concerns about the vendor contract;
no decision yet, revisit next week. Chloe is out until the 14th.""",
     [Choice("Who owns the budget follow-up?", ["Ana", "Ben", "Chloe", "finance"]),
      Noul("Was a decision made about the vendor contract?"),
      Noul("Is Chloe available next week?")]),

    ("Reservation request",
     "Hi! Could we get a table for six this Saturday around 8pm? One of us is vegan.",
     [Choice("How many people is the reservation for?", ["2", "4", "6", "8"]),
      Noul("Does the group have a dietary requirement?"),
      Choice("What day is the reservation for?", ["Friday", "Saturday", "Sunday", "cannot tell"])]),

    ("Missing information",
     "Policy: Refunds are allowed within 14 days of purchase.\nCustomer: I'd like a refund for my headphones please.",
     [Choice("Is the customer eligible for a refund?", ["yes", "no", "cannot tell"])]),
]


def main():
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else "drramos/verdict"
    vd = Verdict.load(checkpoint=checkpoint)
    for title, state, questions in SCENARIOS:
        print(f"\n━━ {title} ━━")
        print("   " + state.replace("\n", "\n   "))
        for q, a in zip(questions, vd.ask(state, questions)):
            top = sorted(a.probs.items(), key=lambda kv: -kv[1])[:3]
            rest = "  ".join(f"{k}:{v:.2f}" for k, v in top)
            extra = f"  E[s]={a.expected:.1f}" if hasattr(a, "expected") else ""
            print(f"   {type(q).__name__:6} {q.text:52} → {a.value!r:<32} {a.confidence:.2f}{extra}   [{rest}]")


if __name__ == "__main__":
    main()
