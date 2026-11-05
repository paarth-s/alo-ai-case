"""Deterministic checks for the customer-record harness (no LLM, runs in under a second).

    python -m evals.suites.customer_eval
"""
import json
import sys
from datetime import date

from src.context.customers import CustomerStore

from evals.common import AS_OF, ROOT, load_questions

DATA = ROOT / "data" / "customers" / "customer_order_history.json"
# Hand-checked against the raw dates: Feb 8 -> Mar 1 is 21 days; Jan 15 -> Mar 1 is 45 days.
EXPECTED_DAYS = {"ALO-2025-010228": 21, "ALO-2025-001145": 45}


def main() -> int:
    store, as_of = CustomerStore(DATA), date.fromisoformat(AS_OF)
    customers = json.loads(DATA.read_text())["customers"]
    owner = {o["order_id"]: c["customer_id"] for c in customers for o in c["orders"]}
    failures = []

    def check(name, ok, detail=""):
        print(f"{'ok  ' if ok else 'FAIL'} {name}{' - ' + detail if detail and not ok else ''}")
        if not ok:
            failures.append(name)

    for q in load_questions():
        if not q["customer_id"]:
            continue
        ctx = store.render(q["customer_id"], as_of)
        check(f"{q['id']} loads session customer's record", ctx is not None and q["customer_id"] in ctx)
        for oid in q["gold_customer"]:
            check(f"{q['id']} gold order {oid} belongs to the session customer and is in context",
                  owner.get(oid) == q["customer_id"] and oid in ctx)
        others = [c for c in customers if c["customer_id"] != q["customer_id"]]
        leaked = [c["customer_id"] for c in others if c["customer_id"] in ctx or c["name"] in ctx
                  or any(o["order_id"] in ctx for o in c["orders"])]
        check(f"{q['id']} context contains no other customer's data", not leaked, f"leaked {leaked}")
        check(f"{q['id']} context omits email / payment details",
              all(c["email"] not in ctx for c in customers)
              and all(o["payment_method"] not in ctx for c in customers for o in c["orders"] if o.get("payment_method")))

    ctx_all = "\n".join(store.render(c["customer_id"], as_of) for c in customers)
    for oid, days in EXPECTED_DAYS.items():
        line = next(l for l in ctx_all.splitlines() if l.startswith(f"Order {oid}"))
        check(f"days_since_order for {oid} is {days}", f"({days} days ago)" in line)
    check("no session customer -> no record", store.render(None, as_of) is None)
    check("unknown customer id -> no record", store.render("CUST-99999", as_of) is None)

    print(f"\n{'all checks passed' if not failures else f'{len(failures)} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
