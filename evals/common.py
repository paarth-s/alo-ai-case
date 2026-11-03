"""Shared eval helpers: question loading and validation."""
import json
from pathlib import Path

from src.ingestion.ingest import load_all

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = ROOT / "evals" / "questions.jsonl"  # dev set: used while building and tuning
HELDOUT_PATH = ROOT / "evals" / "questions_heldout.jsonl"  # never used for tuning decisions; reported at the end
RESULTS_DIR = ROOT / "evals" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)  # scripts write here; a fresh clone may not have it
AS_OF = "2025-03-01"  # the fixed "today" for every date-dependent answer

DOMAINS = {"product", "policy", "customer", "adversarial"}
DIFFICULTIES = {"easy", "medium", "hard"}
TYPES = {"answerable", "trap", "partial", "refuse", "abstain"}
ROUTES = {"product", "policy", "customer"}
REQUIRED = {"id", "query", "domain", "difficulty", "type", "customer_id", "expected_domains", "gold_anchors",
            "gold_customer", "reference_answer", "must_include", "must_not_include", "failure_modes"}


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate(questions: list[dict], data_dir: Path = ROOT / "data") -> list[str]:
    """Return a list of problems; empty means every label points at something real."""
    chunk_ids = {c.id for c in load_all(data_dir)}
    customers = json.loads((data_dir / "customers" / "customer_order_history.json").read_text())["customers"]
    orders = {o["order_id"] for c in customers for o in c["orders"]}
    cust_ids = {c["customer_id"] for c in customers}

    problems, seen = [], set()
    for q in questions:
        qid = q.get("id", "?")
        if missing := REQUIRED - q.keys():
            problems.append(f"{qid}: missing fields {sorted(missing)}")
            continue
        if qid in seen:
            problems.append(f"{qid}: duplicate id")
        seen.add(qid)
        if q["domain"] not in DOMAINS or q["difficulty"] not in DIFFICULTIES or q["type"] not in TYPES:
            problems.append(f"{qid}: bad domain/difficulty/type")
        problems += [f"{qid}: unknown route {d!r}" for d in q["expected_domains"] if d not in ROUTES]
        problems += [f"{qid}: gold anchor {a!r} is not a chunk id" for a in q["gold_anchors"] if a not in chunk_ids]
        problems += [f"{qid}: gold order {o!r} not in customer data" for o in q["gold_customer"] if o not in orders]
        if q["customer_id"] and q["customer_id"] not in cust_ids:
            problems.append(f"{qid}: unknown customer {q['customer_id']!r}")
        if "customer" in q["expected_domains"] and not q["customer_id"]:
            problems.append(f"{qid}: routes to customer but has no customer_id")
    return problems
