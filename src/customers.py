"""Direct access to structured customer records. No embeddings: the record is looked up by the
*session's* customer id, never by anything in the query text, so a question that names someone
else ("what did Jordan Kim order?") cannot change whose data is loaded."""
import json
from datetime import date
from pathlib import Path

NOT_FOR_LLM = {"email", "payment_method", "delivery_address"}  # PII the model never needs


class CustomerStore:
    def __init__(self, path: str | Path = "data/customers/customer_order_history.json"):
        self._by_id = {c["customer_id"]: c for c in json.loads(Path(path).read_text())["customers"]}

    def get(self, customer_id: str | None) -> dict | None:
        return self._by_id.get(customer_id)

    def render(self, customer_id: str | None, as_of: date) -> str | None:
        """Compact text for the prompt, or None if there is no such session customer.

        Adds days_since_order / days_since_delivery so the model never counts calendar days. It does NOT
        apply the return window: that rule comes from the retrieved policy, so policy edits need no code change."""
        c = self.get(customer_id)
        if c is None:
            return None
        lines = [f"Customer {c['customer_id']} ({c['name']}) | tier {c['loyalty_tier']} | {c['loyalty_points']} points "
                 f"| member since {c['member_since']} | today is {as_of.isoformat()}"]
        for o in c["orders"]:
            lines.append(f"\nOrder {o['order_id']} | placed {o['order_date']} ({(as_of - date.fromisoformat(o['order_date'])).days} days ago)"
                         f" | status: {o['order_status']} | channel: {o['channel']}")
            if o.get("delivered_date"):
                lines[-1] += f" | delivered {o['delivered_date']} ({(as_of - date.fromisoformat(o['delivered_date'])).days} days ago)"
            for k, v in o.items():
                if k in ("order_id", "order_date", "order_status", "channel", "items", "delivered_date") or k in NOT_FOR_LLM or v is None:
                    continue
                if k == "return_initiated_date":  # a plain duration, so the model need not count days itself
                    gap = (date.fromisoformat(v) - date.fromisoformat(o["order_date"])).days
                    v = f"{v} ({gap} days after order" + (
                        f", {(date.fromisoformat(v) - date.fromisoformat(o['delivered_date'])).days} days after delivery)"
                        if o.get("delivered_date") else ")")
                lines.append(f"  {k}: {v}")
            for i in o["items"]:
                sale = f", {i['discount_pct']}% off" if i["was_discounted"] else ""
                lines.append(f"  - {i['quantity']}x {i['name']} ({i['sku']}) {i['color']} size {i['size']} "
                             f"@ ${i['unit_price']}{sale}{' [FINAL SALE]' if i['final_sale'] else ''}")
        return "\n".join(lines)
