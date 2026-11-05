"""Minimal local web demo: python -m src.web.app  ->  http://127.0.0.1:8000

Standard library only. The page posts a question (and the signed-in customer, if any) to /ask and shows the
answer, its status, the sources cited, how the question was routed, and exactly what context the model saw.
The API key stays on the server (.env); the page never sees it.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.generation.generate import DEFAULT_AS_OF, Assistant

HERE = Path(__file__).parent
CUSTOMERS = json.loads(Path("data/customers/customer_order_history.json").read_text())["customers"]
EXAMPLES = ["What is the return window for leggings?", "Which fabric is the Airlift legging made from?",
            "Can I return the leggings from my Cyber Monday order?", "What did I buy last season?",
            "What benefits do All Access members get?", "Who won the game last night?"]

assistant, lock = Assistant(), threading.Lock()  # one shared assistant; requests are served one at a time


def ask(query: str, customer_id: str | None) -> dict:
    with lock:
        a = assistant.answer(query, customer_id or None)
    return {"status": a.status, "answer": a.answer, "citations": a.citations, "data_issues": a.data_issues,
            "routed": a.route, "context": a.context, "context_ids": a.context_ids,
            "tool_calls": [{"name": c["name"], "args": c["args"], "returned": c["ids"]} for c in a.tool_calls],
            "ms": round(sum([(a.stats.get("route") or {}).get("ms", 0), a.stats.get("retrieve_ms", 0),
                             (a.stats.get("model") or {}).get("ms", 0), a.stats.get("tool_ms", 0)]))}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, kind: str):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/config":
            cfg = {"as_of": DEFAULT_AS_OF if isinstance(DEFAULT_AS_OF, str) else DEFAULT_AS_OF.isoformat(), "examples": EXAMPLES,
                   "customers": [{"id": c["customer_id"], "name": c["name"], "tier": c["loyalty_tier"]} for c in CUSTOMERS]}
            self._send(200, json.dumps(cfg).encode(), "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/ask":
            return self._send(404, b"not found", "text/plain")
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            query = str(body.get("query", "")).strip()
            if not query:
                return self._send(400, b'{"error": "empty question"}', "application/json")
            self._send(200, json.dumps(ask(query, body.get("customer_id"))).encode(), "application/json")
        except Exception as e:  # show the failure in the page instead of a dead connection
            self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}).encode(), "application/json")

    def log_message(self, *args):  # keep the terminal quiet
        pass


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8000), Handler)  # localhost only: this serves a model with your API key
    print("Alo assistant demo on http://127.0.0.1:8000  (Ctrl+C to stop)")
    server.serve_forever()
