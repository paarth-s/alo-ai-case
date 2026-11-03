"""Build docs/review.html: every question from the brief, with its answer and explanation in different colours.

    python docs/build_review.py

The page is generated from the "### Q" blocks in REQUIREMENTS.md, ADR.md and PRODUCTION_MEMO.md, so it cannot drift from them.
Each block has the shape:   ### <id>. <question>   then   **Answer.** ...   then   **Why.** ...
"""
import html
import re
from pathlib import Path

DOCS = Path(__file__).resolve().parent
# (file, section heading to keep or None for all, label shown on the card, group heading in the page)
SOURCES = [
    ("REQUIREMENTS.md", "System: RAG implementation", "System", "1. The RAG system"),
    ("ADR.md", None, "ADR", "2. Architecture decision record"),
    ("REQUIREMENTS.md", "Evaluation framework", "Eval", "3. The evaluation framework"),
    ("PRODUCTION_MEMO.md", None, "Memo", "4. Reliability and production readiness"),
]


def parse(path: Path) -> list[dict]:
    cards, cur, mode, section = [], None, None, ""
    for line in path.read_text().splitlines():
        if line.startswith("## "):
            cur, section = None, line[3:].strip()
        elif line.startswith("### "):
            m = re.match(r"###\s+(\S+?)\.\s+(.*)", line)
            cur = {"id": m.group(1), "q": m.group(2), "answer": [], "why": [], "section": section} if m else None
            mode = None
            if cur:
                cards.append(cur)
        elif cur is not None:
            if line.startswith("**Answer.**"):
                mode, _ = "answer", cur["answer"].append(line[len("**Answer.**"):].strip())
            elif line.startswith("**Why.**"):
                mode = "why"
                rest = line[len("**Why.**"):].strip()
                if rest:
                    cur["why"].append(rest)
            elif mode:
                cur[mode].append(line)
    return cards


def inline(s: str) -> str:
    s = html.escape(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    return s


def blocks(lines: list[str]) -> str:
    """Tiny markdown: paragraphs, '- ' bullets, '1. ' numbered lists."""
    out, para, items, kind = [], [], [], None

    def flush():
        nonlocal para, items, kind
        if para:
            out.append("<p>" + inline(" ".join(para)) + "</p>")
        if items:
            out.append(f"<{kind}>" + "".join(f"<li>{inline(i)}</li>" for i in items) + f"</{kind}>")
        para, items, kind = [], [], None

    for raw in lines:
        line = raw.rstrip()
        bullet, number = re.match(r"^\s*-\s+(.*)", line), re.match(r"^\s*\d+\.\s+(.*)", line)
        if not line.strip():
            flush()
        elif bullet or number:
            want = "ul" if bullet else "ol"
            if para or (items and kind != want):
                flush()
            kind = want
            items.append((bullet or number).group(1))
        elif items and line.startswith("  "):
            items[-1] += " " + line.strip()  # wrapped bullet
        else:
            if items:
                flush()
            para.append(line.strip())
    flush()
    return "".join(out)


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Case Study Review</title>
<style>
:root{--bg:#fbfbfd;--fg:#1c1c21;--mut:#5f6068;--bd:#dcdce3;--card:#fff;
 --q-bg:#e8edff;--q-fg:#1d3a99;--q-bd:#9db0f5;
 --a-bg:#e3f6e9;--a-fg:#0f4d27;--a-bd:#85cfa0;
 --w-bg:#fdf0d6;--w-fg:#5c3d05;--w-bd:#e6c36b}
@media (prefers-color-scheme:dark){:root{--bg:#121216;--fg:#ececf1;--mut:#a0a0ac;--bd:#33333c;--card:#1b1b21;
 --q-bg:#1d2750;--q-fg:#bccaff;--q-bd:#3d5ad0;
 --a-bg:#12331f;--a-fg:#b4f0c9;--a-bd:#2c8a52;
 --w-bg:#392a0a;--w-fg:#f6dca0;--w-bd:#a07a1c}}
*{box-sizing:border-box}
body{margin:0 auto;max-width:880px;padding:0 16px 70px;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,sans-serif}
header{padding:22px 0 8px}h1{font-size:1.5rem;margin:0 0 6px}
.legend{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0}
.chip{border-radius:999px;padding:3px 12px;font-size:.82rem;font-weight:600;border:1px solid}
.chip.q{background:var(--q-bg);color:var(--q-fg);border-color:var(--q-bd)}
.chip.a{background:var(--a-bg);color:var(--a-fg);border-color:var(--a-bd)}
.chip.w{background:var(--w-bg);color:var(--w-fg);border-color:var(--w-bd)}
.bar{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--bd);padding:9px 0;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.bar b{font-variant-numeric:tabular-nums}
button{font:inherit;font-size:.85rem;border:1px solid var(--bd);background:var(--card);color:var(--fg);border-radius:6px;padding:4px 10px;cursor:pointer}
h2{font-size:1.15rem;margin:34px 0 4px;padding-bottom:6px;border-bottom:2px solid var(--bd)}
.card{background:var(--card);border:1px solid var(--bd);border-radius:12px;margin:16px 0;overflow:hidden}
.card.done{opacity:.62}
.q{background:var(--q-bg);color:var(--q-fg);border-bottom:1px solid var(--q-bd);padding:11px 14px;display:flex;gap:10px;align-items:flex-start}
.q .tag{flex:none;font-size:.72rem;font-weight:700;letter-spacing:.04em;border:1px solid var(--q-bd);border-radius:5px;padding:1px 6px;margin-top:3px}
.q .text{font-weight:600}
.q label{flex:none;margin-left:auto;font-size:.78rem;font-weight:500;white-space:nowrap}
.a,.w{padding:11px 14px}.a{background:var(--a-bg);color:var(--a-fg);border-bottom:1px solid var(--a-bd)}
.w{background:var(--w-bg);color:var(--w-fg);font-size:.93rem}
.lab{display:block;font-size:.7rem;font-weight:800;letter-spacing:.08em;margin-bottom:3px;opacity:.85}
.a p,.w p{margin:.25em 0}.w ul,.w ol{margin:.3em 0;padding-left:1.3em}.w li{margin:.2em 0}
code{font:.88em ui-monospace,monospace;background:rgba(127,127,127,.18);padding:1px 5px;border-radius:4px}
.hide .w{display:none}
.links{font-size:.88rem;color:var(--mut)}.links a{color:inherit}
@media print{.bar{display:none}.card{break-inside:avoid}}
</style></head><body>
<header><h1>Case Study Review</h1>
<div class="links">__COUNT__ questions from the brief. Source of truth: <a href="ADR.md">ADR.md</a> · <a href="PRODUCTION_MEMO.md">PRODUCTION_MEMO.md</a> · <a href="REQUIREMENTS.md">REQUIREMENTS.md</a> · <a href="../evals/FAILURE_ANALYSIS.md">FAILURE_ANALYSIS.md</a></div>
<div class="legend"><span class="chip q">Question</span><span class="chip a">Answer</span><span class="chip w">Why: the concise explanation and evidence</span></div></header>
<div class="bar"><span><b id="done">0</b> / <b>__COUNT__</b> reviewed</span><button id="toggle">Hide explanations</button><button id="reset">Reset</button></div>
__BODY__
<script>
const KEY="case-review-v1";let st={};try{st=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
const save=()=>{try{localStorage.setItem(KEY,JSON.stringify(st))}catch(e){}};
const boxes=[...document.querySelectorAll("input[type=checkbox][data-id]")];
function paint(){let n=0;boxes.forEach(b=>{b.checked=!!st[b.dataset.id];b.closest(".card").classList.toggle("done",b.checked);n+=b.checked});document.getElementById("done").textContent=n}
boxes.forEach(b=>b.addEventListener("change",()=>{st[b.dataset.id]=b.checked;save();paint()}));
document.getElementById("toggle").onclick=e=>{const h=document.body.classList.toggle("hide");e.target.textContent=h?"Show explanations":"Hide explanations"};
document.getElementById("reset").onclick=()=>{st={};save();paint()};paint();
</script></body></html>"""


def main() -> None:
    parsed = {name: parse(DOCS / name) for name in {s[0] for s in SOURCES}}
    body, count = [], 0
    for file, section, label, group in SOURCES:
        cards = [c for c in parsed[file] if section is None or c["section"] == section]
        body.append(f"<h2>{html.escape(group)}</h2>")
        for c in cards:
            count += 1
            cid = f"{label}-{c['id']}"
            body.append(
                f'<section class="card"><div class="q"><span class="tag">{html.escape(label)} {html.escape(c["id"])}</span>'
                f'<span class="text">{inline(c["q"])}</span><label><input type="checkbox" data-id="{cid}"> reviewed</label></div>'
                f'<div class="a"><span class="lab">ANSWER</span>{blocks(c["answer"])}</div>'
                f'<div class="w"><span class="lab">WHY</span>{blocks(c["why"])}</div></section>')
    (DOCS / "review.html").write_text(PAGE.replace("__BODY__", "\n".join(body)).replace("__COUNT__", str(count)))
    print(f"wrote docs/review.html with {count} questions")


if __name__ == "__main__":
    main()
