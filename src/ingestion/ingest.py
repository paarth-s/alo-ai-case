"""Turn the policy files into chunks, one per section.

Only policy prose is chunked and embedded. The product catalog and customer orders are structured data: they are
loaded as-is and queried with exact lookups (src/context/), never searched. Chunk ids double as source anchors for
the eval gold labels (e.g. "returns§6"), so re-chunking never invalidates the question set.
"""
import hashlib
import re
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path

DOC_SLUGS = {
    "returns_and_exchanges_policy": "returns",
    "shipping_policy": "shipping",
    "alo_access_loyalty_program": "loyalty",
}


@dataclass
class Chunk:
    id: str
    domain: str  # "policy" (the only vector domain)
    text: str  # what the LLM sees
    embed_text: str  # what gets embedded (can drop low-signal fields)
    meta: dict = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(f"{self.text}\0{self.embed_text}".encode()).hexdigest()[:16]


# ---------- policies ----------

def _number_and_title(heading: str, parent_num: str | None = None) -> tuple[str, str]:
    """'2.1 Standard Shipping' -> ('2.1', ...). Unnumbered headings get a slug id."""
    m = re.match(r"(\d+(?:\.\d+)*)\.?\s+(.*)", heading)
    if m:
        return m.group(1), m.group(2)
    slug = re.sub(r"[^a-z0-9]+", "-", re.sub(r"\(.*?\)", "", heading).lower()).strip("-")
    return (f"{parent_num}.{slug}" if parent_num else slug), heading


# Sections that are always needed together are one chunk: (doc, target section) -> (sections folded into it, new title).
#  * returns 1 + 1.1: the 30-day rule and the eligibility conditions it comes with
#  * loyalty 4 + 5.*: what each tier needs and what it gets
MERGES = {
    ("returns", "1"): (["1.1"], None),
    ("loyalty", "4"): (["5.vip", "5.a-list", "5.all-access"], "Membership Tiers and Benefits"),
}


def _expand_tiers(tiers: list[tuple[str, list[str]]]) -> list[tuple[str, list[str]]]:
    """Make each tier's benefits stand alone: 'All VIP benefits, plus:' becomes the VIP benefits written out."""
    bullets: dict[str, list[str]] = {}  # tier name -> its complete benefit list
    out = []
    for title, lines in tiers:
        name = re.sub(r"\s*\(.*\)", "", title).strip()  # "A-List (300–999 points)" -> "A-List"
        inherited, own, lead = [], [], "Complete list of benefits:"
        for line in lines:
            m = re.match(r"\s*-\s*All (.+?) benefits, plus:?\s*$", line)
            if m and m.group(1) in bullets:
                inherited, lead = bullets[m.group(1)], f"Includes all {m.group(1)} benefits. Complete list of benefits:"
            elif line.strip():
                own.append(line)
        bullets[name] = inherited + own
        out.append((title, [lead, *bullets[name]]))
    return out


def _merge_sections(slug: str, blocks: list[list]) -> list[list]:
    """Fold the sections named in MERGES into their target; the target keeps its id so references stay valid."""
    by_num = {b[0]: b for b in blocks}
    for (doc, target), (children, new_title) in MERGES.items():
        if doc != slug or target not in by_num or not all(c in by_num for c in children):
            continue
        kids = [(by_num[c][1][-1], by_num[c][2]) for c in children]
        shown = _expand_tiers(kids) if doc == "loyalty" else kids
        body, compact = list(by_num[target][2]), list(by_num[target][2])
        for title, lines in shown:
            body += ["", f"**{title}**", *lines]
        for title, lines in kids:  # the source wording: nothing repeated, so it fits the embedder's window
            compact += ["", f"**{title}**", *lines]
        by_num[target][2] = body
        if compact != body:
            by_num[target].append(compact)  # 4th element: lines to embed, when they differ from the lines shown
        if new_title:
            by_num[target][1] = by_num[target][1][:-1] + [new_title]
        blocks = [b for b in blocks if b[0] not in children]
    return blocks


def doc_slug(stem: str) -> str:
    """'returns_and_exchanges_policy' and 'returns_and_exchanges_policy_2025q2' are two versions of the same policy."""
    for known, slug in DOC_SLUGS.items():
        if stem == known or stem.startswith(known + "_"):
            return slug
    raise ValueError(f"unknown policy file {stem!r}; add its base name to DOC_SLUGS")


def read_version(text: str) -> tuple[str, str]:
    """(version label, ISO effective date). Falls back to the first day of the version's quarter when no date is given."""
    version = re.search(r"\*\*Document Version:\*\*\s*(\S+)", text).group(1)
    m = re.search(r"\*\*Effective Date:\*\*\s*(.+?)\s*$", text, re.M)
    if m:
        try:
            return version, datetime.strptime(m.group(1).strip(), "%B %d, %Y").date().isoformat()
        except ValueError:
            pass
    q = re.match(r"(\d{4})-Q([1-4])", version)
    return version, f"{q.group(1)}-{3 * (int(q.group(2)) - 1) + 1:02d}-01" if q else "1970-01-01"


def policy_chunks(path: Path, archived: bool = False) -> list[Chunk]:
    """Split on headings. Each chunk carries its doc title and heading path so it stands alone.
    The version in force keeps the stable ids (returns§1); an older or not-yet-effective version is archived under
    versioned ids (returns@2025-Q1§1) and excluded from retrieval by default."""
    slug = doc_slug(path.stem)
    text = path.read_text()
    lines = text.splitlines()
    title = lines[0].lstrip("# ").strip()
    version, effective_iso = read_version(text)
    effective = re.search(r"\*\*Effective Date:\*\*\s*(.+?)\s*$", text, re.M)
    prefix = f"{slug}@{version}" if archived else slug
    vmeta = {"doc_version": version, "effective_iso": effective_iso, "archived": archived}

    blocks, h2, cur = [], None, None  # block = [section number, heading path, body lines]
    for line in lines[1:]:
        if line.startswith("### "):
            num, ttl = _number_and_title(line[4:].strip(), h2[0] if h2 else None)
            cur = [num, [h2[1], ttl] if h2 else [ttl], []]
            blocks.append(cur)
        elif line.startswith("## "):
            num, ttl = _number_and_title(line[3:].strip())
            h2, cur = (num, ttl), [num, [ttl], []]
            blocks.append(cur)
        elif cur is not None:
            cur[2].append(line)

    blocks = _merge_sections(slug, blocks)

    chunks = []
    scope = re.search(r"\*\*Applies To:\*\*\s*(.+)", text)
    if scope:  # the title block holds scope facts (e.g. "U.S. only for full benefits")
        full = f"{title} > Scope\n\nApplies to: {scope.group(1).strip()}"
        chunks.append(Chunk(f"{prefix}§scope", "policy", full, full,
                            {"doc": slug, "section": "scope", "heading": "Scope", **vmeta,
                             "effective_date": effective.group(1) if effective else None}))
    clean = lambda ls: "\n".join(l for l in ls if l.strip() != "---" and not l.startswith("*This policy")).strip()
    for block in blocks:
        num, headings, body_lines = block[:3]
        body = clean(body_lines)
        if len(body.split()) < 5:  # a bare parent heading whose content lives in its subsections
            continue
        head = f"{title} > {' > '.join(headings)}\n\n"
        full = head + body
        embedded = head + clean(block[3]) if len(block) > 3 else full  # merged chunks may embed a shorter form than they show
        chunks.append(Chunk(
            id=f"{prefix}§{num}", domain="policy", text=full, embed_text=embedded,
            meta={"doc": slug, "section": num, "heading": " > ".join(headings), **vmeta,
                  "effective_date": effective.group(1) if effective else None},
        ))
    return chunks


def load_all(data_dir: Path, as_of: str | None = None) -> list[Chunk]:
    """All policy chunks. For each policy, the version in force on `as_of` (ISO date; default: the newest) keeps the stable ids and is
    searchable; every other version is archived. A version that is not yet effective is archived until its date arrives."""
    by_doc: dict[str, list[tuple[str, str, Path]]] = {}
    for path in sorted((data_dir / "policies").glob("*.md")):
        version, eff = read_version(path.read_text())
        by_doc.setdefault(doc_slug(path.stem), []).append((eff, version, path))
    chunks = []
    for slug, versions in by_doc.items():
        versions.sort()
        in_force = [v for v in versions if as_of is None or v[0] <= as_of]
        current = (in_force[-1] if in_force else versions[0])[2]  # nothing effective yet: fall back to the oldest rather than none
        for _, _, path in versions:
            chunks += policy_chunks(path, archived=path != current)
    return chunks


def searchable(chunks: list[Chunk]) -> list[Chunk]:
    """What goes into the vector index. Archived policy versions stay in the source files for audit but are never embedded:
    nothing needs to search them, and leaving them out makes a version change cost only the chunks whose text differs."""
    return [c for c in chunks if not c.meta.get("archived")]
