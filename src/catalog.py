"""Direct, deterministic access to the product catalog: the model asks, code answers from the data.

Product questions are about exact attributes (price, inseam, fabric, "which have no side seams"), which
retrieval can miss, truncate or blur. These tools filter the structured records instead, so a list is complete
and every value comes straight from the file. Every result carries an `id` that the model cites, and
`total_matches` so a truncated list can never be mistaken for a complete one.
"""
import json
from pathlib import Path

from .bm25 import tokenize

DROPPED = {"is_final_sale_eligible"}  # contradicts the order data; Final Sale depends on the discount (policy returns §6)
SUMMARY_FIELDS = ["sku", "name", "gender", "category", "subcategory", "fabric", "price_usd", "fit", "rise",
                  "inseam", "inseam_by_size", "features", "best_for", "restocking_frequency", "is_bestseller", "material"]

TOOLS = [
    {"name": "get_product",
     "description": "Look up a product by SKU or by name and return its full record (price, sizes, colors, fit, inseam, features, care, restocking, description). "
                    "A name matches every product containing all the words, so 'Goddess Ribbed' can return several products: check which one the customer means.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string", "description": "A SKU such as W5766R-BLK, or words from the product name."}},
                      "required": ["query"]}},
    {"name": "search_products",
     "description": "Filter the catalog and return matching product summaries. All filters are optional and combine with AND. "
                    "Use it for lists, comparisons, 'cheapest', recommendations and 'which products have X'. "
                    "Check total_matches: if it exceeds the number returned, the list is incomplete.",
     "input_schema": {"type": "object", "properties": {
         "gender": {"type": "string", "description": "Women, Men or Unisex"},
         "category": {"type": "string", "description": "e.g. Leggings, Sports Bras, Shorts, Pants, Tops, Accessories"},
         "subcategory": {"type": "string"},
         "fabric": {"type": "string", "description": "Fabric name, e.g. Airlift, Airbrush, Alosoft, Alo Softsculpt, Conquer, Alo Vapor"},
         "best_for": {"type": "string", "description": "Use case, e.g. 'hot yoga', 'training', 'lounging'"},
         "feature": {"type": "string", "description": "A feature or tag, e.g. 'no side seams', 'flatlock', 'liner'"},
         "keyword": {"type": "string", "description": "Words that must all appear somewhere in the product text"},
         "size": {"type": "string", "description": "A size the product must be available in, e.g. XL"},
         "color": {"type": "string", "description": "A color the product must be available in"},
         "min_price": {"type": "number"}, "max_price": {"type": "number"},
         "sort_by": {"type": "string", "enum": ["price_asc", "price_desc"]},
         "limit": {"type": "integer", "description": "Max results, default 10, max 25"}}}},
    {"name": "get_fabric",
     "description": "Describe Alo's fabrics (composition, compression, finish, key properties, care). Omit name to list all fabrics, "
                    "which is how to answer questions such as 'the softest fabric'.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}}},
]


def _has(values: list[str] | None, wanted: str) -> bool:
    return any(wanted.lower() in str(v).lower() for v in (values or []))


class Catalog:
    def __init__(self, path: str | Path = "data/products/alo_product_catalog.json"):
        data = json.loads(Path(path).read_text())
        self.fabrics = data["fabric_glossary"]
        self.products = {p["sku"]: p for p in data["products"]}

    # ---- views ----
    def _fabric_view(self, name: str) -> dict:
        g = self.fabrics[name]
        return {"id": f"fabric:{name}", "name": name, **g}

    def _full(self, p: dict) -> dict:
        out = {"id": p["sku"], **{k: v for k, v in p.items() if k not in DROPPED}}
        if p.get("fabric") in self.fabrics:
            g = self.fabrics[p["fabric"]]
            out["fabric_details"] = {"composition": g["composition"], "compression_level": g["compression_level"]}
        return out

    def _summary(self, p: dict) -> dict:
        out = {"id": p["sku"], **{k: p[k] for k in SUMMARY_FIELDS if k in p and p[k] is not None}}
        if p.get("fabric") in self.fabrics:
            out["compression_level"] = self.fabrics[p["fabric"]]["compression_level"]
        return out

    # ---- tools ----
    def get_product(self, query: str) -> dict:
        q = query.strip()
        hit = next((p for s, p in self.products.items() if s.lower() == q.lower()), None)
        matches = [hit] if hit else [p for p in self.products.values() if set(tokenize(q)) and set(tokenize(q)) <= set(tokenize(p["name"]))]
        return {"query": query, "total_matches": len(matches), "products": [self._full(p) for p in matches]}

    def search_products(self, gender=None, category=None, subcategory=None, fabric=None, best_for=None, feature=None,
                        keyword=None, size=None, color=None, min_price=None, max_price=None, sort_by=None, limit=10) -> dict:
        def ok(p: dict) -> bool:
            return all([
                gender is None or p["gender"].lower() == gender.lower(),
                category is None or category.lower() in p["category"].lower(),
                subcategory is None or subcategory.lower() in p["subcategory"].lower(),
                fabric is None or (p.get("fabric") or "").lower() == fabric.lower(),
                best_for is None or _has(p.get("best_for"), best_for),
                feature is None or _has(p.get("features"), feature) or _has(p.get("tags"), feature),
                keyword is None or set(tokenize(keyword)) <= set(tokenize(json.dumps({k: v for k, v in p.items() if k not in ("available_sizes", "available_colors")}))),
                size is None or any(size.lower() == s.lower() for s in p.get("available_sizes", [])),
                color is None or _has(p.get("available_colors"), color),
                min_price is None or p["price_usd"] >= min_price,
                max_price is None or p["price_usd"] <= max_price])
        matches = [p for p in self.products.values() if ok(p)]
        if sort_by:
            matches.sort(key=lambda p: p["price_usd"], reverse=sort_by == "price_desc")
        shown = matches[:max(1, min(int(limit or 10), 25))]
        return {"total_matches": len(matches), "returned": len(shown), "products": [self._summary(p) for p in shown]}

    def get_fabric(self, name: str | None = None) -> dict:
        if name is None:
            return {"fabrics": [self._fabric_view(n) for n in self.fabrics]}
        hit = next((n for n in self.fabrics if n.lower() == name.strip().lower()), None)
        if not hit:
            return {"error": f"no fabric named {name!r}", "available": list(self.fabrics)}
        return {"fabrics": [self._fabric_view(hit)]}

    def execute(self, tool: str, args: dict) -> tuple[dict, list[str]]:
        """Run a tool call. Returns (result, ids of the records returned); a bad call returns an error, never raises."""
        try:
            result = getattr(self, tool)(**args)
        except (AttributeError, TypeError, ValueError) as exc:
            return {"error": f"invalid call to {tool}: {exc}"}, []
        ids = [r["id"] for r in result.get("products", []) + result.get("fabrics", [])]
        return result, ids
