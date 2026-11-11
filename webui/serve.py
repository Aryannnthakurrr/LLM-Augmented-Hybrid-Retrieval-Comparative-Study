"""Results viewer for the completed study.

Serves a read-only web UI over the artefacts the study already produced:
results.json, data/queries.json, data/products.json, the judge cache
(.cache/judge) and the enriched index metadata (.cache/index_enriched).
Nothing is recomputed and no LLM or embedding model is loaded, so it runs
with the standard library alone:

    python webui/serve.py            # http://127.0.0.1:8000
    python webui/serve.py --port 8080 --no-browser
    python webui/serve.py --export results-viewer.html   # single offline file
"""

from __future__ import annotations

import argparse
import json
import re
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent

ENRICHED_MARKER = "User might search for:"


def _read_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _load_judge_rationales() -> dict[str, dict[str, str]]:
    """query_id -> product name -> judge rationale (results.json stores them blank)."""
    rationales: dict[str, dict[str, str]] = {}
    judge_dir = ROOT / ".cache" / "judge"
    if not judge_dir.is_dir():
        return rationales
    for path in sorted(judge_dir.glob("judge_*.json")):
        match = re.match(r"judge_(Q\d+)_", path.name)
        if not match:
            continue
        items = _read_json(path)
        if isinstance(items, list):
            rationales[match.group(1)] = {
                item.get("product_name", ""): item.get("rationale", "")
                for item in items
                if isinstance(item, dict)
            }
    return rationales


def _load_enriched_queries() -> list[list[str]]:
    """product_index -> LLM-generated user-style queries from the enriched index."""
    meta_path = ROOT / ".cache" / "index_enriched" / "metadata.json"
    if not meta_path.is_file():
        return []
    texts = _read_json(meta_path).get("document_texts", [])
    out: list[list[str]] = []
    for text in texts:
        _, _, tail = text.partition(ENRICHED_MARKER)
        out.append([line.strip()[2:] for line in tail.splitlines() if line.strip().startswith("- ")])
    return out


def build_bundle() -> dict:
    results = _read_json(ROOT / "results.json")
    products = _read_json(ROOT / "data" / "products.json")
    rationales = _load_judge_rationales()
    enriched = _load_enriched_queries()

    # Fill in the judge rationales that results.json left empty
    for pipeline in results["pipelines"].values():
        for query in pipeline["per_query"]:
            by_name = rationales.get(query["query_id"], {})
            for judged in query["judge_scores"]:
                if not judged.get("rationale"):
                    judged["rationale"] = by_name.get(judged["product_name"], "")

    # Only ship product details for products that appear in some result list
    used = {
        r["product_index"]
        for pipeline in results["pipelines"].values()
        for query in pipeline["per_query"]
        for r in query["results"]
    }
    product_details = {}
    for idx in sorted(used):
        p = dict(products[idx])
        p.pop("description", None)  # already carried on each result row
        p["enriched_queries"] = enriched[idx] if idx < len(enriched) else []
        product_details[str(idx)] = p

    return {
        "metadata": results["metadata"],
        "pipelines": results["pipelines"],
        "products": product_details,
    }


def export_standalone(out_path: Path) -> None:
    """Write index.html with the bundle embedded, so it opens without the server."""
    bundle = build_bundle()
    products = _read_json(ROOT / "data" / "products.json")
    # Each listing description is repeated on up to 3 x 10 result rows; keep one copy per product
    for idx, product in bundle["products"].items():
        product["description"] = products[int(idx)]["description"]
    for pipeline in bundle["pipelines"].values():
        for query in pipeline["per_query"]:
            for result in query["results"]:
                result.pop("description", None)

    data = json.dumps(bundle, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    marker = '<script>\n"use strict";'
    if marker not in html:
        raise SystemExit("Could not find the script block in webui/index.html")
    html = html.replace(marker, f'<script type="application/json" id="embedded-data">{data}</script>\n{marker}', 1)
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, bundle_bytes: bytes, **kwargs):
        self._bundle_bytes = bundle_bytes
        super().__init__(*args, **kwargs)

    def do_GET(self):  # noqa: N802 (http.server naming)
        if self.path.split("?", 1)[0] == "/api/data":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(self._bundle_bytes)))
            self.end_headers()
            self.wfile.write(self._bundle_bytes)
            return
        super().do_GET()

    def log_message(self, format, *args):  # keep the console quiet
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the study results viewer.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--export", type=Path, metavar="FILE",
                        help="write a self-contained HTML file instead of serving")
    args = parser.parse_args()

    if args.export:
        export_standalone(args.export)
        return

    bundle_bytes = json.dumps(build_bundle(), ensure_ascii=False).encode("utf-8")
    handler = partial(Handler, directory=str(STATIC_DIR), bundle_bytes=bundle_bytes)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Results viewer running at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
