"""Export the synthetic CWM marketplace dataset (and the study's LLM outputs) for Kaggle.

Run from the project root:  python scripts/export_kaggle_dataset.py
Writes CSV/JSON files into kaggle_dataset/. Everything is read from data/, results.json
and the on-disk LLM caches, so no API calls are made.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "kaggle_dataset"
LIST_FIELDS = ("materials", "reuse_applications", "compliance_tags", "environment")


def cache_path(cache_dir: Path, key: str) -> Path:
    """Same naming scheme as VertexLLMClient._cache_path."""
    hashed = hashlib.sha256(key.encode()).hexdigest()[:16]
    safe_key = re.sub(r"[^\w\-]", "_", key)[:60]
    return cache_dir / f"{safe_key}_{hashed}.json"


def write_csv(name: str, fieldnames: list[str], rows: list[dict]) -> None:
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{name}: {len(rows)} rows")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    products = json.loads((ROOT / "data" / "products.json").read_text(encoding="utf-8"))
    queries = json.loads((ROOT / "data" / "queries.json").read_text(encoding="utf-8"))
    results = json.loads((ROOT / "results.json").read_text(encoding="utf-8"))["pipelines"]

    for i, p in enumerate(products, start=1):
        p["product_id"] = f"P{i:04d}"
    id_by_name = {p["name"]: p["product_id"] for p in products}

    # products.json / products.csv
    ordered = [{"product_id": p["product_id"], **{k: v for k, v in p.items() if k != "product_id"}} for p in products]
    (OUT / "products.json").write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = list(ordered[0].keys())
    write_csv("products.csv", fields,
              [{k: "; ".join(v) if k in LIST_FIELDS else v for k, v in p.items()} for p in ordered])

    # queries.csv
    write_csv("queries.csv", ["query_id", "query_text"],
              [{"query_id": q["query_id"], "query_text": q["text"]} for q in queries])

    # enrichment_queries.csv: six LLM-generated user-style queries per product
    rows = []
    for p in products:
        generated = json.loads(cache_path(ROOT / ".cache" / "llm", f"enrich_{p['name']}").read_text(encoding="utf-8"))
        rows += [{"product_id": p["product_id"], "query_number": n, "generated_query": q.strip()}
                 for n, q in enumerate(generated, start=1)]
    write_csv("enrichment_queries.csv", ["product_id", "query_number", "generated_query"], rows)

    # query_rewrites.csv: LLM rewrite of each test query (pipeline 3)
    write_csv("query_rewrites.csv", ["query_id", "query_text", "rewritten_query", "rewrite_reasoning"], [
        {"query_id": r["query_id"], "query_text": r["query_text"],
         "rewritten_query": r["metadata"]["rewritten_query"], "rewrite_reasoning": r["metadata"]["rewrite_reasoning"]}
        for r in results["rewrite_enriched"]["per_query"]])

    # relevance_judgments.csv: pooled LLM judgments (one per query-product pair)
    rows = []
    for q in queries:
        pooled = {r["name"] for pipe in results.values()
                  for run in pipe["per_query"] if run["query_id"] == q["query_id"] for r in run["results"]}
        judged = json.loads(cache_path(ROOT / ".cache" / "judge", f"judge_{q['query_id']}_{len(pooled)}").read_text(encoding="utf-8"))
        for j in judged:
            name = j["product_name"].strip()
            rows.append({"query_id": q["query_id"], "product_id": id_by_name.get(name, ""), "product_name": name,
                         "relevance": max(1, min(5, int(j["score"]))), "rationale": j.get("rationale", "")})
    write_csv("relevance_judgments.csv", ["query_id", "product_id", "product_name", "relevance", "rationale"], rows)

    # retrieval_runs.csv: top-10 ranking of every pipeline with component scores
    rows = []
    for pipeline, data in results.items():
        for run in data["per_query"]:
            scores = {j["product_name"]: j["score"] for j in run["judge_scores"]}
            rows += [{"pipeline": pipeline, "query_id": run["query_id"], "rank": r["rank"],
                      "product_id": id_by_name[r["name"]], "fused_score": r["score"], "bm25_score": r["bm25_score"],
                      "dense_score": r["dense_score"], "relevance": scores.get(r["name"], 1)} for r in run["results"]]
    write_csv("retrieval_runs.csv",
              ["pipeline", "query_id", "rank", "product_id", "fused_score", "bm25_score", "dense_score", "relevance"], rows)


if __name__ == "__main__":
    main()
