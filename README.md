# LLM-Augmented Hybrid Retrieval for Informal Queries

Where does an LLM help most in search: on the documents, or on the query?

In a specialised marketplace, sellers describe products in technical language and buyers search in everyday
words. Someone typing *"something strong for the floor"* will not match a listing titled *"hand-cleaned kiln-sound
solid brick batch with compressive strength certification"*. This project measures two ways of closing that gap
with an LLM, on the same hybrid search backend:

- **Enrich the documents.** Before indexing, an LLM writes six short queries a buyer might type for each
  listing, and they are indexed with it. The LLM runs once per listing, offline, and never at search time.
- **Rewrite the query.** At search time, an LLM turns the vague query into a more specific one.

The enrichment idea is HyDE turned around: instead of writing a hypothetical document for each query at search
time, write hypothetical queries for each document at indexing time. It is close to doc2query.

**Result:** enriching documents gave almost all of the gain, at no search-time cost. Rewriting queries on top of
that added about 1% on average and made more queries worse than better.

## Pipelines

| # | Pipeline | What happens | LLM call at search time |
|---|----------|--------------|:---:|
| 1 | `basic_hybrid` | Query → index of the raw listings | No |
| 2 | `enriched_hybrid` | Query → index of listings plus their LLM-written buyer queries | No |
| 3 | `rewrite_enriched` | LLM-rewritten query → the same enriched index | Yes |

All three share one backend: BM25 (sparse) and FAISS over BGE-large-en-v1.5 embeddings (dense), fused by a
weighted sum of min-max normalised scores (0.4 BM25, 0.6 dense).

## Results

1,000 synthetic listings, 50 deliberately vague queries (3.9 words on average), top 10 results per pipeline.
Every retrieved listing was scored 1 to 5 by an LLM judge.

| Pipeline | Mean relevance (1–5) | Results scored 4 or 5 | MRR | NDCG@5 | NDCG@10 |
|----------|---------------------:|----------------------:|----:|-------:|--------:|
| `basic_hybrid` | 2.956 | 37.4% | 0.8069 | 0.7864 | 0.9003 |
| `enriched_hybrid` | 3.246 | 47.2% | 0.8869 | 0.7814 | 0.8866 |
| `rewrite_enriched` | 3.280 | 48.4% | 0.8717 | 0.7901 | 0.8980 |

- **Enrichment:** mean relevance up 9.8% and MRR up 9.9% over plain hybrid search.
- **Rewriting on top of enrichment:** mean relevance up a further 1.0%, MRR slightly down.
- **NDCG barely moves.** It measures how well each pipeline orders its own results, and all three use the same
  ranking. The gain from enrichment is in which listings are retrieved, not in their order.
  [`docs/evaluation-explained.md`](docs/evaluation-explained.md) works through this.

Query by query, on mean relevance:

| Comparison | Better | Worse | Same |
|------------|-------:|------:|-----:|
| Enriched vs plain hybrid | 23 | 12 | 15 |
| Rewrite + enriched vs enriched | 18 | 20 | 12 |

Rewriting helps when the query is genuinely vague and hurts when it is already specific, because the rewrite adds
constraints the buyer never stated:

| Query | Plain | Enriched | Rewrite + enriched | What happened |
|-------|------:|---------:|-------------------:|---------------|
| "salvaged windows" | 1.0 | 3.6 | 3.0 | Plain search found nothing relevant; the enriched index did |
| "material for a fence" | 1.6 | 1.4 | 2.9 | The rewrite filled in what a fence needs |
| "pipe offcuts for plumbing" | 3.9 | 3.7 | 1.6 | The rewrite narrowed it to "PVC or copper … residential plumbing systems" |

**Conclusion:** enrich documents by default, and rewrite a query only when it is detected to be ambiguous.

## Browse the results

The results viewer shows the aggregate metrics, score distributions, each query's rewrite, and the three
pipelines' top 10 side by side with the judge's reason for every score.

- **Offline:** open [`webui/results-viewer.html`](webui/results-viewer.html) in a browser. It is a single file
  with all the data inside it.
- **From the repository:**

  ```bash
  python webui/serve.py
  ```

  This serves the viewer at http://127.0.0.1:8000 from `results.json`, `data/` and `.cache/`. It uses only the
  Python standard library, so nothing needs installing.

## How it was measured

- **Catalogue:** 1,000 synthetic construction-waste listings, 20 in each of 50 categories, generated from a fixed
  specification.
- **Queries:** 50 short, informal queries written by an LLM to read like a non-expert buyer.
- **Judging:** for each query, the top 10 from all three pipelines are pooled and de-duplicated, and each listing
  is scored once. Every pipeline is then scored against the same judgments.
- **Metrics:** mean judge score, MRR (first result scoring 3 or more), NDCG@5 and NDCG@10.

| Role | Model |
|------|-------|
| Catalogue generation | Gemini 2.5 Flash |
| Enrichment, query rewriting, query generation, judging | Gemini 2.5 Flash Lite (Vertex AI) |
| Embeddings | BAAI/bge-large-en-v1.5 |

About 1,100 LLM calls in total. Every response is cached in `.cache/llm/`, so a re-run replays the same
enrichments, rewrites and judgments and gives identical results.

## Limits

- The catalogue and the queries are synthetic. Real listings and real buyers will be messier.
- The judge is an LLM from the same model family that wrote the queries and enrichments, which can favour
  LLM-style phrasing. There are no human relevance labels.
- 50 queries is a small set, and no significance tests were run. A difference as small as the 1.0% from rewriting
  is indicative only.
- One domain was tested.

## Run it

You need Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

```bash
uv run python -m cwm_research validate-db
```

```bash
uv run python -m cwm_research run
```

The run needs Google Cloud credentials (`gcloud auth application-default login`) with Vertex AI enabled, and
the embedding model (about 1.3 GB, downloaded on first use). With the committed cache it makes only two small
calls to check which models are available, replays everything else from disk, and reproduces `results.json`,
`results-summary.md` and `methodology.md`. Delete `.cache/` for a completely fresh run.

To look at the results, none of this is needed: use the viewer above.

## Dataset

[`kaggle_dataset/`](kaggle_dataset) holds the whole test collection as CSV and JSON: the catalogue, the queries,
the 6,000 enrichment queries, the 50 rewrites, 1,046 graded relevance judgments with reasons, and the 1,500
ranked results. `python scripts/export_kaggle_dataset.py` rebuilds it from the files in this repository.

## Documentation

| Document | Contents |
|----------|----------|
| [`docs/writeup.md`](docs/writeup.md) | The full write-up: background, method, per-query analysis, failure modes, references |
| [`docs/evaluation-explained.md`](docs/evaluation-explained.md) | How each metric was computed, with worked examples, and why NDCG stays flat |
| [`methodology.md`](methodology.md) | Method report generated by the run |
| [`results-summary.md`](results-summary.md) | All 50 queries across the three pipelines |

## Repository layout

```
src/cwm_research/    The study: indexing, enrichment, rewriting, retrieval, judging, reporting
webui/               Results viewer (serve.py, index.html) and its single-file export
data/                products.json (1,000 listings) and queries.json (50 queries)
kaggle_dataset/      The test collection as CSV and JSON
scripts/             Dataset export
docs/                Write-up and evaluation notes
.cache/              Cached LLM responses, judgments and built indexes
results.json         Every result, score and judge reason
```

## License

MIT. See [LICENSE](LICENSE).
