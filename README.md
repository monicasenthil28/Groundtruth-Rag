# Groundtruth-Rag

## RAG Retrieval Evaluation & Regression Platform

Groundtruth-Rag is a domain-agnostic foundation for comparing retrieval architectures against a human-verified golden dataset. It is intended to make changes to chunking, embeddings, retrieval, and reranking measurable instead of relying on subjective chatbot testing.

The supported comparison is **Dense** vs **Dense + BM25** vs **Dense + BM25 + CrossEncoder**. BM25 is also available as a standalone configuration. A human-reviewed OSHA retrieval benchmark has been run locally; regression baselines preserve its aggregate measurements without including source text or labels.

## Current Scope

- Typed contracts for chunks, evaluation queries, hits, and retrieval runs
- Abstract retriever and separate retrieval/generation metric interfaces
- Deterministic Recall@K, Precision@K, Hit Rate@K, reciprocal rank, MRR, and binary nDCG@K
- Retrieval evaluation runner with per-query results, aggregate metrics, and mean/p50/p95 retrieval latency
- JSONL dataset validation, versioned JSON run results, and metric-by-metric comparison
- A fixed-ranking, synthetic demo for offline evaluation; it is not semantic search or a benchmark
- PDF, text, and Markdown ingestion with deterministic word chunking and versioned corpus JSONL
- Dense, BM25, Hybrid RRF, and Hybrid + CrossEncoder retrievers
- Package and test foundation for Python 3.11+

See [architecture](docs/ARCHITECTURE.md) for the planned evaluation flow.

## Development

```bash
python -m pip install -e ".[dev,models]"
python -m pytest
python -m compileall src
groundtruth --help
groundtruth version
groundtruth validate-dataset datasets/demo/queries.jsonl
mkdir -p reports/tmp
groundtruth evaluate-demo reports/tmp/baseline.json
groundtruth evaluate-demo reports/tmp/candidate.json
groundtruth compare reports/tmp/baseline.json reports/tmp/candidate.json
```

These two demo runs contain identical fixed rankings; their latency will vary. Temporary results in `reports/tmp/` are ignored by Git.

## Ingest And Evaluate

```bash
groundtruth ingest --input path/to/documents --output datasets/corpus.jsonl --chunk-size 150 --chunk-overlap 30
groundtruth evaluate --corpus datasets/corpus.jsonl --dataset datasets/eval.jsonl --config configs/dense.yaml --output reports/dense.json
groundtruth evaluate --corpus datasets/corpus.jsonl --dataset datasets/eval.jsonl --config configs/hybrid_rerank.yaml --output reports/hybrid_rerank.json
groundtruth compare reports/dense.json reports/hybrid_rerank.json
```

`ingest` accepts one `.pdf`, `.txt`, or `.md` file, or a directory of those files. PDF page numbers are preserved. Chunk size and overlap are word counts. Each corpus JSONL line contains `schema_version: "1.0"` and a `DocumentChunk`. Create output parent directories before running commands.

## Retrieval Configurations

`configs/dense.yaml` uses normalized MiniLM embeddings and cosine-equivalent dot product. `configs/bm25.yaml` uses lowercase regex word tokens. `configs/hybrid.yaml` combines Dense and BM25 rankings with Reciprocal Rank Fusion (RRF). `configs/hybrid_rerank.yaml` applies a CrossEncoder to a bounded hybrid candidate pool. The model package is installed with the `models` extra; normal tests use fakes and do not download models.

Raw BM25 and cosine scores have different scales, so averaging them would make the blend arbitrary. RRF combines rank positions instead. See [retrieval design](docs/RETRIEVAL.md) for cache, candidate depth, and latency details.

## Golden Dataset

One UTF-8 JSON object per line. Required fields are `query_id` (unique non-empty string), `query` (non-empty string), and `relevant_chunk_ids` (non-empty array of unique chunk ID strings). Optional fields are `reference_answer` (string or null) and `tags` (array of unique strings). Blank lines and duplicate IDs are rejected.

```json
{"query_id":"q001","query":"Example question","relevant_chunk_ids":["doc1:p3:c2"],"tags":["definition"]}
```

Retrieval evaluation uses human relevance judgments and ordered chunk IDs. Generation evaluation remains a separate, unimplemented interface; no answer-quality claims are produced. `datasets/demo/` contains four synthetic chunks, three queries, and predetermined rankings for tests.

**Keep chunking fixed across compared runs.** Changing chunk size or overlap changes chunk boundaries and IDs, invalidating current chunk-level relevance labels. Relabel or add source-span judgments before comparing different chunking strategies.

## Creating a Golden Dataset

1. Place the exact source PDF at `datasets/raw/osha3120.pdf`. Raw files are ignored by Git. Run `groundtruth prepare-osha` once to produce the local `datasets/osha/corpus.jsonl` with fixed 150-word chunks and 30-word overlap, plus `datasets/osha/corpus_manifest.json` with source and corpus hashes. The command refuses to replace an existing canonical corpus.
2. Review or append answerable, unlabeled questions in `datasets/osha/questions_draft.jsonl`. Do not put `relevant_chunk_ids` in this file. The ten initial questions are prompts for human review, not labels.
3. Run `groundtruth label --corpus datasets/osha/corpus.jsonl --questions datasets/osha/questions_draft.jsonl --output datasets/osha/eval.jsonl`. Candidate discovery unions Dense, BM25, and Hybrid results. Use `more`, `search <phrase>`, or `inspect <chunk ID>` to check source passages. Select one or several IDs and explicitly confirm each answer. Progress is saved beside the draft file and resumes after interruption.
4. Review every draft question. Verify answerable questions; explicitly skip any question the corpus cannot answer. Skipped questions remain in review progress but are excluded from `eval.jsonl`. Run `groundtruth label-status --questions datasets/osha/questions_draft.jsonl --dataset datasets/osha/eval.jsonl`, then `groundtruth audit-dataset --corpus datasets/osha/corpus.jsonl --questions datasets/osha/questions_draft.jsonl --dataset datasets/osha/eval.jsonl`. Optional review copy: `groundtruth export-review --corpus datasets/osha/corpus.jsonl --dataset datasets/osha/eval.jsonl --output reports/tmp/osha-review.md`.
5. After all drafts are reviewed, at least 25 answerable questions are explicitly verified, and audit passes, run `groundtruth benchmark-osha`. It evaluates Dense, BM25, Hybrid, and Hybrid + CrossEncoder on the same corpus and verified labels. Skipped questions do not contribute to retrieval or latency metrics; unanswerable-query evaluation is not supported. Run JSON, summary, and ranking-only error analysis go to `reports/benchmarks/osha/`.

The OSHA corpus and labeling progress remain local. The final, audited 29-query benchmark dataset is versioned at `benchmarks/osha/eval.jsonl`; it contains no source passages. See [golden dataset guidance](docs/GOLDEN_DATASET.md).

## OSHA Benchmark

The OSHA benchmark uses 29 verified answerable queries; q013 was explicitly skipped and excluded from retrieval metrics. The raw PDF, canonical corpus, and full benchmark runs remain local artifacts. The final labels and review counts are committed under `benchmarks/osha/`. The versioned aggregate baseline is [baselines/osha/v1.json](baselines/osha/v1.json).

## Reproducing the OSHA Benchmark

```bash
python -m pip install -e ".[dev,models]"
groundtruth benchmark-setup osha
groundtruth benchmark-status osha
groundtruth benchmark-osha
groundtruth regress --baseline baselines/osha/v1.json --candidate reports/benchmarks/osha/summary.json --policy configs/regression/osha.yaml --output reports/regressions/osha/latest
```

Use `groundtruth benchmark-setup osha --source path/to/osha3120.pdf` when the pinned PDF is already available. The raw OSHA PDF is not committed. Setup verifies its SHA-256 before ingestion, rebuilds the corpus with fixed 150/30 word chunks, then checks 1 document, 45 pages, 91 chunks, the corpus fingerprint, and the committed 29-query golden dataset. It reuses a valid existing corpus without overwriting it. A different source revision or corpus fingerprint stops reproduction. q013 remains a reviewed skip, not an EvalQuery. See [benchmark reproducibility](docs/BENCHMARK_REPRODUCIBILITY.md) for provenance and runtime caveats.

## Benchmark Results

Full benchmark reports are local under `reports/benchmarks/osha/`; the committed baseline contains only aggregates and provenance, not source passages or golden labels.

## Retrieval Regression Testing

1. Run and audit a benchmark, then save its aggregate baseline: `groundtruth baseline-create --summary reports/benchmarks/osha/summary.json --output baselines/osha/v1.json --name osha-v1`. Existing baselines are protected; use `--force` only for an intentional replacement.
2. Change the retrieval implementation or configuration, then run a new candidate benchmark with the same corpus, golden dataset, and required K values.
3. Compare: `groundtruth regress --baseline baselines/osha/v1.json --candidate path/to/candidate/summary.json --policy configs/regression/osha.yaml --output reports/regressions/osha/latest`.
4. Inspect `regression.json` and `REGRESSION.md` for PASS/WARN/FAIL per metric and retriever. FAIL exits nonzero; WARN exits zero unless `--fail-on-warn` is set.

Comparison rejects incompatible corpus fingerprints, dataset hashes, query counts, retrievers, or required K values. The OSHA policy disables latency gating by default because timing depends on hardware and runtime conditions; measured latency remains visible in reports. See [regression guidance](docs/REGRESSION.md).

## Continuous Integration

Pull requests and pushes to `main` run tests, Ruff, `compileall`, CLI smoke checks, `groundtruth benchmark-verify osha`, and a synthetic regression gate. The normal CI workflow uses `.[dev]`; it does **not** download the OSHA PDF or model weights. The synthetic PASS/WARN/FAIL fixtures exercise the regression engine, not real retrieval quality.

The full OSHA benchmark runs only through the manual GitHub Actions workflow. It verifies the pinned public source, rebuilds the corpus, runs neural retrieval, and compares the result with baseline v1. It is manual because model downloads and benchmark execution take longer, OSHA hosting can be unavailable, and runner latency varies. Reports are uploaded as artifacts; the raw PDF, generated corpus, and model files are not. A source download failure stops the workflow rather than substituting another revision.
