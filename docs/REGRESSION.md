# Retrieval Regression Testing

`baseline-create` validates a completed benchmark summary and its sibling run files, then writes a versioned aggregate-only baseline. It does not run retrieval. The baseline records the corpus fingerprint, dataset SHA-256 when present, query count, K values, run IDs, aggregate metrics, and latency. It contains no corpus text, query text, hits, or relevance labels. Existing baselines are not replaced without `--force`.

`regress` compares a candidate summary with that baseline under a YAML policy. Corpus fingerprint, dataset hash (when available), and query count must match. Every policy-selected retriever, metric, and required K must exist in both. Unsupported schemas, duplicate JSON keys, malformed values, and nonfinite numbers are rejected as incompatible input rather than scored as regressions. When neither side has a dataset hash, only the dataset filename can be checked; prefer run files with a hash for strong compatibility.

Quality metrics (Recall@K, Precision@K, Hit Rate@K, MRR, nDCG@K) are higher-is-better and use absolute drops. Mean, p50, and p95 latency are lower-is-better and use relative percentage increases. A metric passes below its warning threshold, warns at or above the warning threshold, and fails at or above the failure threshold. Any FAIL makes the overall result FAIL; otherwise any WARN makes it WARN. Disabled metrics are measured in the report but do not influence status. Per-retriever rules can override defaults. Thresholds are engineering policy choices, **not statistical significance tests**.

Latency is sensitive to hardware, warmup, background load, and runtime versions. The example OSHA policy leaves latency gating disabled; enable it only for a controlled environment. A baseline latency of zero with a positive candidate latency cannot support a relative increase calculation and is rejected. A zero-to-zero comparison passes.

Exit codes: PASS is 0, WARN is 0 by default (1 with `--fail-on-warn`), FAIL is 1, and invalid or incompatible input is 2. The CLI writes `regression.json` and `REGRESSION.md` after compatibility succeeds, including WARN and FAIL results. It never edits golden labels.

The baseline and candidate must use the same fixed chunking and human-reviewed dataset. Different corpora, changed relevance judgments, or changed query sets require a new baseline. Per-query rank-change analysis and CI enforcement are left for later work; this phase does not evaluate generation or unanswerable queries.
