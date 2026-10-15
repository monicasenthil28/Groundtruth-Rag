# Retrieval Robustness

`robustness-analyze` reads two existing run JSON files. It does not load models or rerun retrieval. Baseline and candidate must use the same retriever family, corpus fingerprint, dataset hash, K values, query IDs, query text, and golden relevance IDs. The analysis recalculates stored per-query metrics from hits using the existing retrieval metric functions, and rejects inconsistent data.

The resampling unit is the **query**, not a chunk, hit, or page. For each selected metric, the analyzer forms aligned candidate-minus-baseline per-query differences, then samples queries with replacement. The observed delta is the mean paired difference. The default 10,000 replicates use a local NumPy generator with seed 2026. A 95% percentile interval uses the 2.5th and 97.5th percentiles of bootstrap mean deltas. Repeating with the same runs, config, and environment gives deterministic quality output; the JSON omits timestamps. The report contains metric summaries and provenance only, not source text, query text, labels, or hit lists.

`UNCHANGED` means every paired per-query metric is exactly identical. `STABLE_IMPROVEMENT` requires a positive observed delta and lower interval bound above zero. `STABLE_REGRESSION` requires a negative observed delta and upper bound below zero. Other cases are `INCONCLUSIVE`. These are descriptive benchmark-sample labels, not population-level proof or formal universal statistical significance. With 29 OSHA queries, intervals reflect only resampling that small reviewed benchmark. Bootstrap fractions above/below zero are replicate proportions, not probabilities that a retriever is correct.

```powershell
groundtruth robustness-analyze --baseline-run reports/benchmarks/osha/hybrid_rerank.json --candidate-run reports/experiments/osha/retrieval-v1/candidates/rerank-pool-30/hybrid_rerank.json --config configs/robustness/osha.yaml --output reports/robustness/osha/rerank-pool-30
```

Hybrid candidates should use `reports/benchmarks/osha/hybrid.json`; Hybrid + CrossEncoder candidates use `hybrid_rerank.json`. The CLI checks this family match. The default config includes Recall, Hit Rate, and nDCG at K=1/3/5/10 plus MRR. The core also supports Precision at those K values. Iterations, seed, confidence level, and selected metrics are configurable.

Latency is a different uncertainty question. `latency-trials` validates the fixed experiment specification, builds and indexes one candidate (including model loading), excludes full-query warmup runs, then measures five full-query trials. Each trial records mean, p50, and p95 retrieval milliseconds. The summary reports mean trial mean, median trial p50, median trial p95, and minimum/maximum p95. It records OS, Python, CPU/GPU identifiers where available, CUDA availability, and dependency versions without hostnames. Hardware, runtime, cache, and background load affect timing; these local measurements do not generalize to other machines. No latency confidence interval or stability class is inferred.

```powershell
groundtruth latency-trials --experiment experiments/osha/retrieval-v1.yaml --candidate rerank-pool-30 --config configs/robustness/osha.yaml --output reports/robustness/osha/rerank-pool-30/latency.json
groundtruth robustness-analyze --baseline-run reports/benchmarks/osha/hybrid_rerank.json --candidate-run reports/experiments/osha/retrieval-v1/candidates/rerank-pool-30/hybrid_rerank.json --config configs/robustness/osha.yaml --latency reports/robustness/osha/rerank-pool-30/latency.json --output reports/robustness/osha/rerank-pool-30
```

The optional `--latency` argument attaches a matching trial report to quality output without mixing their uncertainty estimates. Generated files stay local under `reports/robustness/` and are not committed. Phase 10 PASS/WARN/FAIL regression thresholds and PROMOTE_CANDIDATE/REVIEW/REJECT decisions remain independent and unchanged; a human reviews all evidence before promotion. No generation evaluation is included.
