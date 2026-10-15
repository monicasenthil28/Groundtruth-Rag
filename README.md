# Groundtruth-Rag

**Measure retrieval changes against human judgments, not chatbot impressions.**

Groundtruth-Rag is a Python retrieval engineering and evaluation platform. It builds a deterministic corpus, compares four retrieval strategies on human-verified relevance labels, and carries evidence through regression tests, failure analysis, experiments, robustness checks, and a human-gated release process. It is not a chatbot or a RAG application UI.

**Current state:** the planned v1.0 *retrieval scope* is complete. The package version remains `0.1.0`. OSHA retrieval **v2** is active, with **v1** preserved as an explicit rollback target. Generation evaluation is not implemented.

![Groundtruth-Rag architecture](docs/assets/architecture/groundtruth-rag.svg)

[Explore the interactive architecture](docs/assets/architecture/groundtruth-rag.html) · [Edit the source-backed diagram](docs/assets/architecture/groundtruth-rag.architecture.json)

## What gets measured

The fixed OSHA benchmark contains **29 human-verified, answerable queries**. Question `q013` was reviewed and skipped; `q019` remains included. One pinned 45-page PDF becomes **91 chunks** using 150-word chunks with 30-word overlap. The same corpus and labels are used for Dense, BM25, Hybrid Reciprocal Rank Fusion (RRF), and Hybrid + CrossEncoder. Runs record Recall@K, Precision@K, Hit Rate@K, MRR, nDCG@K, and retrieval latency.

| Retriever | Recall@3 | nDCG@3 | MRR |
| --- | ---: | ---: | ---: |
| Dense | 0.4655 | 0.4113 | 0.4999 |
| BM25 | 0.5690 | 0.5462 | 0.6518 |
| Hybrid RRF | 0.5948 | 0.5549 | 0.6402 |
| Hybrid + CrossEncoder (active v2) | **0.8534** | **0.7764** | **0.7989** |

These are results for this fixed OSHA benchmark, not universal retrieval claims. The active v2 reranker (`rerank-pool-30`) matched the original Hybrid + CrossEncoder baseline on Recall@3 and nDCG@3; its MRR result was **inconclusive**. The observed MRR delta was `-0.003448`, with a 95% paired-bootstrap interval of `[-0.010345, 0]`. Latency is environment-sensitive: the Phase 11 median trial p95 was **1365.47 ms**, while the v2 validation run p95 was **1416.03 ms**. Do not read this as a general speed improvement. Metrics above come from [the committed v2 aggregate baseline](baselines/osha/v2.json); methodology is in [benchmark reproducibility](docs/BENCHMARK_REPRODUCIBILITY.md) and [robustness](docs/ROBUSTNESS.md).

## Engineering loop

`Change → benchmark → aggregate and query-level regression → failure forensics → candidate experiment and Pareto review → paired robustness analysis → human approval → validation → versioned baseline → activation or rollback`

Every promotion is tied to the same benchmark identity. Golden labels are never silently rewritten; incompatible corpus or dataset identities are rejected by the comparison path. Changing chunk boundaries changes chunk IDs and invalidates the existing chunk-level judgments. [v2's manifest](promotions/osha/v2.yaml) is `ACTIVE` after a passing validation; [the registry](configs/versions/osha/registry.yaml) points to v2 and retains v1 for rollback. The [v1](baselines/osha/v1.json) and [v2](baselines/osha/v2.json) aggregate baselines and their [v1](baselines/osha/v1.query.json) and [v2](baselines/osha/v2.query.json) query baselines remain versioned.

## Run it

Python 3.11+ is required. The `models` extra is needed for real Dense and CrossEncoder runs; tests use fakes and do not download models.

```bash
python -m pip install -e ".[dev,models]"
groundtruth --help
groundtruth benchmark-verify osha
groundtruth config-status osha
```

To reproduce the historical fixed benchmark, acquire the [pinned source](benchmarks/osha/benchmark.yaml) (or pass a verified local PDF), then run:

```bash
groundtruth benchmark-setup osha --source path/to/osha3120.pdf
groundtruth benchmark-status osha
groundtruth benchmark-osha
```

`benchmark-setup` checks the PDF SHA-256 before creating or reusing the canonical corpus. Without `--source`, it attempts the official source; `--release-fallback` is available for the pinned release asset. The `benchmark-osha` command is the historical four-retriever benchmark using its fixed configs, **not** an automatic rerun of active v2. It loads models and writes local reports. The normal CI workflow does not run it; the [manual OSHA workflow](.github/workflows/osha-benchmark.yml) performs real benchmark reproduction. See [reproduction instructions](docs/BENCHMARK_REPRODUCIBILITY.md) before comparing new results.

<details>
<summary>Inspect the evaluation and release workflows</summary>

```bash
# Compare a compatible saved candidate summary with the original baseline.
groundtruth regress --baseline baselines/osha/v1.json --query-baseline baselines/osha/v1.query.json --candidate reports/benchmarks/osha/summary.json --policy configs/regression/osha.yaml --output reports/regressions/osha

# Inspect the active version. Rollback is an explicit state-changing action.
groundtruth config-status osha
groundtruth promotion-rollback osha --to v1
```

The regression example requires a fresh compatible `reports/benchmarks/osha/summary.json`; it does not create one. For labeling, forensics, experiments, robustness, and the complete promotion sequence, use the dedicated guides below rather than treating isolated commands as a release recipe.

</details>

## Reproducibility and integrity

The benchmark pins the source PDF hash, `pypdf` version, chunking parameters, corpus fingerprint, and golden dataset identity. Candidate configs, run IDs, and both aggregate and query-level baselines carry provenance through validation and activation. The pinned source SHA-256 is `3ea7380fa10860417ed30720eaf8beb7a85d106e3dafb97b880001df6552767b`; the corpus fingerprint is `55bebfded1835db967ef9fed5688ff0d8ad919a42c308cbc555972419df3531e`; the golden dataset SHA-256 is `78dd4d33573577c9ce5b285699afea6c91ad6bfcce2499f8a629960cb058f85e`. Raw source PDFs and the canonical corpus are local artifacts, while verified labels and baselines are versioned.

| Area | Where to look |
| --- | --- |
| `src/groundtruth_rag/` | Ingestion, retrievers, evaluation, reporting, experiments, robustness, promotion, CLI |
| `benchmarks/` | Fixed benchmark specification, golden dataset, regression policy |
| `configs/`, `baselines/` | Retriever versions, active registry, aggregate and query baselines |
| `experiments/`, `promotions/` | Candidate specifications and approved promotion manifest |
| `tests/`, `.github/workflows/` | Offline tests, CI, and manual benchmark automation |

## Documentation

[Architecture](docs/ARCHITECTURE.md) · [Retrieval](docs/RETRIEVAL.md) · [Golden dataset](docs/GOLDEN_DATASET.md) · [Regression](docs/REGRESSION.md) · [Benchmark reproducibility](docs/BENCHMARK_REPRODUCIBILITY.md) · [Experiments](docs/EXPERIMENTS.md) · [Robustness](docs/ROBUSTNESS.md) · [Promotion and rollback](docs/PROMOTION.md)

**Scope boundary:** this project evaluates retrieval quality and release decisions. It does not evaluate generated answers or claim end-to-end RAG response quality.
