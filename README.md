# Groundtruth-Rag

## RAG Retrieval Evaluation & Regression Platform

Groundtruth-Rag is a domain-agnostic foundation for comparing retrieval architectures against a human-verified golden dataset. It is intended to make changes to chunking, embeddings, retrieval, and reranking measurable instead of relying on subjective chatbot testing.

The planned comparison is **Dense** vs **Dense + BM25** vs **Dense + BM25 + CrossEncoder**. Those retrievers and the evaluation runner are not implemented in this bootstrap phase.

## Current Scope

- Typed contracts for chunks, evaluation queries, hits, and retrieval runs
- Abstract retriever and separate retrieval/generation metric interfaces
- A `groundtruth version` CLI command
- Package and test foundation for Python 3.11+

See [architecture](docs/ARCHITECTURE.md) for the planned evaluation flow.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m compileall src
groundtruth --help
groundtruth version
```

The optional `benchmark` dependency group records libraries planned for future retrieval, ingestion, and reporting work. It is not needed to exercise the current contracts.

## Benchmark Results

Benchmark pending — results will be generated from the real evaluation dataset.
