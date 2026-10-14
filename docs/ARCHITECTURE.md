# Architecture

RAG teams frequently change chunking, embeddings, retrieval methods, and rerankers without knowing whether retrieval quality improved. Groundtruth-Rag will run repeatable evaluations against a human-verified golden dataset. The framework is domain agnostic; an OSHA Lockout/Tagout corpus is planned as the first real benchmark, not embedded in the core.

```text
Document Corpus
      |
      v
Chunking
      |
      v
Retriever Configuration
      |
      v
Evaluation Runner
      |
      v
Retrieval Metrics
      |
      v
Optional Generation Evaluation
      |
      v
Benchmark Comparison
      |
      v
Regression Gate
```

## Responsibilities

- **Ingestion:** Read source documents and produce stable `DocumentChunk` records with provenance. No reader or chunker exists yet.
- **Retrieval:** Implement `Retriever.index(chunks)` and `retrieve(query, k)`. Dense, hybrid, and reranked pipelines will share this contract. Scores may differ by retriever; rank and chunk ID provide the common comparison surface.
- **Evaluation:** Match `EvalQuery.relevant_chunk_ids` to retrieved chunk IDs. Planned metrics are Recall@K, Precision@K, Hit Rate@K, MRR, and nDCG@K. The current `RetrievalMetric` is an interface only.
- **Generation evaluation:** Optional answer-level assessment, separate from retrieval metrics. Planned metrics are faithfulness and answer relevance; the current `GenerationMetric` is an interface only.
- **Reporting:** Compare named, reproducible run configurations, latency, and metric results. No reports are generated yet.
- **Regression testing:** Compare a candidate run to an established baseline and apply explicit thresholds. No gate is active until real benchmark results exist.

The CLI currently provides `--help` and `version`. Future commands are `ingest`, `validate-dataset`, `evaluate`, `compare`, and `report`; none are exposed until implemented.
