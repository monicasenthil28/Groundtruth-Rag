# Retrieval Design

All four configurations implement the same `Retriever.index(chunks)` and `retrieve(query, k)` contract. The evaluation runner sees only ordered `RetrievalHit` records. The corpus JSONL format is independent of models.

## Ingestion And IDs

PDFs are read page by page with pypdf. Text and Markdown are one logical document each. Whitespace is normalized before word-based chunking. For a directory, supported files are processed in sorted relative-path order; unsupported files are reported. Chunk metadata records a zero-based `chunk_index`, `word_start`, exclusive `word_end`, and SHA-256 content hash. Chunk ID is SHA-256 over document ID, page, chunk index, and content hash. Document ID is a hash of the logical source path. The same files and chunk settings produce the same IDs.

## Dense

`sentence-transformers/all-MiniLM-L6-v2` encodes chunk text in batches. Corpus and query vectors are L2-normalized; NumPy dot product ranks them by cosine similarity. Equal scores break by chunk ID. Embeddings are kept in memory for retrieval.

## BM25

`rank-bm25` indexes lowercase regex `\w+` tokens. No stemming, stopword removal, or hidden NLP pipeline is applied. Equal scores break by chunk ID.

## Hybrid RRF

Dense and BM25 each retrieve a configurable candidate depth. For every candidate, RRF adds `1 / (rrf_k + rank)` from each list where it appears, with one-based ranks and default `rrf_k=60`. The fused list sorts by score and then chunk ID. Raw cosine and BM25 scores are never averaged because their scales are not comparable. Hit metadata retains component ranks and scores plus the RRF score.

## CrossEncoder

The hybrid result provides a bounded candidate pool (default 20). `cross-encoder/ms-marco-MiniLM-L-6-v2` scores `(query, chunk text)` pairs. Final order uses descending CrossEncoder score, breaking ties by chunk ID. Hits retain RRF diagnostics and add the CrossEncoder score.

## Cache And Tradeoffs

Dense embeddings are cached under `.cache/groundtruth-rag/`, or `GROUNDTRUTH_CACHE_DIR` when set. The cache key includes model name, ordered chunk IDs and text, and normalization format. Changing the corpus or model invalidates the key. Cache files are ignored by Git. Query embeddings are computed for each retrieval call.

Larger candidate depths can improve recall before fusion or reranking, while increasing retrieval and CrossEncoder latency. Models load during indexing, before per-query retrieval timing. The run JSON records per-query and aggregate retrieval latency plus the ordered corpus fingerprint. Comparison rejects different corpus fingerprints. Compare quality and latency as separate metrics; there is no composite score.

## Golden Label Constraint

The current golden format names relevant chunk IDs. Changing chunk size or overlap changes chunk boundaries and IDs, so comparisons must keep chunking fixed. Cross-chunking studies require relabeling or future source-span judgments. No OSHA labels or benchmark results are included in this phase.
