# Human Golden Dataset

## Why Human Verification Matters

Retrieval metrics compare ranked chunk IDs to relevance judgments. An LLM or retriever can suggest candidates, but neither can certify the final labels: doing so would bias the benchmark toward its own search behavior. The final `relevant_chunk_ids` are written only after a person selects passages and confirms the question in `groundtruth label`.

## Relevance Rule

A chunk is relevant when its source text directly supports an answer to the question. Select every chunk needed to substantiate the answer, including adjacent chunks when a passage spans a boundary. Do not label a chunk merely because it repeats query words. Keep the reference answer concise and source grounded. Use `skip` for questions that cannot be answered from the corpus; skipped questions do not enter this answerable-only benchmark.

## Candidate Bias

The labeling view unions Dense top 10, BM25 top 10, and Hybrid top 10 by default, deduplicated by chunk ID. `--candidate-depth` changes that starting depth; `more` expands it. `search <phrase>` scans the whole corpus, and `inspect <chunk ID>` displays a full passage. Check the source PDF page before confirming an exact fact. Candidate ranks and scores are diagnostic hints, never labels.

## Progress And Audit

The ten seed questions in `datasets/osha/questions_draft.jsonl` are unlabeled. Append further answerable questions to reach a first benchmark of at least 25; existing questions must retain their IDs and text after labeling starts. Review state is written atomically to `questions_draft.progress.json` after each edit, skip, and confirmation. Verified records are exported to `eval.jsonl`. Editing a verified record requires an explicit prompt and another confirmation. If output differs from progress after an interruption, inspect both and use `--reconcile-output` only after confirming that progress is authoritative.

Every draft question must receive human review. Answerable questions are verified; a genuinely unanswerable question may be explicitly skipped. A review is complete when every draft is verified or skipped, with none unlabeled or in progress. `label-status` reports reviewed completion separately from verified-answerable coverage.

`audit-dataset` checks unique IDs, complete human review, nonempty relevance sets for verified questions, chunk existence, manifest/corpus fingerprint, source PDF hash, reference-answer counts, tags, and source/page coverage. Skipped IDs remain visible in progress and the audit summary but must not appear in `eval.jsonl`. The OSHA benchmark requires at least 25 verified answerable queries. The audit reports unusually broad labels for review; it does not repair data. `export-review` creates a Markdown copy with full relevant passages for visual inspection; treat that export as source material and do not add it to a tooling commit.

## Fixed Corpus And Reproducibility

Place the exact OSHA 3120 PDF at `datasets/raw/osha3120.pdf`. `prepare-osha` ingests it once with 150-word chunks and 30-word overlap, records source SHA-256 and corpus fingerprint in `datasets/osha/corpus_manifest.json`, and refuses to regenerate an existing corpus. The raw PDF and text-bearing corpus remain outside Git. All four benchmark configurations use the same corpus, golden labels, and K values. Run JSON records config, dataset and source hashes, package/Python versions, timestamp, and corpus fingerprint.

Chunk-level IDs depend on boundaries. Changing chunk size or overlap invalidates labels. A future cross-chunking benchmark needs relabeling or source-span relevance judgments. No unanswerable-query metrics or generation evaluation are included here.
Skipped queries do not contribute to Recall, Precision, Hit Rate, MRR, nDCG, or latency aggregates.
