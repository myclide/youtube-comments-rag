# YouTube Comments RAG: Evaluated Retrieval and Local Comment Analysis

A retrieval-augmented generation system for analyzing large-scale YouTube comment collections.

This project began as a local RAG prototype using MiniLM embeddings, FAISS retrieval, and Llama 3 through Ollama. It was subsequently rebuilt around a reproducible information-retrieval benchmark, modern dense retrieval, cross-encoder reranking, statistical evaluation, latency analysis, and an updated Streamlit application.

The main retrieval result is:

> **Qwen3-Embedding-0.6B + BGE-reranker-v2-m3 improved nDCG@10 from 0.6365 to 0.7347 (+15.4%) over the original MiniLM dense baseline on a 60-query development benchmark.**

The paired nDCG improvement had a 95% bootstrap confidence interval of:

```text
[+0.0544, +0.1437]
```

with a paired randomization:

```text
p = 0.00005
```

The evaluated Qwen3 + BGE retrieval pipeline is now integrated into the local Streamlit RAG application.

---

# Overview

The system answers questions about the discussion under YouTube videos by retrieving relevant comments and using them as evidence for local LLM generation.

Example questions include:

- What laptops or computer setups do viewers recommend for programming?
- What hardware specifications do commenters consider important?
- What complaints do viewers have about this product?
- What do viewers say about a song's lyrics and performance?
- What questions are commenters asking about an upcoming phone?
- What do viewers say about the animals shown in this video?

The current quality-first application pipeline is:

```text
YouTube comment corpus
        ↓
Qwen3-Embedding-0.6B
        ↓
Dense Top-25 candidate retrieval
        ↓
BAAI/bge-reranker-v2-m3
        ↓
Top-10 evidence
        ↓
Local Llama 3 via Ollama
        ↓
Streamlit grounded answer
```

The original MiniLM + FAISS implementation is retained as a legacy baseline.

---

# Project Evolution

## Original prototype

The first version used:

```text
YouTube comments
      ↓
all-MiniLM-L6-v2 embeddings
      ↓
FAISS retrieval
      ↓
Top-k comments
      ↓
Llama 3 via Ollama
      ↓
Streamlit answer
```

This demonstrated a functional local RAG workflow, but it did not contain:

- a graded retrieval benchmark
- pooled relevance judgments
- modern retriever comparison
- cross-encoder reranking
- statistical significance testing
- latency analysis
- controlled candidate-set ablations

The project was therefore rebuilt with retrieval quality as an explicitly evaluated component.

---

# Current System Architecture

## Offline path

```text
YouTube Data API
        ↓
148,020 reconstructed comments
        ↓
Within-video duplicate consolidation
        ↓
142,963 retrieval documents
        ↓
Qwen3 document encoding
        ↓
Normalized 1024-dimensional embeddings
        ↓
Local serving index artifacts
```

## Online path

```text
User question
        ↓
Qwen3 query embedding
        ↓
Exact normalized dot-product retrieval
        ↓
Top-25 candidates
        ↓
BGE cross-encoder reranking
        ↓
Top-10 evidence
        ↓
Llama 3 through Ollama
        ↓
Grounded answer
```

The modern serving implementation currently uses exact normalized dot-product retrieval rather than an approximate nearest-neighbor index.

FAISS remains part of the legacy MiniLM backend only.

---

# Dataset Reconstruction

The project originally started from a Kaggle YouTube-comment export.

During data audit, the original CSV was found to be unsuitable as the final retrieval corpus because multiline comments could be represented incorrectly by the exported format.

The video population was therefore reconstructed using the official YouTube Data API v3.

## Video population

| Statistic | Value |
|---|---:|
| Parsed unique video IDs | 466 |
| Available videos | 399 |
| Unavailable videos | 67 |
| Successfully processed videos | 386 |
| Skipped videos | 13 |
| Failed collection jobs | 0 |
| Videos contributing comments | 382 |

Collection was implemented as a resumable per-video process.

Each video was written to an individual JSONL file so interrupted runs could continue without restarting the entire collection job.

---

# Corpus Construction

The reconstructed raw corpus contains:

| Statistic | Value |
|---|---:|
| Raw comments | 148,020 |
| Unique comment IDs | 148,020 |
| Contributing videos | 382 |
| Multiline comments | 12,716 |
| Multiline share | 8.59% |
| Mean comment length | 73.85 characters |
| Median comment length | 39 characters |

Exact normalized text repetition was analyzed separately.

Global exact-text repetition was **not** removed because identical comments appearing under different videos can represent legitimate independent evidence.

Instead, identical normalized text was collapsed only when it occurred repeatedly within the same video.

After within-video consolidation:

| Statistic | Value |
|---|---:|
| Raw comments | 148,020 |
| Retrieval documents | 142,963 |
| Rows collapsed | 5,057 |
| Corpus reduction | 3.42% |
| Repeated-text groups | 2,385 |
| Maximum within-video occurrence count | 140 |

Repeated evidence is not discarded completely.

Each retrieval document preserves metadata such as:

- occurrence count
- number of unique authors
- total likes
- maximum likes
- first publication time
- last publication time

A stable document identifier is constructed as:

```text
video_id::representative_comment_id
```

---

# Retrieval Benchmark

A manually reviewed development benchmark was constructed to compare retrieval systems.

## Benchmark composition

```text
20 videos
60 queries
3 queries per video
```

Videos were selected across different corpus sizes and content categories.

The benchmark includes three query types:

```text
semantic_recommendation
attribute
entity_lexical
```

These cover questions that require:

- semantic recommendation retrieval
- attribute-specific evidence
- named entity or lexical matching

---

# Relevance Judgments

Each query-document pair uses a three-level relevance scale.

| Relevance | Meaning |
|---|---|
| `0` | Not relevant |
| `1` | Related but provides limited answer value |
| `2` | Directly useful for answering the query |

Judging context includes:

```text
video title
query
query type
comment text
```

The video title can restore shared context, but it cannot create an opinion or fact that the comment itself does not express.

---

# Pooled Qrels

The initial candidate pool included results from:

- BM25
- MiniLM dense retrieval
- lexical candidate discovery
- deterministic random sampling

When Qwen3 was introduced, it retrieved documents that had not been judged in the original pool.

Instead of treating those unjudged results as irrelevant, the benchmark pool was expanded.

Qwen3 contributed:

```text
419 novel candidates
```

with labels:

| Relevance | Count |
|---|---:|
| 0 | 215 |
| 1 | 54 |
| 2 | 150 |

The final expanded judgment pool contains:

| Statistic | Value |
|---|---:|
| Queries | 60 |
| Judged query-document pairs | 3,758 |
| Relevance 0 | 2,339 |
| Relevance 1 | 435 |
| Relevance 2 | 984 |
| Duplicate query-document pairs | 0 |
| Unresolved judgments | 0 |

This expanded qrels set is used for the final unified retrieval comparisons.

---

# Evaluation Metrics

## nDCG@10

The primary ranking metric is graded nDCG@10.

Gain is defined as:

```text
gain = 2^relevance - 1
```

Therefore:

```text
rel=0 → gain 0
rel=1 → gain 1
rel=2 → gain 3
```

Direct-answer evidence receives more weight than merely related evidence.

---

## MRR@10

MRR uses:

```text
relevance >= 1
```

as the binary relevant threshold.

It emphasizes how early the first useful result appears.

---

## Precision@10

Precision@10 measures the fraction of the first ten results satisfying:

```text
relevance >= 1
```

---

## Strict diagnostics

Additional metrics treat only:

```text
relevance = 2
```

as relevant.

These include:

- Strict MRR@10
- Strict Precision@10

They help distinguish direct-answer evidence from weaker topical matches.

---

# Main Retrieval Results

All systems below are evaluated against the same expanded pooled qrels.

| System | nDCG@10 | MRR@10 | Precision@10 | Strict P@10 |
|---|---:|---:|---:|---:|
| BM25 | 0.4443 | 0.7547 | 0.4617 | 0.3200 |
| MiniLM dense | 0.6365 | 0.8994 | 0.6367 | 0.4650 |
| Qwen3 dense | 0.6606 | 0.9103 | 0.6400 | 0.4683 |
| MiniLM Top-25 → BGE | 0.7135 | 0.9232 | 0.6767 | 0.5233 |
| **Qwen3 Top-25 → BGE** | **0.7347** | **0.9461** | **0.6967** | **0.5533** |
| MiniLM ∪ Qwen3 → BGE | 0.7375 | 0.9361 | 0.6983 | 0.5567 |
| BM25 ∪ MiniLM ∪ Qwen3 → BGE | 0.7383 | 0.9472 | 0.6933 | 0.5483 |

The highest raw nDCG belongs to the three-retriever union.

That system was **not** selected as the final architecture.

---

# Why Qwen3 + BGE Was Selected

Candidate-set ablation produced:

| Candidate set | Avg. candidates/query | Avg. relevant | Avg. rel=2 |
|---|---:|---:|---:|
| MiniLM | 25.00 | 12.85 | 9.02 |
| Qwen3 | 25.00 | 13.72 | 9.75 |
| BM25 ∪ MiniLM | 41.67 | 17.18 | 11.72 |
| MiniLM ∪ Qwen3 | 35.32 | 17.72 | 12.50 |
| BM25 ∪ MiniLM ∪ Qwen3 | 50.20 | 21.18 | 14.65 |

The three-retriever union increases reranker candidate volume from:

```text
25.0
```

to:

```text
50.2
```

documents per query.

Its nDCG improvement over Qwen3-only candidate generation is only:

```text
0.7347 → 0.7383
```

or:

```text
+0.0037 absolute
```

Paired statistical analysis gave:

```text
95% CI = [-0.0198, +0.0309]
p = 0.790832
```

There is therefore no statistical evidence that the larger candidate union is superior on this benchmark.

The final architecture is consequently:

```text
Qwen3 Top-25
      ↓
BGE reranker
```

This avoids approximately doubling reranker candidate volume for an unsupported quality difference.

---

# Improvement Over the Original MiniLM Baseline

| Metric | MiniLM | Qwen3 + BGE | Relative change |
|---|---:|---:|---:|
| nDCG@10 | 0.6365 | 0.7347 | **+15.41%** |
| MRR@10 | 0.8994 | 0.9461 | +5.19% |
| Precision@10 | 0.6367 | 0.6967 | **+9.42%** |

The main result is therefore:

> Replacing the original MiniLM dense retriever with Qwen3 candidate retrieval plus BGE cross-encoder reranking improved nDCG@10 by approximately **15.4%** on the 60-query development benchmark.

---

# Statistical Validation

Query-level paired testing was used because systems were evaluated on the same 60 queries.

The statistical analysis used:

```text
10,000 paired bootstrap samples
100,000 paired sign-flip randomization samples
```

with 95% bootstrap confidence intervals.

---

## nDCG@10

Qwen3 + BGE versus MiniLM:

```text
MiniLM:       0.6365
Qwen3 + BGE:  0.7347

Absolute delta:
+0.0981

Relative improvement:
+15.41%

95% bootstrap CI:
[+0.0544, +0.1437]

Paired randomization p-value:
0.000050

Query wins / ties / losses:
39 / 8 / 13
```

The confidence interval remains above zero and the paired randomization result provides strong evidence of improved nDCG on this benchmark.

---

## Precision@10

```text
Absolute delta:
+0.0600

Relative improvement:
+9.42%

95% CI:
[+0.0167, +0.1050]

p = 0.013610
```

---

## MRR@10

```text
Absolute delta:
+0.0467

Relative improvement:
+5.19%

95% CI:
[-0.0014, +0.0992]

p = 0.086489
```

MRR improves numerically, but the benchmark does not provide sufficient evidence to describe this MRR improvement as statistically significant at the 0.05 level.

---

# Negative Ablations

Additional complexity did not always improve retrieval.

## Reciprocal Rank Fusion

An equal-weight BM25 + MiniLM Reciprocal Rank Fusion experiment produced:

```text
nDCG@10 = 0.5892
MRR@10  = 0.8769
P@10    = 0.5683
```

on the original evaluation pool.

MiniLM alone achieved:

```text
nDCG@10 = 0.6449
```

on that same pool.

RRF therefore reduced nDCG by approximately:

```text
8.64%
```

relative to MiniLM in that experiment.

The hybrid configuration was rejected rather than being included simply because hybrid retrieval is more complex.

---

# CPU Latency Benchmark

Latency was measured in the local CPU evaluation environment.

The benchmark contained:

```text
60 queries
20 videos
6,841 benchmark documents
3 repetitions per query
```

Model loading and offline document indexing were excluded from online query latency.

## Online latency

| System | Mean | Median | p95 | p99 | Serial QPS |
|---|---:|---:|---:|---:|---:|
| MiniLM dense | 6.43 ms | 6.39 ms | 7.44 ms | 10.16 ms | 155.475 |
| Qwen3 dense | 74.27 ms | 73.48 ms | 84.72 ms | 90.67 ms | 13.464 |
| Qwen3 Top-25 → BGE | 3156.65 ms | 2665.49 ms | 7878.01 ms | 9417.25 ms | 0.317 |

The estimated incremental BGE reranking cost was approximately:

```text
3082 ms/query
```

or roughly:

```text
97.6%
```

of the mean Qwen3 + BGE pipeline latency in this CPU environment.

The cross-encoder is therefore the primary online latency bottleneck.

These values should not be generalized to GPU or production-serving hardware.

---

# Offline Embedding Throughput

The latency benchmark measured offline encoding of 6,841 documents:

| Model | Documents/sec |
|---|---:|
| MiniLM | 475.94 |
| Qwen3-Embedding-0.6B | 24.80 |

The final full serving index was then built for all:

```text
142,963
```

retrieval documents.

The full index build produced:

```text
Embedding shape:
(142963, 1024)

Encoding time:
6054.36 seconds

Throughput:
23.61 documents/sec
```

on the local CPU environment.

Document embeddings are generated offline and are not part of the normal online query path.

---

# Rerank-Depth Quality / Latency Trade-off

Qwen3 candidate depth was varied before BGE reranking.

| Rerank depth | nDCG@10 | MRR@10 | Precision@10 | nDCG retention | Estimated mean pipeline latency |
|---|---:|---:|---:|---:|---:|
| Top-10 | 0.6636 | 0.9067 | 0.6400 | 90.33% | 1.454 s |
| Top-15 | 0.7108 | 0.9335 | 0.6800 | 96.75% | 2.322 s |
| Top-20 | 0.7188 | 0.9403 | 0.6800 | 97.84% | 2.829 s |
| **Top-25** | **0.7347** | **0.9461** | **0.6967** | **100%** | **3.128 s** |

Top-15 is exposed in the Streamlit application as a CPU-balanced alternative.

However, Top-25 remains the reference quality-first configuration.

The latency values in this depth experiment are estimated pipeline values formed from measured BGE reranking latency plus paired previously measured Qwen3 dense latency.

They are not same-run end-to-end measurements.

---

# Serving Index

The modern application does not re-encode every document when a user submits a query.

Instead, document embeddings are generated offline.

Run:

```powershell
python processing\build_modern_index_v1.py
```

The generated local serving artifacts are stored under:

```text
data/modern_index_v1/
├── document_embeddings.npy
├── metadata.parquet
└── manifest.json
```

The embedding matrix contains:

```text
142,963 × 1,024
```

normalized Qwen3 embeddings.

These generated artifacts are excluded from Git because of their size.

---

# Serving Reproduction Validation

The modern serving implementation was compared against the frozen evaluation results.

Validation used benchmark query `q001`:

```text
What laptops or computer setups do viewers recommend for programming?
```

for video:

```text
dxPhJ0wfp0g
```

The serving pipeline reproduced:

```text
Dense Top-25 overlap:
25 / 25

Dense candidate-set match:
True

Final BGE Top-10 overlap:
10 / 10

Final BGE Top-10 exact ranking match:
True

Public retrieve() Top-10 exact match:
True
```

The exact ordering inside the dense Top-25 candidate set differed slightly between the separately encoded serving index and the frozen experiment.

However:

- all 25 candidate documents were identical
- the BGE reranker received the same candidate set
- the final Top-10 result set was identical
- the final Top-10 ranking order was identical

The difference is therefore confined to ordering among dense candidates before reranking.

---

# Local RAG Application

The evaluated retrieval pipeline is integrated into the Streamlit application.

The default application path is:

```text
User question
      ↓
Qwen3-Embedding-0.6B query encoding
      ↓
Dense candidate retrieval
      ↓
Top-25 comments
      ↓
BAAI/bge-reranker-v2-m3
      ↓
Top-10 evidence
      ↓
Local Llama 3 via Ollama
      ↓
Grounded answer
```

The application supports:

- single-video retrieval
- global retrieval
- Qwen3 + BGE modern backend
- MiniLM + FAISS legacy backend
- quality-first Top-25 reranking
- CPU-balanced Top-15 reranking
- configurable final context size
- inspection of retrieved comments
- dense retrieval scores
- reranker scores
- links back to source videos

The application automatically uses CUDA when available and otherwise runs on CPU.

---

# End-to-End Smoke Test

The complete application path was tested with:

```text
Video:
dxPhJ0wfp0g

Question:
What laptops or computer setups do viewers recommend for programming?

Candidate depth:
25

Final LLM evidence:
10 comments
```

The application successfully completed:

```text
Qwen3 query encoding
        ↓
Top-25 dense retrieval
        ↓
BGE reranking
        ↓
Top-10 evidence
        ↓
Ollama Llama 3 generation
        ↓
Streamlit answer
```

The first retrieved result had:

```text
Dense score:
0.7160

BGE reranker score:
0.9993
```

matching the validated serving retrieval output.

This confirms functional end-to-end integration.

It does **not** constitute a formal evaluation of answer-generation quality.

---

# Generation Grounding

The generation prompt instructs Llama 3 to:

- use only retrieved comments
- avoid outside knowledge
- explicitly state when evidence is missing
- paraphrase instead of copying comments directly
- produce a concise answer with supporting details

However, retrieval evaluation and answer-generation evaluation are intentionally treated separately.

The current benchmark formally evaluates:

```text
retrieval quality
```

not:

```text
answer factuality
answer completeness
hallucination rate
generation preference
```

A future answer-level benchmark would be required before making quantitative claims about generation quality.

---

# Installation

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

---

# Running Ollama

The application uses a locally served Llama 3 model through Ollama.

Confirm that Ollama is installed and the model is available:

```powershell
ollama list
```

The application currently expects:

```text
llama3
```

A simple test is:

```powershell
ollama run llama3 "Reply with exactly: OK"
```

---

# Running the Application

First build the modern serving index:

```powershell
python processing\build_modern_index_v1.py
```

Then start Streamlit:

```powershell
python -m streamlit run app\main.py
```

The modern backend is selected by default:

```text
Modern: Qwen3 + BGE
```

Two reranking profiles are available:

```text
Quality-first:
Top-25 → BGE

CPU-balanced:
Top-15 → BGE
```

---

# Reproducing the Retrieval Experiments

## 1. Build the retrieval corpus

```powershell
python processing\build_corpus_v1.py
```

---

## 2. Build the initial candidate pool

```powershell
python evaluation\build_candidate_pool.py
```

---

## 3. Evaluate initial baselines

```powershell
python evaluation\evaluate_baselines_v1.py
```

---

## 4. Run the RRF ablation

```powershell
python evaluation\evaluate_rrf_v1.py
```

---

## 5. Build the Qwen3 candidate pool

```powershell
python evaluation\build_qwen3_dense_pool_v1.py
```

Novel Qwen3 candidates must be judged before expanding the qrels.

---

## 6. Expand pooled qrels

```powershell
python evaluation\expand_qrels_with_qwen3.py
```

---

## 7. Evaluate systems on the expanded pool

```powershell
python evaluation\evaluate_expanded_systems_v1.py
```

---

## 8. Evaluate reranker candidate sets

```powershell
python evaluation\evaluate_reranker_candidate_sets_v1.py
```

---

## 9. Statistical validation

```powershell
python evaluation\statistical_significance_v1.py
```

---

## 10. CPU latency benchmark

```powershell
python evaluation\benchmark_latency_v1.py
```

---

## 11. Rerank-depth trade-off

```powershell
python evaluation\evaluate_rerank_depth_tradeoff_v1.py
```

---

## 12. Build modern serving embeddings

```powershell
python processing\build_modern_index_v1.py
```

---

## 13. Validate serving reproduction

```powershell
python evaluation\validate_modern_serving_v1.py
```

Expected final validation:

```text
DENSE CANDIDATE REPRODUCTION: PASS

BGE RERANKING REPRODUCTION: PASS

Public API exact Top10 match: True

MODERN SERVING VALIDATION: PASS
```

---

# Repository Structure

```text
youtube-comments-rag/
│
├── app/
│   ├── main.py
│   ├── modern_retrieval.py
│   ├── rag_pipeline.py
│   ├── models.py
│   ├── preprocess.py
│   └── __init__.py
│
├── ingestion/
│   ├── collect_comments.py
│   ├── collect_dataset.py
│   └── collect_dataset_resume.py
│
├── processing/
│   ├── build_corpus_v1.py
│   └── build_modern_index_v1.py
│
├── evaluation/
│   ├── benchmark_v1_queries.csv
│   ├── benchmark_v1_videos.csv
│   ├── build_candidate_pool.py
│   ├── evaluate_baselines_v1.py
│   ├── evaluate_rrf_v1.py
│   ├── build_qwen3_dense_pool_v1.py
│   ├── expand_qrels_with_qwen3.py
│   ├── evaluate_expanded_systems_v1.py
│   ├── evaluate_reranker_candidate_sets_v1.py
│   ├── statistical_significance_v1.py
│   ├── benchmark_latency_v1.py
│   ├── evaluate_rerank_depth_tradeoff_v1.py
│   └── validate_modern_serving_v1.py
│
├── data/
│   ├── corpus_v1/
│   ├── structured_comments/
│   └── modern_index_v1/
│
├── README.md
├── requirements.txt
└── .gitignore
```

Large raw datasets, reconstructed per-video files, Parquet corpus data, and generated embedding matrices are intentionally excluded from normal Git tracking.

---

# Important Result Artifacts

The repository retains compact experiment summaries so the main README results can be checked against generated outputs.

Important result directories include:

```text
evaluation/expanded_qrels_v1/
evaluation/expanded_results_v1/
evaluation/reranker_candidate_ablation_v1/
evaluation/statistical_significance_v1/
evaluation/latency_v1/
evaluation/rerank_depth_tradeoff_v1/
```

These contain summary CSV and JSON reports for:

- expanded qrels
- retrieval metrics
- candidate-set ablations
- statistical tests
- CPU latency
- reranking-depth experiments

---

# Limitations

## Development benchmark

The 60-query benchmark was used repeatedly during architecture development, including:

- baseline comparison
- RRF rejection
- reranker evaluation
- Qwen3 evaluation
- candidate-set selection
- rerank-depth selection

It should therefore be described as a:

```text
development benchmark
```

rather than an untouched held-out test set.

A stronger evaluation would freeze the architecture first and then construct an additional unseen benchmark.

---

## Single-video benchmark scope

The reported 60-query benchmark evaluates retrieval within a selected video's comment collection.

The application additionally supports global retrieval across the complete corpus.

Global retrieval is operational but is **not represented by the reported benchmark metrics**.

---

## Retrieval versus generation evaluation

The quantitative metrics reported in this project validate retrieval.

They do not directly measure:

- hallucination rate
- answer faithfulness
- answer completeness
- factual consistency
- user preference

The end-to-end Streamlit test confirms that generation works operationally, not that generation quality has been formally benchmarked.

---

## CPU latency

The reported latency values were measured in one local CPU environment.

They should not be interpreted as expected latency on:

- GPU serving
- cloud inference
- optimized ONNX runtimes
- quantized models
- dedicated inference servers

---

## Dataset availability

Some original video IDs were unavailable through the YouTube API during reconstruction.

The final corpus therefore represents the subset that remained accessible during collection.

---

## Human relevance judgments

The benchmark uses a consistent relevance rubric and adjudication workflow, but relevance labels still involve human interpretation.

---

# Future Work

Potential extensions include:

- construct a fully held-out retrieval test benchmark
- build a formal answer-faithfulness evaluation
- measure grounded-generation error rates
- evaluate multilingual retrieval behavior
- compare GPU reranker latency
- evaluate smaller or quantized rerankers
- add approximate nearest-neighbor retrieval for global search
- test FAISS or HNSW for the modern Qwen3 embedding space
- evaluate query-adaptive reranking depth
- add citation-level evidence attribution
- package benchmark artifacts for easier external reproduction

---

# Final Architecture Decision

The current quality-first retrieval architecture is:

```text
Qwen3-Embedding-0.6B
        ↓
Dense Top-25 retrieval
        ↓
BAAI/bge-reranker-v2-m3
        ↓
Top-10 evidence
```

It was selected because it achieved:

```text
nDCG@10 = 0.7347
```

versus:

```text
MiniLM baseline = 0.6365
```

for a relative improvement of:

```text
+15.41%
```

with:

```text
95% CI = [+0.0544, +0.1437]
p = 0.00005
```

on the 60-query development benchmark.

A larger BM25 + MiniLM + Qwen3 candidate union achieved slightly higher raw nDCG:

```text
0.7383
```

but approximately doubled the reranker candidate set and did not show a statistically supported improvement over Qwen3-only candidate generation.

The simpler Qwen3 Top-25 → BGE pipeline was therefore selected as the final quality-first architecture.