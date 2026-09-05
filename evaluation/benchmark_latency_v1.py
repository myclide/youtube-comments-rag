from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder,
)


# ============================================================
# Configuration
# ============================================================

MINILM_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

QWEN_MODEL = (
    "Qwen/Qwen3-Embedding-0.6B"
)

RERANKER_MODEL = (
    "BAAI/bge-reranker-v2-m3"
)

TOP_K = 25

REPEATS = 3

WARMUP_QUERIES = 3

DOCUMENT_BATCH_SIZE = 16

RERANK_BATCH_SIZE = 16


# ============================================================
# Paths
# ============================================================

CORPUS_PATH = Path(
    r"data\corpus_v1"
    r"\retrieval_comments_v1.parquet"
)

QUERY_PATH = Path(
    r"evaluation\benchmark_v1_queries.csv"
)

OUTPUT_DIR = Path(
    r"evaluation\latency_v1"
)

QUERY_LATENCY_PATH = (
    OUTPUT_DIR
    / "query_latency_measurements.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "latency_summary.csv"
)

INDEXING_PATH = (
    OUTPUT_DIR
    / "document_encoding_summary.csv"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "latency_report.json"
)


# ============================================================
# Helpers
# ============================================================

def synchronize_if_needed():
    """
    CPU benchmark currently needs no synchronization.

    Kept as a helper so the script remains explicit if moved
    to GPU later.
    """

    return


def percentile(
    values,
    q,
):

    return float(
        np.percentile(
            np.asarray(
                values,
                dtype=float,
            ),
            q,
        )
    )


def summarize_latencies(
    measurements,
):

    values = np.asarray(
        measurements,
        dtype=float,
    )

    mean_ms = float(
        values.mean()
    )

    median_ms = float(
        np.median(
            values
        )
    )

    p95_ms = percentile(
        values,
        95,
    )

    p99_ms = percentile(
        values,
        99,
    )

    qps = (
        1000.0
        / mean_ms
        if mean_ms > 0
        else float("inf")
    )

    return {
        "measurements":
            len(values),

        "mean_ms":
            mean_ms,

        "median_ms":
            median_ms,

        "p95_ms":
            p95_ms,

        "p99_ms":
            p99_ms,

        "serial_qps":
            qps,
    }


def build_document_id(
    frame,
):

    return (
        frame["video_id"].astype(str)
        + "::"
        + frame[
            "representative_comment_id"
        ].astype(str)
    )


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 90)
    print("ONLINE RETRIEVAL LATENCY BENCHMARK")
    print("=" * 90)

    print()
    print(
        "Document indexing is OFFLINE and excluded "
        "from query latency."
    )

    print(
        "Model loading is excluded from query latency."
    )

    print(
        "Benchmark retrieval scope matches the "
        "60-query single-video evaluation."
    )

    # ========================================================
    # Load corpus / queries
    # ========================================================

    print()
    print("Loading corpus and queries...")

    corpus = pd.read_parquet(
        CORPUS_PATH
    )

    queries = pd.read_csv(
        QUERY_PATH,
        dtype=str,
        keep_default_na=False,
    )

    corpus[
        "video_id"
    ] = (
        corpus[
            "video_id"
        ]
        .astype(str)
    )

    corpus[
        "document_id"
    ] = build_document_id(
        corpus
    )

    benchmark_video_ids = set(
        queries[
            "video_id"
        ]
        .astype(str)
    )

    benchmark_corpus = (
        corpus[
            corpus[
                "video_id"
            ]
            .isin(
                benchmark_video_ids
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    print(
        "Benchmark queries:",
        len(
            queries
        ),
    )

    print(
        "Benchmark videos:",
        len(
            benchmark_video_ids
        ),
    )

    print(
        "Benchmark documents:",
        f"{len(benchmark_corpus):,}",
    )

    if len(
        queries
    ) != 60:

        raise RuntimeError(
            "Expected 60 benchmark queries."
        )

    if len(
        benchmark_video_ids
    ) != 20:

        raise RuntimeError(
            "Expected 20 benchmark videos."
        )

    # ========================================================
    # Group documents by video
    # ========================================================

    documents_by_video = {}

    for video_id, group in benchmark_corpus.groupby(
        "video_id",
        sort=False,
    ):

        documents_by_video[
            video_id
        ] = (
            group[
                [
                    "document_id",
                    "comment_text",
                ]
            ]
            .reset_index(
                drop=True
            )
        )

    # ========================================================
    # Load models
    # ========================================================

    print()
    print("=" * 90)
    print("LOADING MODELS")
    print("=" * 90)

    print()
    print(
        "MiniLM:"
    )
    print(
        MINILM_MODEL
    )

    minilm = SentenceTransformer(
        MINILM_MODEL
    )

    print()
    print(
        "Qwen3:"
    )
    print(
        QWEN_MODEL
    )

    qwen = SentenceTransformer(
        QWEN_MODEL
    )

    qwen.max_seq_length = 512

    print()
    print(
        "BGE reranker:"
    )
    print(
        RERANKER_MODEL
    )

    reranker = CrossEncoder(
        RERANKER_MODEL,
        max_length=512,
    )

    # ========================================================
    # OFFLINE document encoding
    # ========================================================

    print()
    print("=" * 90)
    print("OFFLINE DOCUMENT ENCODING")
    print("=" * 90)

    indexing_rows = []

    minilm_embeddings = {}
    qwen_embeddings = {}

    # --------------------------------------------------------
    # MiniLM documents
    # --------------------------------------------------------

    total_docs = 0

    start = time.perf_counter()

    for position, (
        video_id,
        docs,
    ) in enumerate(
        documents_by_video.items(),
        start=1,
    ):

        texts = (
            docs[
                "comment_text"
            ]
            .astype(str)
            .tolist()
        )

        print(
            f"MiniLM "
            f"[{position:02d}/20] "
            f"{video_id}: "
            f"{len(texts)} docs"
        )

        embeddings = minilm.encode(
            texts,
            batch_size=DOCUMENT_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        minilm_embeddings[
            video_id
        ] = embeddings

        total_docs += len(
            texts
        )

    synchronize_if_needed()

    minilm_index_seconds = (
        time.perf_counter()
        - start
    )

    indexing_rows.append({
        "model":
            "minilm_dense",

        "documents":
            total_docs,

        "seconds":
            minilm_index_seconds,

        "docs_per_second":
            total_docs
            / minilm_index_seconds,
    })

    print()
    print(
        "MiniLM offline encoding:",
        f"{minilm_index_seconds:.2f}s",
    )

    print(
        "MiniLM docs/sec:",
        f"{total_docs / minilm_index_seconds:.2f}",
    )

    # --------------------------------------------------------
    # Qwen3 documents
    # --------------------------------------------------------

    total_docs = 0

    start = time.perf_counter()

    for position, (
        video_id,
        docs,
    ) in enumerate(
        documents_by_video.items(),
        start=1,
    ):

        texts = (
            docs[
                "comment_text"
            ]
            .astype(str)
            .tolist()
        )

        print(
            f"Qwen3 "
            f"[{position:02d}/20] "
            f"{video_id}: "
            f"{len(texts)} docs"
        )

        embeddings = qwen.encode(
            texts,
            batch_size=DOCUMENT_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        qwen_embeddings[
            video_id
        ] = embeddings

        total_docs += len(
            texts
        )

    synchronize_if_needed()

    qwen_index_seconds = (
        time.perf_counter()
        - start
    )

    indexing_rows.append({
        "model":
            "qwen3_embedding_0.6b",

        "documents":
            total_docs,

        "seconds":
            qwen_index_seconds,

        "docs_per_second":
            total_docs
            / qwen_index_seconds,
    })

    print()
    print(
        "Qwen3 offline encoding:",
        f"{qwen_index_seconds:.2f}s",
    )

    print(
        "Qwen3 docs/sec:",
        f"{total_docs / qwen_index_seconds:.2f}",
    )

    # ========================================================
    # Warm-up
    # ========================================================

    print()
    print("=" * 90)
    print("WARM-UP")
    print("=" * 90)

    warmup = (
        queries
        .head(
            WARMUP_QUERIES
        )
    )

    for row in warmup.itertuples(
        index=False
    ):

        query_text = str(
            row.query
        )

        video_id = str(
            row.video_id
        )

        docs = (
            documents_by_video[
                video_id
            ]
        )

        # MiniLM warmup
        minilm_query = minilm.encode(
            query_text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        _ = (
            minilm_embeddings[
                video_id
            ]
            @ minilm_query
        )

        # Qwen warmup
        qwen_query = qwen.encode(
            query_text,
            prompt_name="query",
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        qwen_scores = (
            qwen_embeddings[
                video_id
            ]
            @ qwen_query
        )

        top_indices = np.argsort(
            -qwen_scores
        )[:TOP_K]

        top_comments = (
            docs.iloc[
                top_indices
            ][
                "comment_text"
            ]
            .astype(str)
            .tolist()
        )

        rerank_pairs = [
            (
                query_text,
                comment,
            )
            for comment
            in top_comments
        ]

        _ = reranker.predict(
            rerank_pairs,
            batch_size=RERANK_BATCH_SIZE,
            show_progress_bar=False,
        )

    print(
        "Warm-up complete."
    )

    # ========================================================
    # Online latency benchmark
    # ========================================================

    print()
    print("=" * 90)
    print("ONLINE QUERY BENCHMARK")
    print("=" * 90)

    measurement_rows = []

    for repeat in range(
        1,
        REPEATS + 1,
    ):

        print()
        print(
            f"Repeat {repeat}/{REPEATS}"
        )

        for position, row in enumerate(
            queries.itertuples(
                index=False
            ),
            start=1,
        ):

            query_id = str(
                row.query_id
            )

            query_text = str(
                row.query
            )

            video_id = str(
                row.video_id
            )

            docs = (
                documents_by_video[
                    video_id
                ]
            )

            # =================================================
            # MiniLM dense
            #
            # Includes:
            # query encoding
            # +
            # exact dense scoring
            # +
            # Top25 selection
            # =================================================

            synchronize_if_needed()

            start = time.perf_counter()

            minilm_query = minilm.encode(
                query_text,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            scores = (
                minilm_embeddings[
                    video_id
                ]
                @ minilm_query
            )

            _ = np.argpartition(
                -scores,
                kth=min(
                    TOP_K - 1,
                    len(scores) - 1,
                ),
            )[:TOP_K]

            synchronize_if_needed()

            minilm_ms = (
                time.perf_counter()
                - start
            ) * 1000.0

            measurement_rows.append({
                "repeat":
                    repeat,

                "query_id":
                    query_id,

                "video_id":
                    video_id,

                "system":
                    "minilm_dense",

                "latency_ms":
                    minilm_ms,
            })

            # =================================================
            # Qwen3 dense
            #
            # Includes:
            # query encoding
            # +
            # exact dense scoring
            # +
            # Top25 selection
            # =================================================

            synchronize_if_needed()

            start = time.perf_counter()

            qwen_query = qwen.encode(
                query_text,
                prompt_name="query",
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            qwen_scores = (
                qwen_embeddings[
                    video_id
                ]
                @ qwen_query
            )

            top_indices = np.argpartition(
                -qwen_scores,
                kth=min(
                    TOP_K - 1,
                    len(qwen_scores) - 1,
                ),
            )[:TOP_K]

            # Make Top25 deterministic by score.
            top_indices = (
                top_indices[
                    np.argsort(
                        -qwen_scores[
                            top_indices
                        ]
                    )
                ]
            )

            synchronize_if_needed()

            qwen_dense_ms = (
                time.perf_counter()
                - start
            ) * 1000.0

            measurement_rows.append({
                "repeat":
                    repeat,

                "query_id":
                    query_id,

                "video_id":
                    video_id,

                "system":
                    "qwen3_dense",

                "latency_ms":
                    qwen_dense_ms,
            })

            # =================================================
            # Qwen3 + BGE reranker
            #
            # IMPORTANT:
            #
            # This starts from a fresh online query.
            # Qwen encode/search is included again.
            #
            # Therefore this is true pipeline latency,
            # not merely incremental reranking latency.
            # =================================================

            synchronize_if_needed()

            start = time.perf_counter()

            qwen_query_pipeline = qwen.encode(
                query_text,
                prompt_name="query",
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            pipeline_scores = (
                qwen_embeddings[
                    video_id
                ]
                @ qwen_query_pipeline
            )

            pipeline_top_indices = np.argpartition(
                -pipeline_scores,
                kth=min(
                    TOP_K - 1,
                    len(pipeline_scores) - 1,
                ),
            )[:TOP_K]

            pipeline_top_indices = (
                pipeline_top_indices[
                    np.argsort(
                        -pipeline_scores[
                            pipeline_top_indices
                        ]
                    )
                ]
            )

            top_comments = (
                docs.iloc[
                    pipeline_top_indices
                ][
                    "comment_text"
                ]
                .astype(str)
                .tolist()
            )

            rerank_pairs = [
                (
                    query_text,
                    comment,
                )
                for comment
                in top_comments
            ]

            _ = reranker.predict(
                rerank_pairs,
                batch_size=RERANK_BATCH_SIZE,
                show_progress_bar=False,
            )

            synchronize_if_needed()

            qwen_bge_ms = (
                time.perf_counter()
                - start
            ) * 1000.0

            measurement_rows.append({
                "repeat":
                    repeat,

                "query_id":
                    query_id,

                "video_id":
                    video_id,

                "system":
                    "qwen3_bge_reranker",

                "latency_ms":
                    qwen_bge_ms,
            })

            print(
                f"[{position:02d}/60] "
                f"{query_id} | "
                f"MiniLM={minilm_ms:.1f} ms | "
                f"Qwen3={qwen_dense_ms:.1f} ms | "
                f"Qwen3+BGE={qwen_bge_ms:.1f} ms"
            )

    # ========================================================
    # Build result dataframes
    # ========================================================

    measurements = pd.DataFrame(
        measurement_rows
    )

    indexing = pd.DataFrame(
        indexing_rows
    )

    # ========================================================
    # Summary
    # ========================================================

    summary_rows = []

    system_order = [
        "minilm_dense",
        "qwen3_dense",
        "qwen3_bge_reranker",
    ]

    for system in system_order:

        values = (
            measurements[
                measurements[
                    "system"
                ]
                == system
            ][
                "latency_ms"
            ]
            .to_numpy(
                dtype=float
            )
        )

        stats = summarize_latencies(
            values
        )

        summary_rows.append({
            "system":
                system,

            **stats,
        })

    summary = pd.DataFrame(
        summary_rows
    )

    # ========================================================
    # Incremental BGE cost
    # ========================================================

    qwen_mean = float(
        summary.loc[
            summary[
                "system"
            ]
            == "qwen3_dense",
            "mean_ms",
        ]
        .iloc[0]
    )

    pipeline_mean = float(
        summary.loc[
            summary[
                "system"
            ]
            == "qwen3_bge_reranker",
            "mean_ms",
        ]
        .iloc[0]
    )

    estimated_reranker_increment_ms = (
        pipeline_mean
        - qwen_mean
    )

    # ========================================================
    # Save
    # ========================================================

    measurements.to_csv(
        QUERY_LATENCY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    indexing.to_csv(
        INDEXING_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    report = {
        "benchmark_queries":
            60,

        "benchmark_videos":
            20,

        "repeats":
            REPEATS,

        "online_measurements_per_system":
            60 * REPEATS,

        "retrieval_scope":
            "single_video_benchmark",

        "top_k_before_reranking":
            TOP_K,

        "model_loading_included":
            False,

        "offline_document_encoding_included":
            False,

        "latency_summary":
            summary.to_dict(
                orient="records"
            ),

        "offline_document_encoding":
            indexing.to_dict(
                orient="records"
            ),

        "estimated_reranker_increment_mean_ms":
            estimated_reranker_increment_ms,
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # Console output
    # ========================================================

    print()
    print("=" * 90)
    print("ONLINE LATENCY RESULTS")
    print("=" * 90)

    for row in summary.itertuples(
        index=False
    ):

        print()
        print(
            row.system
        )

        print(
            "  measurements:",
            row.measurements,
        )

        print(
            "  mean:",
            f"{row.mean_ms:.2f} ms",
        )

        print(
            "  median:",
            f"{row.median_ms:.2f} ms",
        )

        print(
            "  p95:",
            f"{row.p95_ms:.2f} ms",
        )

        print(
            "  p99:",
            f"{row.p99_ms:.2f} ms",
        )

        print(
            "  serial QPS:",
            f"{row.serial_qps:.3f}",
        )

    print()
    print("=" * 90)
    print("OFFLINE DOCUMENT ENCODING")
    print("=" * 90)

    print()
    print(
        indexing
        .round(3)
        .to_string(
            index=False
        )
    )

    print()
    print(
        "Estimated BGE reranking incremental mean:",
        f"{estimated_reranker_increment_ms:.2f} ms",
    )

    print()
    print("=" * 90)
    print("LATENCY BENCHMARK: PASS")
    print("=" * 90)


if __name__ == "__main__":
    main()