from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from sentence_transformers import CrossEncoder


# ============================================================
# Configuration
# ============================================================

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

DEPTHS = [
    10,
    15,
    20,
    25,
]

FINAL_TOP_K = 10

RERANK_BATCH_SIZE = 16

LATENCY_REPEATS = 3

WARMUP_QUERIES = 3


# ============================================================
# Paths
# ============================================================

QWEN_TOP25_PATH = Path(
    r"evaluation\qwen3_dense_v1"
    r"\qwen3_dense_top25.csv"
)

QRELS_PATH = Path(
    r"evaluation\expanded_qrels_v1"
    r"\benchmark_v1_qrels_expanded_qwen3.csv"
)

QUERY_PATH = Path(
    r"evaluation\benchmark_v1_queries.csv"
)

PRIOR_LATENCY_PATH = Path(
    r"evaluation\latency_v1"
    r"\query_latency_measurements.csv"
)

OUTPUT_DIR = Path(
    r"evaluation\rerank_depth_tradeoff_v1"
)

PER_QUERY_QUALITY_PATH = (
    OUTPUT_DIR
    / "rerank_depth_quality_per_query.csv"
)

QUALITY_SUMMARY_PATH = (
    OUTPUT_DIR
    / "rerank_depth_quality_summary.csv"
)

LATENCY_MEASUREMENTS_PATH = (
    OUTPUT_DIR
    / "rerank_depth_latency_measurements.csv"
)

TRADEOFF_SUMMARY_PATH = (
    OUTPUT_DIR
    / "rerank_depth_tradeoff_summary.csv"
)

TOP10_RESULTS_PATH = (
    OUTPUT_DIR
    / "rerank_depth_top10_results.csv"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "rerank_depth_tradeoff_report.json"
)


# ============================================================
# Metric helpers
# ============================================================

def dcg_at_k(
    relevances,
    k,
):

    values = np.asarray(
        relevances[:k],
        dtype=float,
    )

    if len(values) == 0:
        return 0.0

    gains = (
        np.power(
            2.0,
            values,
        )
        - 1.0
    )

    discounts = np.log2(
        np.arange(
            2,
            len(values) + 2,
        )
    )

    return float(
        np.sum(
            gains / discounts
        )
    )


def ndcg_at_k(
    retrieved_relevances,
    all_judged_relevances,
    k,
):

    actual = dcg_at_k(
        retrieved_relevances,
        k,
    )

    ideal_relevances = sorted(
        all_judged_relevances,
        reverse=True,
    )[:k]

    ideal = dcg_at_k(
        ideal_relevances,
        k,
    )

    if ideal == 0:
        return 0.0

    return actual / ideal


def reciprocal_rank_at_k(
    relevances,
    k,
    threshold=1,
):

    for rank, relevance in enumerate(
        relevances[:k],
        start=1,
    ):

        if relevance >= threshold:
            return 1.0 / rank

    return 0.0


def precision_at_k(
    relevances,
    k,
    threshold=1,
):

    top_k = relevances[:k]

    return (
        sum(
            relevance >= threshold
            for relevance in top_k
        )
        / k
    )


def summarize_latency(
    values,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    return {
        "mean_ms":
            float(
                np.mean(values)
            ),

        "median_ms":
            float(
                np.median(values)
            ),

        "p95_ms":
            float(
                np.percentile(
                    values,
                    95,
                )
            ),

        "p99_ms":
            float(
                np.percentile(
                    values,
                    99,
                )
            ),
    }


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 90)
    print("QWEN3 + BGE — RERANK DEPTH QUALITY/LATENCY TRADE-OFF")
    print("=" * 90)

    # ========================================================
    # Load data
    # ========================================================

    qwen = pd.read_csv(
        QWEN_TOP25_PATH,
        dtype={
            "query_id": str,
            "document_id": str,
        },
        keep_default_na=False,
    )

    qrels = pd.read_csv(
        QRELS_PATH,
        dtype={
            "query_id": str,
            "document_id": str,
        },
    )

    queries = pd.read_csv(
        QUERY_PATH,
        dtype=str,
        keep_default_na=False,
    )

    prior_latency = pd.read_csv(
        PRIOR_LATENCY_PATH,
        dtype={
            "query_id": str,
            "video_id": str,
        },
    )

    # ========================================================
    # Numeric conversions
    # ========================================================

    qwen[
        "qwen3_rank"
    ] = pd.to_numeric(
        qwen[
            "qwen3_rank"
        ],
        errors="raise",
    ).astype(int)

    qrels[
        "relevance"
    ] = (
        qrels[
            "relevance"
        ]
        .astype(int)
    )

    prior_latency[
        "repeat"
    ] = (
        prior_latency[
            "repeat"
        ]
        .astype(int)
    )

    prior_latency[
        "latency_ms"
    ] = (
        prior_latency[
            "latency_ms"
        ]
        .astype(float)
    )

    # ========================================================
    # Input integrity
    # ========================================================

    print()
    print("INPUT INTEGRITY")
    print("-" * 90)

    print(
        "Qwen rows:",
        f"{len(qwen):,}",
    )

    print(
        "Qwen queries:",
        qwen[
            "query_id"
        ].nunique(),
    )

    print(
        "Expanded qrels:",
        f"{len(qrels):,}",
    )

    print(
        "Benchmark queries:",
        len(queries),
    )

    if len(qwen) != 1500:

        raise RuntimeError(
            "Expected 1,500 Qwen Top25 rows."
        )

    if (
        qwen[
            "query_id"
        ]
        .nunique()
        != 60
    ):

        raise RuntimeError(
            "Expected 60 Qwen queries."
        )

    if len(queries) != 60:

        raise RuntimeError(
            "Expected 60 benchmark queries."
        )

    duplicate_qwen_pairs = int(
        qwen[
            [
                "query_id",
                "document_id",
            ]
        ]
        .duplicated()
        .sum()
    )

    duplicate_qrels = int(
        qrels[
            [
                "query_id",
                "document_id",
            ]
        ]
        .duplicated()
        .sum()
    )

    print(
        "Duplicate Qwen pairs:",
        duplicate_qwen_pairs,
    )

    print(
        "Duplicate qrels:",
        duplicate_qrels,
    )

    if duplicate_qwen_pairs != 0:

        raise RuntimeError(
            "Duplicate Qwen query-document pairs."
        )

    if duplicate_qrels != 0:

        raise RuntimeError(
            "Duplicate qrels."
        )

    # ========================================================
    # Check ranks 1..25 for every query
    # ========================================================

    for query_id, group in qwen.groupby(
        "query_id",
        sort=False,
    ):

        ranks = sorted(
            group[
                "qwen3_rank"
            ]
            .tolist()
        )

        if ranks != list(
            range(
                1,
                26,
            )
        ):

            raise RuntimeError(
                f"{query_id}: invalid Qwen ranks."
            )

    print(
        "Qwen Top25 ranks: PASS"
    )

    # ========================================================
    # Prior Qwen dense latency
    # ========================================================

    qwen_dense_latency = (
        prior_latency[
            prior_latency[
                "system"
            ]
            == "qwen3_dense"
        ][
            [
                "repeat",
                "query_id",
                "latency_ms",
            ]
        ]
        .rename(
            columns={
                "latency_ms":
                    "qwen_dense_latency_ms"
            }
        )
        .copy()
    )

    print()
    print(
        "Prior Qwen dense latency measurements:",
        len(
            qwen_dense_latency
        ),
    )

    if (
        len(
            qwen_dense_latency
        )
        != 60 * LATENCY_REPEATS
    ):

        raise RuntimeError(
            "Expected 180 prior Qwen dense latency measurements."
        )

    duplicate_latency_keys = int(
        qwen_dense_latency[
            [
                "repeat",
                "query_id",
            ]
        ]
        .duplicated()
        .sum()
    )

    if duplicate_latency_keys != 0:

        raise RuntimeError(
            "Duplicate prior latency query/repeat keys."
        )

    # ========================================================
    # Qrels lookup
    # ========================================================

    qrels_lookup = {}

    for query_id, group in qrels.groupby(
        "query_id",
        sort=False,
    ):

        qrels_lookup[
            query_id
        ] = dict(
            zip(
                group[
                    "document_id"
                ],
                group[
                    "relevance"
                ],
            )
        )

    # ========================================================
    # Query metadata
    # ========================================================

    query_metadata = (
        queries
        .set_index(
            "query_id"
        )
        .to_dict(
            orient="index"
        )
    )

    query_ids = (
        queries[
            "query_id"
        ]
        .tolist()
    )

    # ========================================================
    # Verify all Qwen Top25 are judged
    # ========================================================

    unjudged_total = 0

    for query_id, group in qwen.groupby(
        "query_id",
        sort=False,
    ):

        query_qrels = (
            qrels_lookup[
                query_id
            ]
        )

        unjudged = [
            document_id
            for document_id
            in group[
                "document_id"
            ]
            if document_id
            not in query_qrels
        ]

        unjudged_total += len(
            unjudged
        )

    print(
        "Unjudged Qwen Top25 pairs:",
        unjudged_total,
    )

    if unjudged_total != 0:

        raise RuntimeError(
            "Qwen Top25 contains unjudged documents."
        )

    # ========================================================
    # Load reranker
    # ========================================================

    print()
    print("=" * 90)
    print("LOADING RERANKER")
    print("=" * 90)

    print()
    print(
        RERANKER_MODEL
    )

    model = CrossEncoder(
        RERANKER_MODEL,
        max_length=512,
    )

    # ========================================================
    # Warm-up
    # ========================================================

    print()
    print("=" * 90)
    print("WARM-UP")
    print("=" * 90)

    for row in queries.head(
        WARMUP_QUERIES
    ).itertuples(
        index=False
    ):

        query_id = str(
            row.query_id
        )

        query_text = str(
            row.query
        )

        candidates = (
            qwen[
                qwen[
                    "query_id"
                ]
                == query_id
            ]
            .sort_values(
                "qwen3_rank"
            )
            .head(25)
        )

        pairs = [
            (
                query_text,
                str(comment),
            )
            for comment
            in candidates[
                "comment_text"
            ]
        ]

        _ = model.predict(
            pairs,
            batch_size=RERANK_BATCH_SIZE,
            show_progress_bar=False,
        )

    print(
        "Warm-up complete."
    )

    # ========================================================
    # QUALITY
    #
    # Score each query's full Top25 ONCE.
    # Then depth 10/15/20/25 simply restricts the candidate
    # set before sorting by the same BGE score.
    # ========================================================

    print()
    print("=" * 90)
    print("QUALITY EVALUATION")
    print("=" * 90)

    quality_rows = []

    top10_rows = []

    scored_candidates = {}

    for position, query_id in enumerate(
        query_ids,
        start=1,
    ):

        metadata = (
            query_metadata[
                query_id
            ]
        )

        query_text = str(
            metadata[
                "query"
            ]
        )

        candidates = (
            qwen[
                qwen[
                    "query_id"
                ]
                == query_id
            ]
            .sort_values(
                "qwen3_rank"
            )
            .copy()
            .reset_index(
                drop=True
            )
        )

        pairs = [
            (
                query_text,
                str(comment),
            )
            for comment
            in candidates[
                "comment_text"
            ]
        ]

        scores = model.predict(
            pairs,
            batch_size=RERANK_BATCH_SIZE,
            show_progress_bar=False,
        )

        scores = (
            np.asarray(
                scores
            )
            .reshape(-1)
        )

        if len(scores) != 25:

            raise RuntimeError(
                f"{query_id}: expected 25 reranker scores."
            )

        candidates[
            "reranker_score"
        ] = scores

        scored_candidates[
            query_id
        ] = candidates

        query_qrels = (
            qrels_lookup[
                query_id
            ]
        )

        all_judged_relevances = list(
            query_qrels.values()
        )

        print(
            f"[{position:02d}/60] {query_id}"
        )

        for depth in DEPTHS:

            depth_candidates = (
                candidates[
                    candidates[
                        "qwen3_rank"
                    ]
                    <= depth
                ]
                .copy()
            )

            reranked = (
                depth_candidates
                .sort_values(
                    by=[
                        "reranker_score",
                        "document_id",
                    ],
                    ascending=[
                        False,
                        True,
                    ],
                )
                .head(
                    FINAL_TOP_K
                )
                .copy()
            )

            if len(reranked) != FINAL_TOP_K:

                raise RuntimeError(
                    f"{query_id}/depth={depth}: "
                    "fewer than 10 final results."
                )

            relevances = [
                int(
                    query_qrels[
                        document_id
                    ]
                )
                for document_id
                in reranked[
                    "document_id"
                ]
            ]

            ndcg = ndcg_at_k(
                relevances,
                all_judged_relevances,
                FINAL_TOP_K,
            )

            mrr = reciprocal_rank_at_k(
                relevances,
                FINAL_TOP_K,
                threshold=1,
            )

            precision = precision_at_k(
                relevances,
                FINAL_TOP_K,
                threshold=1,
            )

            strict_mrr = reciprocal_rank_at_k(
                relevances,
                FINAL_TOP_K,
                threshold=2,
            )

            strict_precision = precision_at_k(
                relevances,
                FINAL_TOP_K,
                threshold=2,
            )

            quality_rows.append({
                "depth":
                    depth,

                "query_id":
                    query_id,

                "query_type":
                    metadata[
                        "query_type"
                    ],

                "query":
                    query_text,

                "ndcg_at_10":
                    ndcg,

                "mrr_at_10":
                    mrr,

                "precision_at_10":
                    precision,

                "strict_mrr_at_10_rel2":
                    strict_mrr,

                "strict_precision_at_10_rel2":
                    strict_precision,

                "top10_rel0":
                    relevances.count(
                        0
                    ),

                "top10_rel1":
                    relevances.count(
                        1
                    ),

                "top10_rel2":
                    relevances.count(
                        2
                    ),
            })

            for rank, result in enumerate(
                reranked.itertuples(
                    index=False
                ),
                start=1,
            ):

                top10_rows.append({
                    "depth":
                        depth,

                    "query_id":
                        query_id,

                    "rank":
                        rank,

                    "qwen3_rank":
                        result.qwen3_rank,

                    "document_id":
                        result.document_id,

                    "reranker_score":
                        result.reranker_score,

                    "relevance":
                        query_qrels[
                            result.document_id
                        ],

                    "comment_text":
                        result.comment_text,
                })

    quality = pd.DataFrame(
        quality_rows
    )

    top10_results = pd.DataFrame(
        top10_rows
    )

    metric_columns = [
        "ndcg_at_10",
        "mrr_at_10",
        "precision_at_10",
        "strict_mrr_at_10_rel2",
        "strict_precision_at_10_rel2",
    ]

    quality_summary = (
        quality
        .groupby(
            "depth",
            as_index=False,
        )[
            metric_columns
        ]
        .mean()
        .sort_values(
            "depth"
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # QUALITY RETENTION VS TOP25
    # ========================================================

    top25_quality = (
        quality_summary[
            quality_summary[
                "depth"
            ]
            == 25
        ]
        .iloc[0]
    )

    quality_summary[
        "ndcg_retention_vs_25"
    ] = (
        quality_summary[
            "ndcg_at_10"
        ]
        / top25_quality[
            "ndcg_at_10"
        ]
    )

    quality_summary[
        "ndcg_delta_vs_25"
    ] = (
        quality_summary[
            "ndcg_at_10"
        ]
        - top25_quality[
            "ndcg_at_10"
        ]
    )

    # ========================================================
    # LATENCY
    #
    # Time BGE only.
    #
    # Later merge with previously measured Qwen dense latency
    # using the same repeat/query_id keys.
    # ========================================================

    print()
    print("=" * 90)
    print("RERANKER LATENCY BENCHMARK")
    print("=" * 90)

    latency_rows = []

    for repeat in range(
        1,
        LATENCY_REPEATS + 1,
    ):

        print()
        print(
            f"Repeat {repeat}/{LATENCY_REPEATS}"
        )

        for position, query_id in enumerate(
            query_ids,
            start=1,
        ):

            metadata = (
                query_metadata[
                    query_id
                ]
            )

            query_text = str(
                metadata[
                    "query"
                ]
            )

            candidates = (
                scored_candidates[
                    query_id
                ]
            )

            depth_messages = []

            for depth in DEPTHS:

                depth_candidates = (
                    candidates[
                        candidates[
                            "qwen3_rank"
                        ]
                        <= depth
                    ]
                    .sort_values(
                        "qwen3_rank"
                    )
                )

                pairs = [
                    (
                        query_text,
                        str(comment),
                    )
                    for comment
                    in depth_candidates[
                        "comment_text"
                    ]
                ]

                start = (
                    time.perf_counter()
                )

                _ = model.predict(
                    pairs,
                    batch_size=RERANK_BATCH_SIZE,
                    show_progress_bar=False,
                )

                reranker_ms = (
                    time.perf_counter()
                    - start
                ) * 1000.0

                latency_rows.append({
                    "repeat":
                        repeat,

                    "query_id":
                        query_id,

                    "depth":
                        depth,

                    "reranker_latency_ms":
                        reranker_ms,
                })

                depth_messages.append(
                    f"D{depth}={reranker_ms:.1f}ms"
                )

            print(
                f"[{position:02d}/60] "
                f"{query_id} | "
                + " | ".join(
                    depth_messages
                )
            )

    latency = pd.DataFrame(
        latency_rows
    )

    # ========================================================
    # Merge prior measured Qwen dense latency
    # ========================================================

    latency = (
        latency
        .merge(
            qwen_dense_latency,
            on=[
                "repeat",
                "query_id",
            ],
            how="left",
            validate="many_to_one",
        )
    )

    missing_dense_latency = int(
        latency[
            "qwen_dense_latency_ms"
        ]
        .isna()
        .sum()
    )

    if missing_dense_latency != 0:

        raise RuntimeError(
            "Missing prior Qwen dense latency measurements."
        )

    latency[
        "estimated_pipeline_latency_ms"
    ] = (
        latency[
            "qwen_dense_latency_ms"
        ]
        + latency[
            "reranker_latency_ms"
        ]
    )

    # ========================================================
    # Build latency summaries
    # ========================================================

    latency_summary_rows = []

    for depth in DEPTHS:

        subset = (
            latency[
                latency[
                    "depth"
                ]
                == depth
            ]
        )

        reranker_stats = (
            summarize_latency(
                subset[
                    "reranker_latency_ms"
                ]
            )
        )

        pipeline_stats = (
            summarize_latency(
                subset[
                    "estimated_pipeline_latency_ms"
                ]
            )
        )

        latency_summary_rows.append({
            "depth":
                depth,

            "reranker_mean_ms":
                reranker_stats[
                    "mean_ms"
                ],

            "reranker_median_ms":
                reranker_stats[
                    "median_ms"
                ],

            "reranker_p95_ms":
                reranker_stats[
                    "p95_ms"
                ],

            "pipeline_mean_ms":
                pipeline_stats[
                    "mean_ms"
                ],

            "pipeline_median_ms":
                pipeline_stats[
                    "median_ms"
                ],

            "pipeline_p95_ms":
                pipeline_stats[
                    "p95_ms"
                ],

            "pipeline_p99_ms":
                pipeline_stats[
                    "p99_ms"
                ],

            "estimated_serial_qps":
                (
                    1000.0
                    / pipeline_stats[
                        "mean_ms"
                    ]
                ),
        })

    latency_summary = pd.DataFrame(
        latency_summary_rows
    )

    # ========================================================
    # Join quality + latency
    # ========================================================

    tradeoff = (
        quality_summary
        .merge(
            latency_summary,
            on="depth",
            validate="one_to_one",
        )
        .sort_values(
            "depth"
        )
        .reset_index(
            drop=True
        )
    )

    top25_latency = float(
        tradeoff.loc[
            tradeoff[
                "depth"
            ]
            == 25,
            "pipeline_mean_ms",
        ]
        .iloc[0]
    )

    tradeoff[
        "mean_latency_reduction_vs_25"
    ] = (
        1.0
        - (
            tradeoff[
                "pipeline_mean_ms"
            ]
            / top25_latency
        )
    )

    # ========================================================
    # Save
    # ========================================================

    quality.to_csv(
        PER_QUERY_QUALITY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    quality_summary.to_csv(
        QUALITY_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    latency.to_csv(
        LATENCY_MEASUREMENTS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    tradeoff.to_csv(
        TRADEOFF_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    top10_results.to_csv(
        TOP10_RESULTS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Report
    # ========================================================

    report = {
        "queries":
            60,

        "depths":
            DEPTHS,

        "final_top_k":
            FINAL_TOP_K,

        "latency_repeats":
            LATENCY_REPEATS,

        "latency_method":
            (
                "Measured BGE reranker latency plus "
                "paired previously measured Qwen3 dense latency."
            ),

        "tradeoff":
            tradeoff.to_dict(
                orient="records"
            ),
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
    # Console results
    # ========================================================

    print()
    print("=" * 90)
    print("QUALITY RESULTS")
    print("=" * 90)

    for row in quality_summary.itertuples(
        index=False
    ):

        print()
        print(
            f"Top{row.depth} -> BGE"
        )

        print(
            "  nDCG@10:",
            f"{row.ndcg_at_10:.4f}",
        )

        print(
            "  MRR@10:",
            f"{row.mrr_at_10:.4f}",
        )

        print(
            "  Precision@10:",
            f"{row.precision_at_10:.4f}",
        )

        print(
            "  Strict MRR@10:",
            f"{row.strict_mrr_at_10_rel2:.4f}",
        )

        print(
            "  Strict Precision@10:",
            f"{row.strict_precision_at_10_rel2:.4f}",
        )

        print(
            "  nDCG retention vs Top25:",
            f"{row.ndcg_retention_vs_25 * 100:.2f}%",
        )

    print()
    print("=" * 90)
    print("QUALITY-LATENCY TRADE-OFF")
    print("=" * 90)

    display_columns = [
        "depth",
        "ndcg_at_10",
        "mrr_at_10",
        "precision_at_10",
        "ndcg_retention_vs_25",
        "pipeline_mean_ms",
        "pipeline_p95_ms",
        "estimated_serial_qps",
        "mean_latency_reduction_vs_25",
    ]

    display = (
        tradeoff[
            display_columns
        ]
        .copy()
    )

    print()
    print(
        display
        .round(
            {
                "ndcg_at_10": 4,
                "mrr_at_10": 4,
                "precision_at_10": 4,
                "ndcg_retention_vs_25": 4,
                "pipeline_mean_ms": 2,
                "pipeline_p95_ms": 2,
                "estimated_serial_qps": 3,
                "mean_latency_reduction_vs_25": 4,
            }
        )
        .to_string(
            index=False
        )
    )

    # ========================================================
    # Simple candidate recommendations
    # ========================================================

    print()
    print("=" * 90)
    print("DEPTH CANDIDATES")
    print("=" * 90)

    for threshold in [
        0.99,
        0.98,
        0.95,
    ]:

        eligible = (
            tradeoff[
                tradeoff[
                    "ndcg_retention_vs_25"
                ]
                >= threshold
            ]
            .sort_values(
                [
                    "depth",
                ]
            )
        )

        if not eligible.empty:

            best = (
                eligible.iloc[0]
            )

            print(
                f"Smallest depth retaining >= "
                f"{threshold * 100:.0f}% "
                f"of Top25 nDCG: "
                f"Top{int(best['depth'])}"
            )

    print()
    print("=" * 90)
    print("OUTPUT FILES")
    print("=" * 90)

    print()
    print(
        PER_QUERY_QUALITY_PATH
    )

    print(
        QUALITY_SUMMARY_PATH
    )

    print(
        LATENCY_MEASUREMENTS_PATH
    )

    print(
        TRADEOFF_SUMMARY_PATH
    )

    print(
        TOP10_RESULTS_PATH
    )

    print(
        REPORT_PATH
    )

    print()
    print(
        "RERANK DEPTH TRADE-OFF: PASS"
    )


if __name__ == "__main__":
    main()