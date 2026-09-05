from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

K = 10

# For binary metrics:
#
# relevance 0 = non-relevant
# relevance 1 = relevant
# relevance 2 = highly relevant
#
# MRR / Precision treat >= 1 as relevant.
BINARY_RELEVANCE_THRESHOLD = 1


# ============================================================
# Paths
# ============================================================

POOL_PATH = Path(
    r"evaluation\candidate_pool_v1"
    r"\candidate_pool_benchmark_v1.csv"
)

QRELS_PATH = Path(
    r"evaluation\candidate_pool_v1"
    r"\benchmark_v1_qrels.csv"
)

OUTPUT_DIR = Path(
    r"evaluation\baseline_results_v1"
)

PER_QUERY_PATH = (
    OUTPUT_DIR
    / "baseline_metrics_per_query.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "baseline_metrics_summary.csv"
)

TYPE_SUMMARY_PATH = (
    OUTPUT_DIR
    / "baseline_metrics_by_query_type.csv"
)

TOP10_AUDIT_PATH = (
    OUTPUT_DIR
    / "baseline_top10_judgments.csv"
)

COMPARISON_PATH = (
    OUTPUT_DIR
    / "baseline_system_comparison.csv"
)


# ============================================================
# Systems
# ============================================================

SYSTEMS = {
    "bm25": {
        "rank_column":
            "bm25_rank",

        "score_column":
            "bm25_score",
    },

    "minilm_dense": {
        "rank_column":
            "dense_rank",

        "score_column":
            "dense_score",
    },
}


# ============================================================
# Metric functions
# ============================================================

def dcg_at_k(
    relevances,
    k,
):
    """
    Graded DCG using:

        gain = 2^rel - 1

    DCG =
        sum(
            gain_i / log2(rank_i + 1)
        )

    rank starts from 1.
    """

    relevances = (
        np.asarray(
            relevances[:k],
            dtype=float,
        )
    )

    if len(relevances) == 0:
        return 0.0

    gains = (
        np.power(
            2.0,
            relevances,
        )
        - 1.0
    )

    discounts = np.log2(
        np.arange(
            2,
            len(relevances) + 2,
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
    """
    nDCG@k.

    Actual DCG is based on the system ranking.

    Ideal DCG is based on the highest graded-relevance
    documents available in the frozen qrels.
    """

    actual_dcg = dcg_at_k(
        retrieved_relevances,
        k,
    )

    ideal_relevances = sorted(
        all_judged_relevances,
        reverse=True,
    )[:k]

    ideal_dcg = dcg_at_k(
        ideal_relevances,
        k,
    )

    if ideal_dcg == 0:
        return 0.0

    return (
        actual_dcg
        / ideal_dcg
    )


def reciprocal_rank_at_k(
    relevances,
    k,
    threshold=1,
):
    """
    Reciprocal rank of the first document whose
    relevance >= threshold.
    """

    for rank, relevance in enumerate(
        relevances[:k],
        start=1,
    ):

        if relevance >= threshold:

            return (
                1.0
                / rank
            )

    return 0.0


def precision_at_k(
    relevances,
    k,
    threshold=1,
):
    """
    Precision@k.

    relevance >= threshold is binary relevant.
    """

    top_k = (
        relevances[:k]
    )

    relevant_count = sum(
        relevance >= threshold
        for relevance
        in top_k
    )

    return (
        relevant_count
        / k
    )


def hit_rate_at_k(
    relevances,
    k,
    threshold=1,
):
    """
    Diagnostic metric only.

    Returns 1 when at least one relevant document
    appears in Top-k.
    """

    return float(
        any(
            relevance >= threshold
            for relevance
            in relevances[:k]
        )
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
    print("BENCHMARK V1 BASELINE EVALUATION")
    print("=" * 90)

    # ========================================================
    # Load frozen retrieval pool
    # ========================================================

    pool = pd.read_csv(
        POOL_PATH,
        dtype={
            "query_id":
                str,

            "video_id":
                str,

            "document_id":
                str,
        },
        keep_default_na=False,
    )

    # Convert ranks/scores explicitly.
    for column in [
        "bm25_rank",
        "dense_rank",
        "bm25_score",
        "dense_score",
    ]:

        pool[column] = pd.to_numeric(
            pool[column],
            errors="coerce",
        )

    # ========================================================
    # Load frozen qrels
    # ========================================================

    qrels = pd.read_csv(
        QRELS_PATH,
        dtype={
            "query_id":
                str,

            "document_id":
                str,
        },
    )

    qrels[
        "relevance"
    ] = (
        qrels[
            "relevance"
        ]
        .astype(int)
    )

    # ========================================================
    # Integrity
    # ========================================================

    print()
    print("INPUT INTEGRITY")
    print("-" * 90)

    print(
        "Pool rows:",
        f"{len(pool):,}",
    )

    print(
        "Qrels rows:",
        f"{len(qrels):,}",
    )

    print(
        "Pool queries:",
        pool[
            "query_id"
        ].nunique(),
    )

    print(
        "Qrels queries:",
        qrels[
            "query_id"
        ].nunique(),
    )

    qrels_duplicates = (
        qrels[
            [
                "query_id",
                "document_id",
            ]
        ]
        .duplicated()
        .sum()
    )

    pool_duplicates = (
        pool[
            [
                "query_id",
                "document_id",
            ]
        ]
        .duplicated()
        .sum()
    )

    print(
        "Duplicate qrel pairs:",
        qrels_duplicates,
    )

    print(
        "Duplicate pool pairs:",
        pool_duplicates,
    )

    if qrels_duplicates != 0:

        raise RuntimeError(
            "Duplicate query-document pairs in qrels."
        )

    if pool_duplicates != 0:

        raise RuntimeError(
            "Duplicate query-document pairs in pool."
        )

    if (
        qrels[
            "query_id"
        ].nunique()
        != 60
    ):

        raise RuntimeError(
            "Expected 60 queries in qrels."
        )

    # ========================================================
    # Build qrels lookup
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
    # Evaluate systems
    # ========================================================

    metric_rows = []

    audit_rows = []

    query_ids = (
        pool[
            "query_id"
        ]
        .drop_duplicates()
        .tolist()
    )

    for system_name, system_config in (
        SYSTEMS.items()
    ):

        rank_column = (
            system_config[
                "rank_column"
            ]
        )

        score_column = (
            system_config[
                "score_column"
            ]
        )

        print()
        print("=" * 90)

        print(
            "SYSTEM:",
            system_name,
        )

        print("=" * 90)

        for position, query_id in enumerate(
            query_ids,
            start=1,
        ):

            query_pool = (
                pool[
                    pool[
                        "query_id"
                    ]
                    == query_id
                ]
                .copy()
            )

            # -----------------------------------------------
            # Query metadata
            # -----------------------------------------------

            query_text = (
                query_pool[
                    "query"
                ].iloc[0]
            )

            query_type = (
                query_pool[
                    "query_type"
                ].iloc[0]
            )

            video_id = (
                query_pool[
                    "video_id"
                ].iloc[0]
            )

            # -----------------------------------------------
            # Frozen Top-K system ranking
            # -----------------------------------------------

            ranked = (
                query_pool[
                    query_pool[
                        rank_column
                    ]
                    .notna()
                ]
                .sort_values(
                    rank_column,
                    ascending=True,
                )
                .head(K)
                .copy()
            )

            # -----------------------------------------------
            # We expect exactly Top-10 because candidate
            # pooling stored Top-25 for both baselines.
            # -----------------------------------------------

            if len(ranked) != K:

                raise RuntimeError(
                    f"{system_name} / {query_id}: "
                    f"expected {K} ranked documents, "
                    f"found {len(ranked)}."
                )

            actual_ranks = (
                ranked[
                    rank_column
                ]
                .astype(int)
                .tolist()
            )

            expected_ranks = list(
                range(
                    1,
                    K + 1,
                )
            )

            if actual_ranks != expected_ranks:

                raise RuntimeError(
                    f"{system_name} / {query_id}: "
                    f"Top-{K} ranks are not 1..{K}. "
                    f"Found {actual_ranks}"
                )

            # -----------------------------------------------
            # Every baseline Top-10 MUST be judged.
            # -----------------------------------------------

            query_qrels = (
                qrels_lookup[
                    query_id
                ]
            )

            unjudged = [
                document_id
                for document_id
                in ranked[
                    "document_id"
                ]
                if document_id
                not in query_qrels
            ]

            if unjudged:

                raise RuntimeError(
                    f"{system_name} / {query_id}: "
                    f"{len(unjudged)} Top-{K} docs "
                    f"are unjudged."
                )

            retrieved_relevances = [
                int(
                    query_qrels[
                        document_id
                    ]
                )
                for document_id
                in ranked[
                    "document_id"
                ]
            ]

            all_judged_relevances = list(
                query_qrels.values()
            )

            # -----------------------------------------------
            # Metrics
            # -----------------------------------------------

            ndcg = ndcg_at_k(
                retrieved_relevances,
                all_judged_relevances,
                K,
            )

            mrr = reciprocal_rank_at_k(
                retrieved_relevances,
                K,
                threshold=(
                    BINARY_RELEVANCE_THRESHOLD
                ),
            )

            precision = precision_at_k(
                retrieved_relevances,
                K,
                threshold=(
                    BINARY_RELEVANCE_THRESHOLD
                ),
            )

            hit_rate = hit_rate_at_k(
                retrieved_relevances,
                K,
                threshold=(
                    BINARY_RELEVANCE_THRESHOLD
                ),
            )

            # Strict diagnostics:
            #
            # only relevance=2 counts as relevant.
            strict_mrr = (
                reciprocal_rank_at_k(
                    retrieved_relevances,
                    K,
                    threshold=2,
                )
            )

            strict_precision = (
                precision_at_k(
                    retrieved_relevances,
                    K,
                    threshold=2,
                )
            )

            metric_rows.append({
                "system":
                    system_name,

                "query_id":
                    query_id,

                "video_id":
                    video_id,

                "query_type":
                    query_type,

                "query":
                    query_text,

                "ndcg_at_10":
                    ndcg,

                "mrr_at_10":
                    mrr,

                "precision_at_10":
                    precision,

                "hit_rate_at_10":
                    hit_rate,

                "strict_mrr_at_10_rel2":
                    strict_mrr,

                "strict_precision_at_10_rel2":
                    strict_precision,

                "top10_rel0":
                    retrieved_relevances.count(
                        0
                    ),

                "top10_rel1":
                    retrieved_relevances.count(
                        1
                    ),

                "top10_rel2":
                    retrieved_relevances.count(
                        2
                    ),
            })

            # -----------------------------------------------
            # Top-10 audit rows
            # -----------------------------------------------

            for _, row in ranked.iterrows():

                document_id = (
                    row[
                        "document_id"
                    ]
                )

                relevance = int(
                    query_qrels[
                        document_id
                    ]
                )

                audit_rows.append({
                    "system":
                        system_name,

                    "query_id":
                        query_id,

                    "video_id":
                        video_id,

                    "query_type":
                        query_type,

                    "query":
                        query_text,

                    "rank":
                        int(
                            row[
                                rank_column
                            ]
                        ),

                    "score":
                        row[
                            score_column
                        ],

                    "document_id":
                        document_id,

                    "relevance":
                        relevance,

                    "comment_text":
                        row[
                            "comment_text"
                        ],
                })

        print(
            f"Evaluated {len(query_ids)} queries."
        )

    # ========================================================
    # Save per-query results
    # ========================================================

    metrics = pd.DataFrame(
        metric_rows
    )

    metrics.to_csv(
        PER_QUERY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Overall macro-average
    # ========================================================

    metric_columns = [
        "ndcg_at_10",
        "mrr_at_10",
        "precision_at_10",
        "hit_rate_at_10",
        "strict_mrr_at_10_rel2",
        "strict_precision_at_10_rel2",
    ]

    summary = (
        metrics
        .groupby(
            "system",
            as_index=False,
        )[
            metric_columns
        ]
        .mean()
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Query-type summary
    # ========================================================

    type_summary = (
        metrics
        .groupby(
            [
                "system",
                "query_type",
            ],
            as_index=False,
        )[
            metric_columns
        ]
        .mean()
    )

    type_summary.to_csv(
        TYPE_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Direct BM25 vs MiniLM comparison
    # ========================================================

    bm25 = (
        metrics[
            metrics[
                "system"
            ]
            == "bm25"
        ][
            [
                "query_id",
                "ndcg_at_10",
                "mrr_at_10",
                "precision_at_10",
            ]
        ]
        .rename(
            columns={
                "ndcg_at_10":
                    "bm25_ndcg_at_10",

                "mrr_at_10":
                    "bm25_mrr_at_10",

                "precision_at_10":
                    "bm25_precision_at_10",
            }
        )
    )

    dense = (
        metrics[
            metrics[
                "system"
            ]
            == "minilm_dense"
        ][
            [
                "query_id",
                "ndcg_at_10",
                "mrr_at_10",
                "precision_at_10",
            ]
        ]
        .rename(
            columns={
                "ndcg_at_10":
                    "minilm_ndcg_at_10",

                "mrr_at_10":
                    "minilm_mrr_at_10",

                "precision_at_10":
                    "minilm_precision_at_10",
            }
        )
    )

    comparison = bm25.merge(
        dense,
        on="query_id",
        how="inner",
        validate="one_to_one",
    )

    comparison[
        "ndcg_delta_minilm_minus_bm25"
    ] = (
        comparison[
            "minilm_ndcg_at_10"
        ]
        - comparison[
            "bm25_ndcg_at_10"
        ]
    )

    comparison[
        "mrr_delta_minilm_minus_bm25"
    ] = (
        comparison[
            "minilm_mrr_at_10"
        ]
        - comparison[
            "bm25_mrr_at_10"
        ]
    )

    comparison[
        "precision_delta_minilm_minus_bm25"
    ] = (
        comparison[
            "minilm_precision_at_10"
        ]
        - comparison[
            "bm25_precision_at_10"
        ]
    )

    comparison.to_csv(
        COMPARISON_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Save Top-10 audit
    # ========================================================

    audit = pd.DataFrame(
        audit_rows
    )

    audit.to_csv(
        TOP10_AUDIT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Console results
    # ========================================================

    print()
    print("=" * 90)
    print("OVERALL BASELINE RESULTS")
    print("=" * 90)

    for _, row in summary.iterrows():

        print()
        print(
            row["system"]
        )

        print(
            "  nDCG@10:",
            f"{row['ndcg_at_10']:.4f}",
        )

        print(
            "  MRR@10:",
            f"{row['mrr_at_10']:.4f}",
        )

        print(
            "  Precision@10:",
            f"{row['precision_at_10']:.4f}",
        )

        print(
            "  HitRate@10:",
            f"{row['hit_rate_at_10']:.4f}",
        )

        print(
            "  Strict MRR@10 (rel=2):",
            f"{row['strict_mrr_at_10_rel2']:.4f}",
        )

        print(
            "  Strict Precision@10 (rel=2):",
            f"{row['strict_precision_at_10_rel2']:.4f}",
        )

    # ========================================================
    # Mean differences
    # ========================================================

    print()
    print("=" * 90)
    print("MINILM - BM25 MEAN DELTA")
    print("=" * 90)

    print()
    print(
        "nDCG@10 delta:",
        f"{comparison['ndcg_delta_minilm_minus_bm25'].mean():+.4f}",
    )

    print(
        "MRR@10 delta:",
        f"{comparison['mrr_delta_minilm_minus_bm25'].mean():+.4f}",
    )

    print(
        "Precision@10 delta:",
        f"{comparison['precision_delta_minilm_minus_bm25'].mean():+.4f}",
    )

    # ========================================================
    # Win / tie / loss by query
    # ========================================================

    print()
    print("=" * 90)
    print("MINILM VS BM25 QUERY WINS")
    print("=" * 90)

    for metric_name, delta_column in [
        (
            "nDCG@10",
            "ndcg_delta_minilm_minus_bm25",
        ),
        (
            "MRR@10",
            "mrr_delta_minilm_minus_bm25",
        ),
        (
            "Precision@10",
            "precision_delta_minilm_minus_bm25",
        ),
    ]:

        delta = (
            comparison[
                delta_column
            ]
        )

        epsilon = 1e-12

        wins = int(
            (
                delta
                > epsilon
            )
            .sum()
        )

        losses = int(
            (
                delta
                < -epsilon
            )
            .sum()
        )

        ties = (
            len(delta)
            - wins
            - losses
        )

        print()
        print(
            metric_name
        )

        print(
            f"  MiniLM wins: {wins}"
        )

        print(
            f"  Ties:        {ties}"
        )

        print(
            f"  BM25 wins:   {losses}"
        )

    # ========================================================
    # By query type
    # ========================================================

    print()
    print("=" * 90)
    print("RESULTS BY QUERY TYPE")
    print("=" * 90)

    display_columns = [
        "system",
        "query_type",
        "ndcg_at_10",
        "mrr_at_10",
        "precision_at_10",
    ]

    print()
    print(
        type_summary[
            display_columns
        ]
        .round(4)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # Outputs
    # ========================================================

    print()
    print("=" * 90)
    print("OUTPUT FILES")
    print("=" * 90)

    print()
    print(
        "Per-query metrics:"
    )
    print(
        PER_QUERY_PATH
    )

    print()
    print(
        "Overall summary:"
    )
    print(
        SUMMARY_PATH
    )

    print()
    print(
        "By query type:"
    )
    print(
        TYPE_SUMMARY_PATH
    )

    print()
    print(
        "System comparison:"
    )
    print(
        COMPARISON_PATH
    )

    print()
    print(
        "Top-10 audit:"
    )
    print(
        TOP10_AUDIT_PATH
    )

    print()
    print(
        "BASELINE EVALUATION: PASS"
    )


if __name__ == "__main__":
    main()