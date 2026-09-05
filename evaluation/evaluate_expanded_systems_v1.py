from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

K = 10

BINARY_RELEVANCE_THRESHOLD = 1


# ============================================================
# Paths
# ============================================================

QRELS_PATH = Path(
    r"evaluation\expanded_qrels_v1"
    r"\benchmark_v1_qrels_expanded_qwen3.csv"
)

QUERY_PATH = Path(
    r"evaluation\benchmark_v1_queries.csv"
)

BASE_POOL_PATH = Path(
    r"evaluation\candidate_pool_v1"
    r"\candidate_pool_benchmark_v1.csv"
)

QWEN_PATH = Path(
    r"evaluation\qwen3_dense_v1"
    r"\qwen3_dense_top25.csv"
)

RERANKER_PATH = Path(
    r"evaluation\reranker_results_v1"
    r"\reranker_top10_results.csv"
)

OUTPUT_DIR = Path(
    r"evaluation\expanded_results_v1"
)

PER_QUERY_PATH = (
    OUTPUT_DIR
    / "expanded_metrics_per_query.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "expanded_metrics_summary.csv"
)

TYPE_SUMMARY_PATH = (
    OUTPUT_DIR
    / "expanded_metrics_by_query_type.csv"
)

COMPARISON_PATH = (
    OUTPUT_DIR
    / "expanded_system_comparison.csv"
)

TOP10_AUDIT_PATH = (
    OUTPUT_DIR
    / "expanded_top10_audit.csv"
)


# ============================================================
# Metrics
# ============================================================

def dcg_at_k(
    relevances,
    k,
):

    relevances = np.asarray(
        relevances[:k],
        dtype=float,
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

    relevant_count = sum(
        relevance >= threshold
        for relevance in top_k
    )

    return relevant_count / k


def hit_rate_at_k(
    relevances,
    k,
    threshold=1,
):

    return float(
        any(
            relevance >= threshold
            for relevance in relevances[:k]
        )
    )


# ============================================================
# Ranking loaders
# ============================================================

def build_rankings_from_base_pool(
    pool,
    rank_column,
    system_name,
):

    rankings = {}

    for query_id, group in pool.groupby(
        "query_id",
        sort=False,
    ):

        ranked = (
            group[
                group[
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

        if len(ranked) != K:

            raise RuntimeError(
                f"{system_name}/{query_id}: "
                f"expected {K} documents, "
                f"found {len(ranked)}."
            )

        actual_ranks = (
            ranked[
                rank_column
            ]
            .astype(int)
            .tolist()
        )

        if actual_ranks != list(
            range(
                1,
                K + 1,
            )
        ):

            raise RuntimeError(
                f"{system_name}/{query_id}: "
                f"invalid ranks {actual_ranks}"
            )

        rankings[
            query_id
        ] = (
            ranked[
                [
                    "document_id",
                    "comment_text",
                ]
            ]
            .reset_index(
                drop=True
            )
        )

    return rankings


def build_qwen_rankings(
    qwen,
):

    rankings = {}

    for query_id, group in qwen.groupby(
        "query_id",
        sort=False,
    ):

        ranked = (
            group
            .sort_values(
                "qwen3_rank",
                ascending=True,
            )
            .head(K)
            .copy()
        )

        if len(ranked) != K:

            raise RuntimeError(
                f"qwen3/{query_id}: "
                f"expected {K} documents."
            )

        actual_ranks = (
            ranked[
                "qwen3_rank"
            ]
            .astype(int)
            .tolist()
        )

        if actual_ranks != list(
            range(
                1,
                K + 1,
            )
        ):

            raise RuntimeError(
                f"qwen3/{query_id}: "
                f"invalid ranks {actual_ranks}"
            )

        rankings[
            query_id
        ] = (
            ranked[
                [
                    "document_id",
                    "comment_text",
                ]
            ]
            .reset_index(
                drop=True
            )
        )

    return rankings


def build_reranker_rankings(
    reranker,
):

    rankings = {}

    for query_id, group in reranker.groupby(
        "query_id",
        sort=False,
    ):

        ranked = (
            group
            .sort_values(
                "rank",
                ascending=True,
            )
            .head(K)
            .copy()
        )

        if len(ranked) != K:

            raise RuntimeError(
                f"reranker/{query_id}: "
                f"expected {K} documents."
            )

        actual_ranks = (
            ranked[
                "rank"
            ]
            .astype(int)
            .tolist()
        )

        if actual_ranks != list(
            range(
                1,
                K + 1,
            )
        ):

            raise RuntimeError(
                f"reranker/{query_id}: "
                f"invalid ranks {actual_ranks}"
            )

        rankings[
            query_id
        ] = (
            ranked[
                [
                    "document_id",
                    "comment_text",
                ]
            ]
            .reset_index(
                drop=True
            )
        )

    return rankings


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 90)
    print("EXPANDED POOLED-QRELS SYSTEM EVALUATION")
    print("=" * 90)

    # ========================================================
    # Load inputs
    # ========================================================

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

    base_pool = pd.read_csv(
        BASE_POOL_PATH,
        dtype={
            "query_id": str,
            "document_id": str,
        },
        keep_default_na=False,
    )

    qwen = pd.read_csv(
        QWEN_PATH,
        dtype={
            "query_id": str,
            "document_id": str,
        },
        keep_default_na=False,
    )

    reranker = pd.read_csv(
        RERANKER_PATH,
        dtype={
            "query_id": str,
            "document_id": str,
        },
        keep_default_na=False,
    )

    # ========================================================
    # Normalize numeric fields
    # ========================================================

    qrels[
        "relevance"
    ] = (
        qrels[
            "relevance"
        ]
        .astype(int)
    )

    for column in [
        "bm25_rank",
        "dense_rank",
    ]:

        base_pool[
            column
        ] = pd.to_numeric(
            base_pool[
                column
            ],
            errors="coerce",
        )

    qwen[
        "qwen3_rank"
    ] = pd.to_numeric(
        qwen[
            "qwen3_rank"
        ],
        errors="raise",
    )

    reranker[
        "rank"
    ] = pd.to_numeric(
        reranker[
            "rank"
        ],
        errors="raise",
    )

    # ========================================================
    # Input integrity
    # ========================================================

    print()
    print("INPUT INTEGRITY")
    print("-" * 90)

    print(
        "Expanded qrels:",
        f"{len(qrels):,}",
    )

    print(
        "Queries:",
        qrels[
            "query_id"
        ].nunique(),
    )

    print(
        "Base candidate rows:",
        f"{len(base_pool):,}",
    )

    print(
        "Qwen Top25 rows:",
        f"{len(qwen):,}",
    )

    print(
        "Reranker Top10 rows:",
        f"{len(reranker):,}",
    )

    qrel_duplicates = int(
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
        "Duplicate qrels:",
        qrel_duplicates,
    )

    if qrel_duplicates != 0:

        raise RuntimeError(
            "Duplicate expanded qrels."
        )

    if len(qrels) != 3758:

        raise RuntimeError(
            "Expected 3,758 expanded qrels."
        )

    if (
        qrels[
            "query_id"
        ]
        .nunique()
        != 60
    ):

        raise RuntimeError(
            "Expected 60 queries."
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
    # Build frozen rankings
    # ========================================================

    print()
    print("BUILDING FROZEN TOP-10 RANKINGS")
    print("-" * 90)

    bm25_rankings = (
        build_rankings_from_base_pool(
            base_pool,
            "bm25_rank",
            "bm25",
        )
    )

    minilm_rankings = (
        build_rankings_from_base_pool(
            base_pool,
            "dense_rank",
            "minilm_dense",
        )
    )

    qwen_rankings = (
        build_qwen_rankings(
            qwen
        )
    )

    reranker_rankings = (
        build_reranker_rankings(
            reranker
        )
    )

    systems = {
        "bm25":
            bm25_rankings,

        "minilm_dense":
            minilm_rankings,

        "qwen3_embedding_0.6b":
            qwen_rankings,

        "bge_reranker_v2_m3":
            reranker_rankings,
    }

    for system_name, rankings in systems.items():

        print(
            f"{system_name}: "
            f"{len(rankings)} queries"
        )

        if len(rankings) != 60:

            raise RuntimeError(
                f"{system_name} does not contain 60 queries."
            )

    # ========================================================
    # Evaluate
    # ========================================================

    metric_rows = []
    audit_rows = []

    query_ids = (
        queries[
            "query_id"
        ]
        .tolist()
    )

    print()
    print("TOP-10 JUDGMENT COVERAGE")
    print("-" * 90)

    for system_name, rankings in systems.items():

        system_unjudged = 0

        for query_id in query_ids:

            ranked = (
                rankings[
                    query_id
                ]
            )

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

            system_unjudged += len(
                unjudged
            )

            if unjudged:

                raise RuntimeError(
                    f"{system_name}/{query_id}: "
                    f"unjudged Top10 documents: "
                    f"{unjudged}"
                )

        print(
            f"{system_name}: "
            f"unjudged Top10 = "
            f"{system_unjudged}"
        )

    # ========================================================
    # Metrics
    # ========================================================

    for system_name, rankings in systems.items():

        for query_id in query_ids:

            metadata = (
                query_metadata[
                    query_id
                ]
            )

            ranked = (
                rankings[
                    query_id
                ]
            )

            query_qrels = (
                qrels_lookup[
                    query_id
                ]
            )

            relevances = [
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

            ndcg = ndcg_at_k(
                relevances,
                all_judged_relevances,
                K,
            )

            mrr = reciprocal_rank_at_k(
                relevances,
                K,
                threshold=1,
            )

            precision = precision_at_k(
                relevances,
                K,
                threshold=1,
            )

            hit_rate = hit_rate_at_k(
                relevances,
                K,
                threshold=1,
            )

            strict_mrr = (
                reciprocal_rank_at_k(
                    relevances,
                    K,
                    threshold=2,
                )
            )

            strict_precision = (
                precision_at_k(
                    relevances,
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
                    metadata[
                        "video_id"
                    ],

                "query_type":
                    metadata[
                        "query_type"
                    ],

                "query":
                    metadata[
                        "query"
                    ],

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

            for rank, row in enumerate(
                ranked.itertuples(
                    index=False
                ),
                start=1,
            ):

                audit_rows.append({
                    "system":
                        system_name,

                    "query_id":
                        query_id,

                    "query_type":
                        metadata[
                            "query_type"
                        ],

                    "query":
                        metadata[
                            "query"
                        ],

                    "rank":
                        rank,

                    "document_id":
                        row.document_id,

                    "relevance":
                        query_qrels[
                            row.document_id
                        ],

                    "comment_text":
                        row.comment_text,
                })

    metrics = pd.DataFrame(
        metric_rows
    )

    audit = pd.DataFrame(
        audit_rows
    )

    # ========================================================
    # Overall macro averages
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

    # Preferred display order.
    system_order = {
        "bm25":
            0,

        "minilm_dense":
            1,

        "qwen3_embedding_0.6b":
            2,

        "bge_reranker_v2_m3":
            3,
    }

    summary[
        "_order"
    ] = (
        summary[
            "system"
        ]
        .map(
            system_order
        )
    )

    summary = (
        summary
        .sort_values(
            "_order"
        )
        .drop(
            columns=[
                "_order"
            ]
        )
        .reset_index(
            drop=True
        )
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

    # ========================================================
    # Direct comparison to MiniLM baseline
    # ========================================================

    minilm = (
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
                    "minilm_ndcg",

                "mrr_at_10":
                    "minilm_mrr",

                "precision_at_10":
                    "minilm_precision",
            }
        )
    )

    comparison_frames = []

    for system_name in [
        "bm25",
        "qwen3_embedding_0.6b",
        "bge_reranker_v2_m3",
    ]:

        system_metrics = (
            metrics[
                metrics[
                    "system"
                ]
                == system_name
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
                        "system_ndcg",

                    "mrr_at_10":
                        "system_mrr",

                    "precision_at_10":
                        "system_precision",
                }
            )
        )

        comparison = (
            minilm
            .merge(
                system_metrics,
                on="query_id",
                validate="one_to_one",
            )
        )

        comparison[
            "system"
        ] = system_name

        comparison[
            "ndcg_delta_vs_minilm"
        ] = (
            comparison[
                "system_ndcg"
            ]
            - comparison[
                "minilm_ndcg"
            ]
        )

        comparison[
            "mrr_delta_vs_minilm"
        ] = (
            comparison[
                "system_mrr"
            ]
            - comparison[
                "minilm_mrr"
            ]
        )

        comparison[
            "precision_delta_vs_minilm"
        ] = (
            comparison[
                "system_precision"
            ]
            - comparison[
                "minilm_precision"
            ]
        )

        comparison_frames.append(
            comparison
        )

    comparisons = pd.concat(
        comparison_frames,
        ignore_index=True,
    )

    # ========================================================
    # Save
    # ========================================================

    metrics.to_csv(
        PER_QUERY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    type_summary.to_csv(
        TYPE_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    comparisons.to_csv(
        COMPARISON_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    audit.to_csv(
        TOP10_AUDIT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Console
    # ========================================================

    print()
    print("=" * 90)
    print("EXPANDED POOLED-QRELS RESULTS")
    print("=" * 90)

    for _, row in summary.iterrows():

        print()
        print(
            row[
                "system"
            ]
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
            "  Strict MRR@10:",
            f"{row['strict_mrr_at_10_rel2']:.4f}",
        )

        print(
            "  Strict Precision@10:",
            f"{row['strict_precision_at_10_rel2']:.4f}",
        )

    # ========================================================
    # Relative improvements
    # ========================================================

    minilm_summary = (
        summary[
            summary[
                "system"
            ]
            == "minilm_dense"
        ]
        .iloc[0]
    )

    print()
    print("=" * 90)
    print("RELATIVE IMPROVEMENT VS MINILM")
    print("=" * 90)

    for system_name in [
        "qwen3_embedding_0.6b",
        "bge_reranker_v2_m3",
    ]:

        row = (
            summary[
                summary[
                    "system"
                ]
                == system_name
            ]
            .iloc[0]
        )

        ndcg_relative = (
            (
                row[
                    "ndcg_at_10"
                ]
                - minilm_summary[
                    "ndcg_at_10"
                ]
            )
            / minilm_summary[
                "ndcg_at_10"
            ]
            * 100
        )

        mrr_relative = (
            (
                row[
                    "mrr_at_10"
                ]
                - minilm_summary[
                    "mrr_at_10"
                ]
            )
            / minilm_summary[
                "mrr_at_10"
            ]
            * 100
        )

        precision_relative = (
            (
                row[
                    "precision_at_10"
                ]
                - minilm_summary[
                    "precision_at_10"
                ]
            )
            / minilm_summary[
                "precision_at_10"
            ]
            * 100
        )

        print()
        print(
            system_name
        )

        print(
            "  nDCG@10 relative:",
            f"{ndcg_relative:+.2f}%",
        )

        print(
            "  MRR@10 relative:",
            f"{mrr_relative:+.2f}%",
        )

        print(
            "  Precision@10 relative:",
            f"{precision_relative:+.2f}%",
        )

    # ========================================================
    # Win / tie / loss vs MiniLM
    # ========================================================

    print()
    print("=" * 90)
    print("QUERY-LEVEL WINS VS MINILM")
    print("=" * 90)

    epsilon = 1e-12

    for system_name in [
        "qwen3_embedding_0.6b",
        "bge_reranker_v2_m3",
    ]:

        system_comparison = (
            comparisons[
                comparisons[
                    "system"
                ]
                == system_name
            ]
        )

        print()
        print(
            system_name
        )

        for metric_name, column in [
            (
                "nDCG@10",
                "ndcg_delta_vs_minilm",
            ),
            (
                "MRR@10",
                "mrr_delta_vs_minilm",
            ),
            (
                "Precision@10",
                "precision_delta_vs_minilm",
            ),
        ]:

            delta = (
                system_comparison[
                    column
                ]
            )

            wins = int(
                (
                    delta > epsilon
                )
                .sum()
            )

            losses = int(
                (
                    delta < -epsilon
                )
                .sum()
            )

            ties = (
                len(delta)
                - wins
                - losses
            )

            print(
                f"  {metric_name}: "
                f"wins={wins}, "
                f"ties={ties}, "
                f"losses={losses}"
            )

    # ========================================================
    # Results by query type
    # ========================================================

    print()
    print("=" * 90)
    print("RESULTS BY QUERY TYPE")
    print("=" * 90)

    display = (
        type_summary[
            [
                "system",
                "query_type",
                "ndcg_at_10",
                "mrr_at_10",
                "precision_at_10",
            ]
        ]
        .copy()
    )

    display[
        "_order"
    ] = (
        display[
            "system"
        ]
        .map(
            system_order
        )
    )

    display = (
        display
        .sort_values(
            [
                "_order",
                "query_type",
            ]
        )
        .drop(
            columns=[
                "_order"
            ]
        )
    )

    print()
    print(
        display
        .round(4)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # Output files
    # ========================================================

    print()
    print("=" * 90)
    print("OUTPUT FILES")
    print("=" * 90)

    print()
    print(
        PER_QUERY_PATH
    )

    print(
        SUMMARY_PATH
    )

    print(
        TYPE_SUMMARY_PATH
    )

    print(
        COMPARISON_PATH
    )

    print(
        TOP10_AUDIT_PATH
    )

    print()
    print(
        "EXPANDED EVALUATION: PASS"
    )


if __name__ == "__main__":
    main()