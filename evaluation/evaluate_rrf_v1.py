from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

TOP_K = 10

# Standard, fixed RRF constant.
#
# Do NOT tune this on Benchmark V1.
RRF_K = 60

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

BASELINE_PER_QUERY_PATH = Path(
    r"evaluation\baseline_results_v1"
    r"\baseline_metrics_per_query.csv"
)

OUTPUT_DIR = Path(
    r"evaluation\rrf_results_v1"
)

PER_QUERY_PATH = (
    OUTPUT_DIR
    / "rrf_metrics_per_query.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "rrf_metrics_summary.csv"
)

TOP10_PATH = (
    OUTPUT_DIR
    / "rrf_top10_results.csv"
)

COMPARISON_PATH = (
    OUTPUT_DIR
    / "rrf_vs_baselines.csv"
)

TYPE_SUMMARY_PATH = (
    OUTPUT_DIR
    / "rrf_metrics_by_query_type.csv"
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

    relevant = sum(
        relevance >= threshold
        for relevance
        in relevances[:k]
    )

    return relevant / k


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 90)
    print("BENCHMARK V1 — BM25 + MINILM RRF")
    print("=" * 90)

    # ========================================================
    # Load data
    # ========================================================

    pool = pd.read_csv(
        POOL_PATH,
        dtype={
            "query_id": str,
            "video_id": str,
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

    qrels[
        "relevance"
    ] = (
        qrels[
            "relevance"
        ]
        .astype(int)
    )

    # Convert ranks.
    pool[
        "bm25_rank"
    ] = pd.to_numeric(
        pool[
            "bm25_rank"
        ],
        errors="coerce",
    )

    pool[
        "dense_rank"
    ] = pd.to_numeric(
        pool[
            "dense_rank"
        ],
        errors="coerce",
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
    # Evaluate
    # ========================================================

    metric_rows = []
    top10_rows = []

    query_ids = (
        pool[
            "query_id"
        ]
        .drop_duplicates()
        .tolist()
    )

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

        # ====================================================
        # IMPORTANT
        #
        # RRF candidate set only includes documents retrieved
        # by BM25 or MiniLM.
        #
        # lexical/random-only pool documents are excluded.
        # ====================================================

        candidates = (
            query_pool[
                query_pool[
                    "bm25_rank"
                ].notna()
                |
                query_pool[
                    "dense_rank"
                ].notna()
            ]
            .copy()
        )

        # ====================================================
        # RRF
        #
        # score =
        #   1 / (k + bm25_rank)
        # + 1 / (k + dense_rank)
        #
        # Missing retrieval contributes 0.
        # ====================================================

        candidates[
            "bm25_rrf"
        ] = np.where(
            candidates[
                "bm25_rank"
            ].notna(),
            1.0
            / (
                RRF_K
                + candidates[
                    "bm25_rank"
                ].fillna(0)
            ),
            0.0,
        )

        candidates[
            "dense_rrf"
        ] = np.where(
            candidates[
                "dense_rank"
            ].notna(),
            1.0
            / (
                RRF_K
                + candidates[
                    "dense_rank"
                ].fillna(0)
            ),
            0.0,
        )

        candidates[
            "rrf_score"
        ] = (
            candidates[
                "bm25_rrf"
            ]
            + candidates[
                "dense_rrf"
            ]
        )

        # Best individual rank for deterministic tie-breaking.
        candidates[
            "best_component_rank"
        ] = (
            candidates[
                [
                    "bm25_rank",
                    "dense_rank",
                ]
            ]
            .min(
                axis=1,
                skipna=True,
            )
        )

        ranked = (
            candidates
            .sort_values(
                by=[
                    "rrf_score",
                    "best_component_rank",
                    "document_id",
                ],
                ascending=[
                    False,
                    True,
                    True,
                ],
            )
            .head(
                TOP_K
            )
            .copy()
        )

        if len(ranked) != TOP_K:

            raise RuntimeError(
                f"{query_id}: only "
                f"{len(ranked)} RRF candidates."
            )

        # ====================================================
        # Every RRF result must already be judged
        # ====================================================

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
                f"{query_id}: "
                f"{len(unjudged)} unjudged RRF docs."
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

        all_relevances = list(
            query_qrels.values()
        )

        ndcg = ndcg_at_k(
            relevances,
            all_relevances,
            TOP_K,
        )

        mrr = reciprocal_rank_at_k(
            relevances,
            TOP_K,
            threshold=1,
        )

        precision = precision_at_k(
            relevances,
            TOP_K,
            threshold=1,
        )

        strict_mrr = reciprocal_rank_at_k(
            relevances,
            TOP_K,
            threshold=2,
        )

        strict_precision = precision_at_k(
            relevances,
            TOP_K,
            threshold=2,
        )

        query_type = (
            query_pool[
                "query_type"
            ].iloc[0]
        )

        query_text = (
            query_pool[
                "query"
            ].iloc[0]
        )

        video_id = (
            query_pool[
                "video_id"
            ].iloc[0]
        )

        metric_rows.append({
            "system":
                "bm25_minilm_rrf",

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

            "strict_mrr_at_10_rel2":
                strict_mrr,

            "strict_precision_at_10_rel2":
                strict_precision,
        })

        # ====================================================
        # Save ranking
        # ====================================================

        for rank, (_, row) in enumerate(
            ranked.iterrows(),
            start=1,
        ):

            document_id = (
                row[
                    "document_id"
                ]
            )

            top10_rows.append({
                "query_id":
                    query_id,

                "query_type":
                    query_type,

                "query":
                    query_text,

                "rank":
                    rank,

                "document_id":
                    document_id,

                "rrf_score":
                    row[
                        "rrf_score"
                    ],

                "bm25_rank":
                    row[
                        "bm25_rank"
                    ],

                "dense_rank":
                    row[
                        "dense_rank"
                    ],

                "relevance":
                    query_qrels[
                        document_id
                    ],

                "comment_text":
                    row[
                        "comment_text"
                    ],
            })

    # ========================================================
    # Results
    # ========================================================

    metrics = pd.DataFrame(
        metric_rows
    )

    metrics.to_csv(
        PER_QUERY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    top10 = pd.DataFrame(
        top10_rows
    )

    top10.to_csv(
        TOP10_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Overall summary
    # ========================================================

    metric_columns = [
        "ndcg_at_10",
        "mrr_at_10",
        "precision_at_10",
        "strict_mrr_at_10_rel2",
        "strict_precision_at_10_rel2",
    ]

    summary = (
        metrics[
            metric_columns
        ]
        .mean()
        .to_frame()
        .T
    )

    summary.insert(
        0,
        "system",
        "bm25_minilm_rrf",
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
            "query_type",
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
    # Compare against frozen baselines
    # ========================================================

    baselines = pd.read_csv(
        BASELINE_PER_QUERY_PATH
    )

    baseline_subset = (
        baselines[
            [
                "system",
                "query_id",
                "ndcg_at_10",
                "mrr_at_10",
                "precision_at_10",
            ]
        ]
    )

    bm25 = (
        baseline_subset[
            baseline_subset[
                "system"
            ]
            == "bm25"
        ]
        .drop(
            columns=[
                "system"
            ]
        )
        .rename(
            columns={
                "ndcg_at_10":
                    "bm25_ndcg",

                "mrr_at_10":
                    "bm25_mrr",

                "precision_at_10":
                    "bm25_precision",
            }
        )
    )

    minilm = (
        baseline_subset[
            baseline_subset[
                "system"
            ]
            == "minilm_dense"
        ]
        .drop(
            columns=[
                "system"
            ]
        )
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

    rrf = (
        metrics[
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
                    "rrf_ndcg",

                "mrr_at_10":
                    "rrf_mrr",

                "precision_at_10":
                    "rrf_precision",
            }
        )
    )

    comparison = (
        bm25
        .merge(
            minilm,
            on="query_id",
            validate="one_to_one",
        )
        .merge(
            rrf,
            on="query_id",
            validate="one_to_one",
        )
    )

    comparison[
        "rrf_minus_minilm_ndcg"
    ] = (
        comparison[
            "rrf_ndcg"
        ]
        - comparison[
            "minilm_ndcg"
        ]
    )

    comparison[
        "rrf_minus_minilm_mrr"
    ] = (
        comparison[
            "rrf_mrr"
        ]
        - comparison[
            "minilm_mrr"
        ]
    )

    comparison[
        "rrf_minus_minilm_precision"
    ] = (
        comparison[
            "rrf_precision"
        ]
        - comparison[
            "minilm_precision"
        ]
    )

    comparison.to_csv(
        COMPARISON_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Console
    # ========================================================

    row = summary.iloc[0]

    print()
    print("=" * 90)
    print("RRF RESULTS")
    print("=" * 90)

    print()
    print(
        "nDCG@10:",
        f"{row['ndcg_at_10']:.4f}",
    )

    print(
        "MRR@10:",
        f"{row['mrr_at_10']:.4f}",
    )

    print(
        "Precision@10:",
        f"{row['precision_at_10']:.4f}",
    )

    print(
        "Strict MRR@10:",
        f"{row['strict_mrr_at_10_rel2']:.4f}",
    )

    print(
        "Strict Precision@10:",
        f"{row['strict_precision_at_10_rel2']:.4f}",
    )

    # ========================================================
    # RRF vs MiniLM
    # ========================================================

    print()
    print("=" * 90)
    print("RRF - MINILM MEAN DELTA")
    print("=" * 90)

    for name, column in [
        (
            "nDCG@10",
            "rrf_minus_minilm_ndcg",
        ),
        (
            "MRR@10",
            "rrf_minus_minilm_mrr",
        ),
        (
            "Precision@10",
            "rrf_minus_minilm_precision",
        ),
    ]:

        print(
            f"{name}: "
            f"{comparison[column].mean():+.4f}"
        )

    # ========================================================
    # Win / tie / loss
    # ========================================================

    print()
    print("=" * 90)
    print("RRF VS MINILM QUERY WINS")
    print("=" * 90)

    epsilon = 1e-12

    for metric, column in [
        (
            "nDCG@10",
            "rrf_minus_minilm_ndcg",
        ),
        (
            "MRR@10",
            "rrf_minus_minilm_mrr",
        ),
        (
            "Precision@10",
            "rrf_minus_minilm_precision",
        ),
    ]:

        delta = comparison[
            column
        ]

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

        print()
        print(metric)

        print(
            "  RRF wins:",
            wins,
        )

        print(
            "  ties:",
            ties,
        )

        print(
            "  MiniLM wins:",
            losses,
        )

    # ========================================================
    # Query types
    # ========================================================

    print()
    print("=" * 90)
    print("RRF BY QUERY TYPE")
    print("=" * 90)

    print()

    print(
        type_summary[
            [
                "query_type",
                "ndcg_at_10",
                "mrr_at_10",
                "precision_at_10",
            ]
        ]
        .round(4)
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 90)
    print("OUTPUT FILES")
    print("=" * 90)

    print()
    print(PER_QUERY_PATH)
    print(SUMMARY_PATH)
    print(TYPE_SUMMARY_PATH)
    print(COMPARISON_PATH)
    print(TOP10_PATH)

    print()
    print(
        "RRF EVALUATION: PASS"
    )


if __name__ == "__main__":
    main()