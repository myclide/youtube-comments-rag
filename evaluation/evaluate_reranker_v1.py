from pathlib import Path

import numpy as np
import pandas as pd

from sentence_transformers import CrossEncoder


# ============================================================
# Configuration
# ============================================================

TOP_K = 10

MODEL_NAME = "BAAI/bge-reranker-v2-m3"

BATCH_SIZE = 16


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

BASELINE_PATH = Path(
    r"evaluation\baseline_results_v1"
    r"\baseline_metrics_per_query.csv"
)

OUTPUT_DIR = Path(
    r"evaluation\reranker_results_v1"
)

PER_QUERY_PATH = (
    OUTPUT_DIR
    / "reranker_metrics_per_query.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "reranker_metrics_summary.csv"
)

TOP10_PATH = (
    OUTPUT_DIR
    / "reranker_top10_results.csv"
)

COMPARISON_PATH = (
    OUTPUT_DIR
    / "reranker_vs_minilm.csv"
)

TYPE_SUMMARY_PATH = (
    OUTPUT_DIR
    / "reranker_metrics_by_query_type.csv"
)


# ============================================================
# Metrics
# ============================================================

def dcg_at_k(relevances, k):

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

    ideal = dcg_at_k(
        sorted(
            all_judged_relevances,
            reverse=True,
        )[:k],
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

    return (
        sum(
            relevance >= threshold
            for relevance in relevances[:k]
        )
        / k
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
    print("BENCHMARK V1 — CROSS-ENCODER RERANKING")
    print("=" * 90)

    # ========================================================
    # Load frozen candidate pool
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

    pool["bm25_rank"] = pd.to_numeric(
        pool["bm25_rank"],
        errors="coerce",
    )

    pool["dense_rank"] = pd.to_numeric(
        pool["dense_rank"],
        errors="coerce",
    )

    # ========================================================
    # Load qrels
    # ========================================================

    qrels = pd.read_csv(
        QRELS_PATH,
        dtype={
            "query_id": str,
            "document_id": str,
        },
    )

    qrels["relevance"] = (
        qrels["relevance"]
        .astype(int)
    )

    qrels_lookup = {}

    for query_id, group in qrels.groupby(
        "query_id",
        sort=False,
    ):

        qrels_lookup[query_id] = dict(
            zip(
                group["document_id"],
                group["relevance"],
            )
        )

    # ========================================================
    # Load reranker
    # ========================================================

    print()
    print("Loading reranker:")
    print(MODEL_NAME)
    print()

    model = CrossEncoder(
        MODEL_NAME,
        max_length=512,
    )

    # ========================================================
    # Evaluate each query
    # ========================================================

    metric_rows = []
    top10_rows = []

    query_ids = (
        pool["query_id"]
        .drop_duplicates()
        .tolist()
    )

    for index, query_id in enumerate(
        query_ids,
        start=1,
    ):

        query_pool = (
            pool[
                pool["query_id"]
                == query_id
            ]
            .copy()
        )

        query_text = (
            query_pool["query"]
            .iloc[0]
        )

        query_type = (
            query_pool["query_type"]
            .iloc[0]
        )

        video_id = (
            query_pool["video_id"]
            .iloc[0]
        )

        # ====================================================
        # Candidate generation:
        #
        # union(
        #   BM25 Top25,
        #   MiniLM Top25
        # )
        #
        # Exclude lexical/random-only candidates.
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
            .reset_index(
                drop=True
            )
        )

        print(
            f"[{index:02d}/60] "
            f"{query_id} "
            f"candidates={len(candidates)}"
        )

        # ====================================================
        # Cross-encoder pairs
        # ====================================================

        pairs = [
            (
                query_text,
                str(comment),
            )
            for comment in candidates[
                "comment_text"
            ]
        ]

        scores = model.predict(
            pairs,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
        )

        scores = (
            np.asarray(scores)
            .reshape(-1)
        )

        if len(scores) != len(candidates):

            raise RuntimeError(
                f"{query_id}: score count mismatch."
            )

        candidates[
            "reranker_score"
        ] = scores

        # ====================================================
        # Deterministic tie-breaking
        # ====================================================

        candidates[
            "best_original_rank"
        ] = (
            candidates[
                [
                    "dense_rank",
                    "bm25_rank",
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
                    "reranker_score",
                    "best_original_rank",
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

        # ====================================================
        # Verify all Top10 are judged
        # ====================================================

        query_qrels = (
            qrels_lookup[
                query_id
            ]
        )

        unjudged = [
            document_id
            for document_id in ranked[
                "document_id"
            ]
            if document_id
            not in query_qrels
        ]

        if unjudged:

            raise RuntimeError(
                f"{query_id}: "
                f"{len(unjudged)} unjudged documents."
            )

        relevances = [
            int(
                query_qrels[
                    document_id
                ]
            )
            for document_id in ranked[
                "document_id"
            ]
        ]

        all_relevances = list(
            query_qrels.values()
        )

        # ====================================================
        # Metrics
        # ====================================================

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

        metric_rows.append({
            "system":
                "bge_reranker_v2_m3",

            "query_id":
                query_id,

            "video_id":
                video_id,

            "query_type":
                query_type,

            "query":
                query_text,

            "candidate_count":
                len(candidates),

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
        # Save Top10
        # ====================================================

        for rank, (_, row) in enumerate(
            ranked.iterrows(),
            start=1,
        ):

            document_id = (
                row["document_id"]
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

                "reranker_score":
                    row[
                        "reranker_score"
                    ],

                "dense_rank":
                    row[
                        "dense_rank"
                    ],

                "bm25_rank":
                    row[
                        "bm25_rank"
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
    # Save results
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

    metric_columns = [
        "ndcg_at_10",
        "mrr_at_10",
        "precision_at_10",
        "strict_mrr_at_10_rel2",
        "strict_precision_at_10_rel2",
    ]

    # ========================================================
    # Overall summary
    # ========================================================

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
        "bge_reranker_v2_m3",
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # By query type
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
    # Compare to MiniLM baseline
    # ========================================================

    baseline = pd.read_csv(
        BASELINE_PATH
    )

    minilm = (
        baseline[
            baseline["system"]
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

    reranker = (
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
                    "reranker_ndcg",

                "mrr_at_10":
                    "reranker_mrr",

                "precision_at_10":
                    "reranker_precision",
            }
        )
    )

    comparison = (
        minilm
        .merge(
            reranker,
            on="query_id",
            validate="one_to_one",
        )
    )

    comparison[
        "ndcg_delta"
    ] = (
        comparison[
            "reranker_ndcg"
        ]
        - comparison[
            "minilm_ndcg"
        ]
    )

    comparison[
        "mrr_delta"
    ] = (
        comparison[
            "reranker_mrr"
        ]
        - comparison[
            "minilm_mrr"
        ]
    )

    comparison[
        "precision_delta"
    ] = (
        comparison[
            "reranker_precision"
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
    # Console summary
    # ========================================================

    row = summary.iloc[0]

    print()
    print("=" * 90)
    print("RERANKER RESULTS")
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
    # Mean delta
    # ========================================================

    print()
    print("=" * 90)
    print("RERANKER - MINILM MEAN DELTA")
    print("=" * 90)

    print()

    print(
        "nDCG@10:",
        f"{comparison['ndcg_delta'].mean():+.4f}",
    )

    print(
        "MRR@10:",
        f"{comparison['mrr_delta'].mean():+.4f}",
    )

    print(
        "Precision@10:",
        f"{comparison['precision_delta'].mean():+.4f}",
    )

    # ========================================================
    # Wins
    # ========================================================

    print()
    print("=" * 90)
    print("RERANKER VS MINILM QUERY WINS")
    print("=" * 90)

    epsilon = 1e-12

    for name, column in [
        (
            "nDCG@10",
            "ndcg_delta",
        ),
        (
            "MRR@10",
            "mrr_delta",
        ),
        (
            "Precision@10",
            "precision_delta",
        ),
    ]:

        delta = (
            comparison[column]
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

        print()
        print(name)

        print(
            "  reranker wins:",
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
    # Query type
    # ========================================================

    print()
    print("=" * 90)
    print("RERANKER BY QUERY TYPE")
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
    print("RERANKER EVALUATION: PASS")
    print("=" * 90)


if __name__ == "__main__":
    main()