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

BASE_POOL_PATH = Path(
    r"evaluation\candidate_pool_v1"
    r"\candidate_pool_benchmark_v1.csv"
)

QWEN_PATH = Path(
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

OUTPUT_DIR = Path(
    r"evaluation\reranker_candidate_ablation_v1"
)

PER_QUERY_PATH = (
    OUTPUT_DIR
    / "reranker_candidate_ablation_per_query.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "reranker_candidate_ablation_summary.csv"
)

TYPE_SUMMARY_PATH = (
    OUTPUT_DIR
    / "reranker_candidate_ablation_by_query_type.csv"
)

TOP10_PATH = (
    OUTPUT_DIR
    / "reranker_candidate_ablation_top10.csv"
)

CANDIDATE_STATS_PATH = (
    OUTPUT_DIR
    / "reranker_candidate_set_stats.csv"
)


# ============================================================
# Candidate configurations
# ============================================================

CONFIGURATIONS = [
    "minilm",
    "qwen3",
    "bm25_minilm",
    "minilm_qwen3",
    "bm25_minilm_qwen3",
]


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
            for relevance
            in relevances[:k]
        )
        / k
    )


# ============================================================
# Candidate mask
# ============================================================

def configuration_mask(
    frame,
    configuration,
):

    if configuration == "minilm":

        return (
            frame["in_minilm"]
        )

    if configuration == "qwen3":

        return (
            frame["in_qwen3"]
        )

    if configuration == "bm25_minilm":

        return (
            frame["in_bm25"]
            |
            frame["in_minilm"]
        )

    if configuration == "minilm_qwen3":

        return (
            frame["in_minilm"]
            |
            frame["in_qwen3"]
        )

    if configuration == "bm25_minilm_qwen3":

        return (
            frame["in_bm25"]
            |
            frame["in_minilm"]
            |
            frame["in_qwen3"]
        )

    raise ValueError(
        f"Unknown configuration: {configuration}"
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
    print("BGE RERANKER — CANDIDATE SET ABLATION")
    print("=" * 90)

    # ========================================================
    # Load inputs
    # ========================================================

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

    # ========================================================
    # Numeric conversion
    # ========================================================

    base_pool["bm25_rank"] = pd.to_numeric(
        base_pool["bm25_rank"],
        errors="coerce",
    )

    base_pool["dense_rank"] = pd.to_numeric(
        base_pool["dense_rank"],
        errors="coerce",
    )

    qwen["qwen3_rank"] = pd.to_numeric(
        qwen["qwen3_rank"],
        errors="raise",
    )

    qrels["relevance"] = (
        qrels["relevance"]
        .astype(int)
    )

    # ========================================================
    # Qrels
    # ========================================================

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
    # Query metadata
    # ========================================================

    query_metadata = (
        queries
        .set_index("query_id")
        .to_dict(
            orient="index"
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
    # Results
    # ========================================================

    metric_rows = []
    top10_rows = []
    candidate_stat_rows = []

    # ========================================================
    # Process each query
    # ========================================================

    for position, query_row in enumerate(
        queries.itertuples(
            index=False
        ),
        start=1,
    ):

        query_id = str(
            query_row.query_id
        )

        query_text = str(
            query_row.query
        )

        query_type = str(
            query_row.query_type
        )

        # ====================================================
        # Base BM25 / MiniLM candidate membership
        # ====================================================

        base_query = (
            base_pool[
                base_pool["query_id"]
                == query_id
            ]
            .copy()
        )

        base_query["in_bm25"] = (
            base_query[
                "bm25_rank"
            ]
            .notna()
        )

        base_query["in_minilm"] = (
            base_query[
                "dense_rank"
            ]
            .notna()
        )

        base_membership = (
            base_query[
                [
                    "document_id",
                    "comment_text",
                    "in_bm25",
                    "in_minilm",
                ]
            ]
            .copy()
        )

        # ====================================================
        # Qwen candidate membership
        # ====================================================

        qwen_query = (
            qwen[
                qwen["query_id"]
                == query_id
            ]
            .copy()
        )

        if len(qwen_query) != 25:

            raise RuntimeError(
                f"{query_id}: expected 25 Qwen docs, "
                f"found {len(qwen_query)}."
            )

        qwen_membership = (
            qwen_query[
                [
                    "document_id",
                    "comment_text",
                ]
            ]
            .copy()
        )

        qwen_membership[
            "in_qwen3"
        ] = True

        # ====================================================
        # Build maximum union
        # ====================================================

        union = (
            base_membership
            .merge(
                qwen_membership,
                on="document_id",
                how="outer",
                suffixes=(
                    "_base",
                    "_qwen",
                ),
                validate="one_to_one",
            )
        )

        # ====================================================
        # Recover comment text
        # ====================================================

        union[
            "comment_text"
        ] = (
            union[
                "comment_text_base"
            ]
            .where(
                union[
                    "comment_text_base"
                ]
                .notna(),
                union[
                    "comment_text_qwen"
                ],
            )
        )

        union = union.drop(
            columns=[
                "comment_text_base",
                "comment_text_qwen",
            ]
        )

        # ====================================================
        # Missing membership means False
        # ====================================================

        for column in [
            "in_bm25",
            "in_minilm",
            "in_qwen3",
        ]:

            union[column] = (
                union[column]
                .fillna(False)
                .astype(bool)
            )

        # ====================================================
        # Important:
        #
        # base_pool also contains lexical/random-only docs.
        # They must NOT enter the max retriever union.
        # ====================================================

        union = (
            union[
                union[
                    "in_bm25"
                ]
                |
                union[
                    "in_minilm"
                ]
                |
                union[
                    "in_qwen3"
                ]
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        # ====================================================
        # Every maximum-union pair must be judged
        # ====================================================

        query_qrels = (
            qrels_lookup[
                query_id
            ]
        )

        unjudged = [
            document_id
            for document_id
            in union[
                "document_id"
            ]
            if document_id
            not in query_qrels
        ]

        if unjudged:

            raise RuntimeError(
                f"{query_id}: "
                f"{len(unjudged)} union candidates unjudged."
            )

        print(
            f"[{position:02d}/60] "
            f"{query_id} "
            f"max_union={len(union)}"
        )

        # ====================================================
        # One reranker inference over max union
        # ====================================================

        pairs = [
            (
                query_text,
                str(comment),
            )
            for comment
            in union[
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

        if len(scores) != len(union):

            raise RuntimeError(
                f"{query_id}: "
                "reranker score count mismatch."
            )

        union[
            "reranker_score"
        ] = scores

        # ====================================================
        # Evaluate every candidate configuration
        # ====================================================

        all_judged_relevances = list(
            query_qrels.values()
        )

        for configuration in CONFIGURATIONS:

            mask = configuration_mask(
                union,
                configuration,
            )

            candidates = (
                union[
                    mask
                ]
                .copy()
            )

            if len(candidates) < TOP_K:

                raise RuntimeError(
                    f"{query_id}/{configuration}: "
                    f"only {len(candidates)} candidates."
                )

            # -----------------------------------------------
            # Candidate statistics
            # -----------------------------------------------

            candidate_relevances = [
                int(
                    query_qrels[
                        document_id
                    ]
                )
                for document_id
                in candidates[
                    "document_id"
                ]
            ]

            candidate_stat_rows.append({
                "query_id":
                    query_id,

                "query_type":
                    query_type,

                "configuration":
                    configuration,

                "candidate_count":
                    len(candidates),

                "candidate_rel0":
                    candidate_relevances.count(
                        0
                    ),

                "candidate_rel1":
                    candidate_relevances.count(
                        1
                    ),

                "candidate_rel2":
                    candidate_relevances.count(
                        2
                    ),

                "candidate_relevant_ge_1":
                    sum(
                        relevance >= 1
                        for relevance
                        in candidate_relevances
                    ),
            })

            # -----------------------------------------------
            # Cross-encoder ranking
            # -----------------------------------------------

            ranked = (
                candidates
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
                    TOP_K
                )
                .copy()
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

            # -----------------------------------------------
            # Metrics
            # -----------------------------------------------

            ndcg = ndcg_at_k(
                relevances,
                all_judged_relevances,
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
                "configuration":
                    configuration,

                "query_id":
                    query_id,

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

            # -----------------------------------------------
            # Save Top10
            # -----------------------------------------------

            for rank, row in enumerate(
                ranked.itertuples(
                    index=False
                ),
                start=1,
            ):

                top10_rows.append({
                    "configuration":
                        configuration,

                    "query_id":
                        query_id,

                    "query_type":
                        query_type,

                    "query":
                        query_text,

                    "rank":
                        rank,

                    "document_id":
                        row.document_id,

                    "reranker_score":
                        row.reranker_score,

                    "relevance":
                        query_qrels[
                            row.document_id
                        ],

                    "comment_text":
                        row.comment_text,
                })

    # ========================================================
    # DataFrames
    # ========================================================

    metrics = pd.DataFrame(
        metric_rows
    )

    top10 = pd.DataFrame(
        top10_rows
    )

    candidate_stats = pd.DataFrame(
        candidate_stat_rows
    )

    metric_columns = [
        "ndcg_at_10",
        "mrr_at_10",
        "precision_at_10",
        "strict_mrr_at_10_rel2",
        "strict_precision_at_10_rel2",
    ]

    # ========================================================
    # Overall
    # ========================================================

    summary = (
        metrics
        .groupby(
            "configuration",
            as_index=False,
        )[
            metric_columns
        ]
        .mean()
    )

    order = {
        name:
            index
        for index, name
        in enumerate(
            CONFIGURATIONS
        )
    }

    summary[
        "_order"
    ] = (
        summary[
            "configuration"
        ]
        .map(order)
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
                "configuration",
                "query_type",
            ],
            as_index=False,
        )[
            metric_columns
        ]
        .mean()
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

    top10.to_csv(
        TOP10_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    candidate_stats.to_csv(
        CANDIDATE_STATS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Console
    # ========================================================

    print()
    print("=" * 90)
    print("CANDIDATE SET ABLATION RESULTS")
    print("=" * 90)

    for _, row in summary.iterrows():

        print()
        print(
            row[
                "configuration"
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
            "  Strict MRR@10:",
            f"{row['strict_mrr_at_10_rel2']:.4f}",
        )

        print(
            "  Strict Precision@10:",
            f"{row['strict_precision_at_10_rel2']:.4f}",
        )

    # ========================================================
    # Candidate coverage
    # ========================================================

    print()
    print("=" * 90)
    print("AVERAGE CANDIDATE SET STATISTICS")
    print("=" * 90)

    candidate_summary = (
        candidate_stats
        .groupby(
            "configuration"
        )
        .agg(
            avg_candidates=(
                "candidate_count",
                "mean",
            ),
            avg_relevant=(
                "candidate_relevant_ge_1",
                "mean",
            ),
            avg_rel2=(
                "candidate_rel2",
                "mean",
            ),
        )
        .reindex(
            CONFIGURATIONS
        )
        .round(2)
    )

    print()
    print(
        candidate_summary
        .to_string()
    )

    # ========================================================
    # Best system
    # ========================================================

    best_index = (
        summary[
            "ndcg_at_10"
        ]
        .idxmax()
    )

    best = (
        summary.loc[
            best_index
        ]
    )

    print()
    print("=" * 90)
    print("BEST CONFIGURATION BY nDCG@10")
    print("=" * 90)

    print()
    print(
        "Configuration:",
        best[
            "configuration"
        ],
    )

    print(
        "nDCG@10:",
        f"{best['ndcg_at_10']:.4f}",
    )

    print(
        "MRR@10:",
        f"{best['mrr_at_10']:.4f}",
    )

    print(
        "Precision@10:",
        f"{best['precision_at_10']:.4f}",
    )

    # ========================================================
    # Query type
    # ========================================================

    print()
    print("=" * 90)
    print("RESULTS BY QUERY TYPE")
    print("=" * 90)

    display = (
        type_summary[
            [
                "configuration",
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
            "configuration"
        ]
        .map(order)
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

    print()
    print("=" * 90)
    print("CANDIDATE ABLATION: PASS")
    print("=" * 90)


if __name__ == "__main__":
    main()