from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


# ============================================================
# Project root
# ============================================================

ROOT_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT_DIR),
    )


from app.modern_retrieval import ModernRAGPipeline


# ============================================================
# Configuration
# ============================================================

QUERY_ID = "q001"

CANDIDATE_DEPTH = 25

FINAL_TOP_K = 10


# ============================================================
# Paths
# ============================================================

QUERY_PATH = (
    ROOT_DIR
    / "evaluation"
    / "benchmark_v1_queries.csv"
)

EXPECTED_DENSE_PATH = (
    ROOT_DIR
    / "evaluation"
    / "qwen3_dense_v1"
    / "qwen3_dense_top25.csv"
)

EXPECTED_RERANK_PATH = (
    ROOT_DIR
    / "evaluation"
    / "reranker_candidate_ablation_v1"
    / "reranker_candidate_ablation_top10.csv"
)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 90)
    print("MODERN SERVING PIPELINE VALIDATION")
    print("=" * 90)

    # ========================================================
    # Load benchmark query
    # ========================================================

    queries = pd.read_csv(
        QUERY_PATH,
        dtype=str,
        keep_default_na=False,
    )

    query_rows = (
        queries[
            queries["query_id"]
            == QUERY_ID
        ]
    )

    if len(query_rows) != 1:

        raise RuntimeError(
            f"Expected exactly one row for {QUERY_ID}."
        )

    query_row = (
        query_rows.iloc[0]
    )

    query_text = str(
        query_row["query"]
    )

    video_id = str(
        query_row["video_id"]
    )

    print()
    print("QUERY")
    print("-" * 90)

    print(
        "Query ID:",
        QUERY_ID,
    )

    print(
        "Video ID:",
        video_id,
    )

    print(
        "Query:",
        query_text,
    )

    # ========================================================
    # Expected frozen Qwen Top25
    # ========================================================

    expected_dense = pd.read_csv(
        EXPECTED_DENSE_PATH,
        dtype={
            "query_id": str,
            "document_id": str,
        },
        keep_default_na=False,
    )

    expected_dense["qwen3_rank"] = pd.to_numeric(
        expected_dense["qwen3_rank"],
        errors="raise",
    ).astype(int)

    expected_dense = (
        expected_dense[
            expected_dense["query_id"]
            == QUERY_ID
        ]
        .sort_values(
            "qwen3_rank"
        )
        .head(
            CANDIDATE_DEPTH
        )
        .reset_index(
            drop=True
        )
    )

    if len(expected_dense) != CANDIDATE_DEPTH:

        raise RuntimeError(
            "Frozen Qwen result does not contain 25 rows."
        )

    expected_dense_ids = (
        expected_dense[
            "document_id"
        ]
        .tolist()
    )

    # ========================================================
    # Expected frozen BGE Top10
    # ========================================================

    expected_rerank = pd.read_csv(
        EXPECTED_RERANK_PATH,
        dtype={
            "query_id": str,
            "document_id": str,
        },
        keep_default_na=False,
    )

    expected_rerank["rank"] = pd.to_numeric(
        expected_rerank["rank"],
        errors="raise",
    ).astype(int)

    expected_rerank = (
        expected_rerank[
            (
                expected_rerank[
                    "configuration"
                ]
                == "qwen3"
            )
            &
            (
                expected_rerank[
                    "query_id"
                ]
                == QUERY_ID
            )
        ]
        .sort_values(
            "rank"
        )
        .head(
            FINAL_TOP_K
        )
        .reset_index(
            drop=True
        )
    )

    if len(expected_rerank) != FINAL_TOP_K:

        raise RuntimeError(
            "Frozen Qwen+BGE result does not contain 10 rows."
        )

    expected_rerank_ids = (
        expected_rerank[
            "document_id"
        ]
        .tolist()
    )

    # ========================================================
    # Load serving pipeline
    # ========================================================

    print()
    print("=" * 90)
    print("LOADING SERVING PIPELINE")
    print("=" * 90)

    pipeline = ModernRAGPipeline(
        device="auto"
    )

    print()
    print(
        "Serving metadata rows:",
        f"{len(pipeline.comments_df):,}",
    )

    print(
        "Serving embedding shape:",
        pipeline.document_embeddings.shape,
    )

    # ========================================================
    # Dense retrieval reproduction
    # ========================================================

    print()
    print("=" * 90)
    print("DENSE TOP25 REPRODUCTION")
    print("=" * 90)

    query_vector = (
        pipeline._encode_query(
            query_text
        )
    )

    serving_candidates = (
        pipeline._dense_candidates(
            query_vector=query_vector,
            candidate_depth=CANDIDATE_DEPTH,
            video_id_filter=video_id,
        )
    )

    serving_dense_ids = (
        serving_candidates[
            "document_id"
        ]
        .astype(str)
        .tolist()
    )

    dense_set_match = (
        set(
            serving_dense_ids
        )
        == set(
            expected_dense_ids
        )
    )

    dense_order_match = (
        serving_dense_ids
        == expected_dense_ids
    )

    overlap = len(
        set(
            serving_dense_ids
        )
        &
        set(
            expected_dense_ids
        )
    )

    print(
        "Expected Top25:",
        len(
            expected_dense_ids
        ),
    )

    print(
        "Serving Top25:",
        len(
            serving_dense_ids
        ),
    )

    print(
        "Top25 overlap:",
        f"{overlap}/25",
    )

    print(
        "Top25 candidate set match:",
        dense_set_match,
    )

    print(
        "Top25 exact ranking match:",
        dense_order_match,
    )

    if not dense_set_match:

        expected_only = [
            document_id
            for document_id
            in expected_dense_ids
            if document_id
            not in serving_dense_ids
        ]

        serving_only = [
            document_id
            for document_id
            in serving_dense_ids
            if document_id
            not in expected_dense_ids
        ]

        print()
        print(
            "Expected-only documents:"
        )

        for document_id in expected_only:
            print(
                " ",
                document_id,
            )

        print()
        print(
            "Serving-only documents:"
        )

        for document_id in serving_only:
            print(
                " ",
                document_id,
            )

        raise RuntimeError(
            "Serving dense candidate set does not reproduce "
            "the frozen Qwen Top25."
        )

    print()
    print(
        "DENSE CANDIDATE REPRODUCTION: PASS"
    )

    # ========================================================
    # BGE reranking reproduction
    # ========================================================

    print()
    print("=" * 90)
    print("BGE TOP10 REPRODUCTION")
    print("=" * 90)

    serving_reranked = (
        pipeline._rerank(
            query=query_text,
            candidates=serving_candidates,
        )
        .head(
            FINAL_TOP_K
        )
        .reset_index(
            drop=True
        )
    )

    serving_rerank_ids = (
        serving_reranked[
            "document_id"
        ]
        .astype(str)
        .tolist()
    )

    rerank_set_match = (
        set(
            serving_rerank_ids
        )
        == set(
            expected_rerank_ids
        )
    )

    rerank_order_match = (
        serving_rerank_ids
        == expected_rerank_ids
    )

    rerank_overlap = len(
        set(
            serving_rerank_ids
        )
        &
        set(
            expected_rerank_ids
        )
    )

    print(
        "Top10 overlap:",
        f"{rerank_overlap}/10",
    )

    print(
        "Top10 result set match:",
        rerank_set_match,
    )

    print(
        "Top10 exact ranking match:",
        rerank_order_match,
    )

    print()
    print("SERVING TOP10")
    print("-" * 90)

    for rank, row in enumerate(
        serving_reranked.itertuples(
            index=False
        ),
        start=1,
    ):

        print(
            f"{rank:02d}. "
            f"{row.document_id} | "
            f"dense={row.dense_score:.4f} | "
            f"reranker={row.reranker_score:.4f}"
        )

    if not rerank_set_match:

        raise RuntimeError(
            "Serving BGE Top10 set does not reproduce "
            "the frozen benchmark result."
        )

    if not rerank_order_match:

        raise RuntimeError(
            "Serving BGE Top10 ranking order does not reproduce "
            "the frozen benchmark result."
        )

    print()
    print(
        "BGE RERANKING REPRODUCTION: PASS"
    )

    # ========================================================
    # Public API smoke test
    # ========================================================

    print()
    print("=" * 90)
    print("PUBLIC RETRIEVE() API")
    print("=" * 90)

    public_results = pipeline.retrieve(
        query=query_text,
        top_k=FINAL_TOP_K,
        video_id_filter=video_id,
        candidate_depth=CANDIDATE_DEPTH,
    )

    public_ids = [
        str(
            result.extra[
                "document_id"
            ]
        )
        for result
        in public_results
    ]

    public_match = (
        public_ids
        == expected_rerank_ids
    )

    print(
        "Returned results:",
        len(
            public_results
        ),
    )

    print(
        "Public API exact Top10 match:",
        public_match,
    )

    if not public_match:

        raise RuntimeError(
            "Public retrieve() output does not reproduce "
            "the benchmark Top10."
        )

    print()
    print("=" * 90)
    print("MODERN SERVING VALIDATION: PASS")
    print("=" * 90)


if __name__ == "__main__":
    main()