from pathlib import Path

import pandas as pd


# ============================================================
# Paths
# ============================================================

JUDGMENTS_PATH = Path(
    r"evaluation\candidate_pool_v1"
    r"\benchmark_v1_judgments_gpt_v2.csv"
)

CORPUS_PATH = Path(
    r"data\corpus_v1"
    r"\retrieval_comments_v1.parquet"
)

OUTPUT_PATH = Path(
    r"evaluation\candidate_pool_v1"
    r"\benchmark_v1_adjudication_blind.csv"
)

MANIFEST_PATH = Path(
    r"evaluation\candidate_pool_v1"
    r"\benchmark_v1_adjudication_manifest.csv"
)


# ============================================================
# Queries requiring full-query review
# ============================================================

FULL_REVIEW_QUERIES = {
    # No highly-relevant documents / suspicious coverage
    "q011",
    "q012",
    "q030",

    # Sparse qrels
    "q018",
    "q039",
    "q050",
    "q054",

    # Same semantic issue:
    # query explicitly asks what viewers ask
    "q036",
    "q059",
}


def main():

    print("=" * 90)
    print("PREPARING BENCHMARK V1 TARGETED ADJUDICATION")
    print("=" * 90)

    # ========================================================
    # Load judgments
    # ========================================================

    judgments = pd.read_csv(
        JUDGMENTS_PATH,
        dtype=str,
        keep_default_na=False,
    )

    # ========================================================
    # Add video titles
    # ========================================================

    corpus = pd.read_parquet(
        CORPUS_PATH,
        columns=[
            "video_id",
            "video_title",
        ],
    )

    corpus["video_id"] = (
        corpus["video_id"]
        .astype(str)
    )

    titles = (
        corpus[
            [
                "video_id",
                "video_title",
            ]
        ]
        .drop_duplicates(
            subset=[
                "video_id"
            ]
        )
    )

    judgments = judgments.merge(
        titles,
        on="video_id",
        how="left",
        validate="many_to_one",
    )

    if judgments["video_title"].isna().any():

        raise RuntimeError(
            "Missing video_title."
        )

    # ========================================================
    # Determine why each row needs adjudication
    # ========================================================

    uncertain_mask = (
        judgments[
            "gpt_uncertain"
        ]
        .str.upper()
        == "TRUE"
    )

    query_review_mask = (
        judgments[
            "query_id"
        ]
        .isin(
            FULL_REVIEW_QUERIES
        )
    )

    selected_mask = (
        uncertain_mask
        | query_review_mask
    )

    review = (
        judgments[
            selected_mask
        ]
        .copy()
    )

    # ========================================================
    # Internal manifest
    #
    # This retains the OLD GPT judgment.
    # Do NOT give this file to Work.
    # ========================================================

    reasons = []

    for row in review.itertuples(
        index=False
    ):

        row_reasons = []

        if (
            str(
                row.gpt_uncertain
            ).upper()
            == "TRUE"
        ):
            row_reasons.append(
                "uncertain"
            )

        if (
            row.query_id
            in FULL_REVIEW_QUERIES
        ):
            row_reasons.append(
                "full_query_review"
            )

        reasons.append(
            "|".join(
                row_reasons
            )
        )

    review[
        "review_reason"
    ] = reasons

    manifest_columns = [
        "label_id",
        "query_id",
        "video_id",
        "video_title",
        "query_type",
        "query",
        "document_id",
        "representative_comment_id",
        "comment_text",
        "gpt_relevance",
        "gpt_reason",
        "gpt_uncertain",
        "review_reason",
    ]

    review[
        manifest_columns
    ].to_csv(
        MANIFEST_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Blind adjudication version
    #
    # IMPORTANT:
    # Remove old judgment completely.
    # ========================================================

    blind_columns = [
        "label_id",
        "query_id",
        "video_id",
        "video_title",
        "query_type",
        "query",
        "document_id",
        "representative_comment_id",
        "comment_text",
    ]

    blind = (
        review[
            blind_columns
        ]
        .copy()
    )

    # Shuffle deterministically so Work does not simply see
    # blocks ordered by previous labeling behavior.
    blind = (
        blind.sample(
            frac=1,
            random_state=20260903,
        )
        .reset_index(
            drop=True
        )
    )

    blind.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Statistics
    # ========================================================

    uncertain_rows = int(
        uncertain_mask.sum()
    )

    full_review_rows = int(
        query_review_mask.sum()
    )

    overlap_rows = int(
        (
            uncertain_mask
            & query_review_mask
        )
        .sum()
    )

    print()
    print("Original judgments:")
    print(
        f"  {len(judgments):,}"
    )

    print()
    print("Uncertain rows:")
    print(
        f"  {uncertain_rows:,}"
    )

    print()
    print("Rows from full-review queries:")
    print(
        f"  {full_review_rows:,}"
    )

    print()
    print("Overlap:")
    print(
        f"  {overlap_rows:,}"
    )

    print()
    print("Unique adjudication rows:")
    print(
        f"  {len(blind):,}"
    )

    print()
    print("Queries represented:")
    print(
        blind[
            "query_id"
        ].nunique()
    )

    print()
    print("Full-review queries:")

    for query_id in sorted(
        FULL_REVIEW_QUERIES
    ):
        count = int(
            (
                judgments[
                    "query_id"
                ]
                == query_id
            )
            .sum()
        )

        print(
            f"  {query_id}: {count}"
        )

    print()
    print("Blind Work file:")
    print(
        OUTPUT_PATH
    )

    print()
    print("Internal manifest:")
    print(
        MANIFEST_PATH
    )

    print()
    print(
        "PREPARATION COMPLETE"
    )


if __name__ == "__main__":
    main()