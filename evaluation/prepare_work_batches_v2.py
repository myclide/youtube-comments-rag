from pathlib import Path

import pandas as pd


UNLABELED_PATH = Path(
    r"evaluation\candidate_pool_v1"
    r"\candidate_pool_benchmark_v1_unlabeled.csv"
)

CORPUS_PATH = Path(
    r"data\corpus_v1"
    r"\retrieval_comments_v1.parquet"
)

OUTPUT_DIR = Path(
    r"evaluation\candidate_pool_v1"
    r"\work_batches_v2"
)

QUERIES_PER_BATCH = 10


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Load unlabeled frozen candidate pool
    # ========================================================

    df = pd.read_csv(
        UNLABELED_PATH,
        keep_default_na=False,
        dtype=str,
    )

    # ========================================================
    # Load video titles from retrieval corpus
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

    video_titles = (
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

    # ========================================================
    # Merge video title into judging data
    # ========================================================

    df = df.merge(
        video_titles,
        on="video_id",
        how="left",
        validate="many_to_one",
    )

    if df["video_title"].isna().any():

        missing_videos = (
            df.loc[
                df[
                    "video_title"
                ].isna(),
                "video_id",
            ]
            .unique()
        )

        raise RuntimeError(
            "Missing video titles for: "
            f"{missing_videos}"
        )

    # --------------------------------------------------------
    # Put video_title beside video_id
    # --------------------------------------------------------

    ordered_columns = [
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

    df = df[
        ordered_columns
    ]

    # ========================================================
    # Split by query
    # ========================================================

    query_ids = (
        df[
            "query_id"
        ]
        .drop_duplicates()
        .tolist()
    )

    print("=" * 80)
    print("PREPARING WORK BATCHES V2")
    print("=" * 80)

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Queries: {len(query_ids)}"
    )

    total_written = 0
    batch_number = 0

    for start in range(
        0,
        len(query_ids),
        QUERIES_PER_BATCH,
    ):

        batch_number += 1

        batch_queries = (
            query_ids[
                start:
                start + QUERIES_PER_BATCH
            ]
        )

        batch_df = (
            df[
                df[
                    "query_id"
                ].isin(
                    batch_queries
                )
            ]
            .copy()
        )

        first_query = (
            batch_queries[0]
        )

        last_query = (
            batch_queries[-1]
        )

        output_path = (
            OUTPUT_DIR
            / (
                f"benchmark_v1_batch_"
                f"{batch_number:02d}_"
                f"{first_query}_to_"
                f"{last_query}_v2.csv"
            )
        )

        batch_df.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
        )

        total_written += len(
            batch_df
        )

        print()
        print(
            f"Batch {batch_number:02d}"
        )

        print(
            f"Queries: "
            f"{first_query} -> "
            f"{last_query}"
        )

        print(
            f"Rows: {len(batch_df)}"
        )

        print(
            f"Saved: {output_path}"
        )

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(
        f"Batches: {batch_number}"
    )

    print(
        f"Rows written: "
        f"{total_written:,}"
    )

    if total_written != len(df):

        raise RuntimeError(
            "Row-count mismatch."
        )

    print(
        "Integrity check: PASS"
    )


if __name__ == "__main__":
    main()