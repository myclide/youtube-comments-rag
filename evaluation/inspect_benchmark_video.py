import pandas as pd


CORPUS_PATH = (
    r"data\corpus_v1\retrieval_comments_v1.parquet"
)

VIDEO_ID = "dxPhJ0wfp0g"


def main():
    df = pd.read_parquet(
        CORPUS_PATH
    )

    video_df = (
        df[
            df["video_id"] == VIDEO_ID
        ]
        .copy()
        .reset_index(drop=True)
    )

    if video_df.empty:
        raise RuntimeError(
            f"No documents found for video {VIDEO_ID}"
        )

    print("=" * 80)
    print("BENCHMARK VIDEO INSPECTION")
    print("=" * 80)

    print(
        "Video title:",
        video_df["video_title"].iloc[0],
    )

    print(
        "Retrieval documents:",
        len(video_df),
    )

    print()

    # ========================================================
    # Most liked comments
    # ========================================================

    print("=" * 80)
    print("TOP 25 BY LIKE COUNT")
    print("=" * 80)

    top_liked = (
        video_df
        .sort_values(
            "max_like_count",
            ascending=False,
        )
        .head(25)
    )

    for i, row in enumerate(
        top_liked.itertuples(),
        start=1,
    ):
        print()
        print(
            f"[{i}] "
            f"likes={row.max_like_count} | "
            f"occurrences={row.occurrence_count}"
        )

        print(
            repr(row.comment_text)
        )

    # ========================================================
    # Deterministic random sample
    # ========================================================

    print()
    print("=" * 80)
    print("RANDOM SAMPLE OF 30 DOCUMENTS")
    print("=" * 80)

    sample_size = min(
        30,
        len(video_df),
    )

    random_sample = (
        video_df.sample(
            n=sample_size,
            random_state=42,
        )
    )

    for i, row in enumerate(
        random_sample.itertuples(),
        start=1,
    ):
        print()
        print(
            f"[{i}] "
            f"likes={row.max_like_count} | "
            f"occurrences={row.occurrence_count}"
        )

        print(
            repr(row.comment_text)
        )


if __name__ == "__main__":
    main()