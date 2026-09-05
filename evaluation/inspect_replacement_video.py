import pandas as pd
from pathlib import Path


CORPUS_PATH = Path(
    r"data\corpus_v1\retrieval_comments_v1.parquet"
)

VIDEO_ID = "a-VCL7U_JEA"

TOP_LIKED_N = 20
RANDOM_N = 20
RANDOM_SEED = 42


def main():

    df = pd.read_parquet(
        CORPUS_PATH
    )

    video_df = (
        df[
            df["video_id"]
            .astype(str)
            == VIDEO_ID
        ]
        .copy()
        .reset_index(drop=True)
    )

    if video_df.empty:
        raise RuntimeError(
            f"No documents found for video {VIDEO_ID}"
        )

    print("=" * 100)
    print("REPLACEMENT BENCHMARK VIDEO INSPECTION")
    print("=" * 100)

    print(
        "video_id:",
        VIDEO_ID,
    )

    print(
        "title:",
        video_df[
            "video_title"
        ].iloc[0],
    )

    print(
        "documents:",
        len(video_df),
    )

    # ========================================================
    # Top liked comments
    # ========================================================

    print()
    print("=" * 100)
    print(
        f"TOP {TOP_LIKED_N} BY LIKE COUNT"
    )
    print("=" * 100)

    if (
        "max_like_count"
        in video_df.columns
    ):

        top_liked = (
            video_df
            .sort_values(
                "max_like_count",
                ascending=False,
            )
            .head(
                TOP_LIKED_N
            )
        )

    else:

        top_liked = (
            video_df
            .head(
                TOP_LIKED_N
            )
        )

    for i, row in enumerate(
        top_liked.itertuples(),
        start=1,
    ):

        print()
        print(
            f"[TOP {i}]"
        )

        if hasattr(
            row,
            "max_like_count",
        ):

            print(
                "likes:",
                row.max_like_count,
            )

        print(
            repr(
                row.comment_text
            )
        )

    # ========================================================
    # Deterministic random sample
    # ========================================================

    print()
    print("=" * 100)
    print(
        f"RANDOM SAMPLE OF {RANDOM_N}"
    )
    print("=" * 100)

    random_sample = (
        video_df
        .sample(
            n=min(
                RANDOM_N,
                len(video_df),
            ),
            random_state=RANDOM_SEED,
        )
    )

    for i, row in enumerate(
        random_sample.itertuples(),
        start=1,
    ):

        print()
        print(
            f"[RANDOM {i}]"
        )

        if hasattr(
            row,
            "max_like_count",
        ):

            print(
                "likes:",
                row.max_like_count,
            )

        print(
            repr(
                row.comment_text
            )
        )


if __name__ == "__main__":
    main()