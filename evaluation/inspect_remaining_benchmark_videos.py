import sys
import pandas as pd
from pathlib import Path


sys.stdout.reconfigure(encoding="utf-8")


CORPUS_PATH = Path(
    r"data\corpus_v1\retrieval_comments_v1.parquet"
)

BENCHMARK_VIDEOS_PATH = Path(
    r"evaluation\benchmark_v1_videos.csv"
)

ALREADY_DONE_VIDEO_ID = "dxPhJ0wfp0g"

TOP_LIKED_N = 8
RANDOM_N = 8
RANDOM_SEED = 42


def main():

    corpus = pd.read_parquet(
        CORPUS_PATH
    )

    benchmark_videos = pd.read_csv(
        BENCHMARK_VIDEOS_PATH
    )

    benchmark_videos = benchmark_videos[
        benchmark_videos["video_id"]
        != ALREADY_DONE_VIDEO_ID
    ].copy()

    print("=" * 100)
    print("REMAINING BENCHMARK VIDEO INSPECTION")
    print("=" * 100)

    print(
        f"Remaining videos: "
        f"{len(benchmark_videos)}"
    )

    for position, video_row in enumerate(
        benchmark_videos.itertuples(
            index=False
        ),
        start=1,
    ):

        video_id = video_row.video_id

        video_df = (
            corpus[
                corpus["video_id"]
                == video_id
            ]
            .copy()
            .reset_index(drop=True)
        )

        print()
        print()
        print("#" * 100)

        print(
            f"VIDEO {position}/"
            f"{len(benchmark_videos)}"
        )

        print("#" * 100)

        print(
            f"video_id: {video_id}"
        )

        print(
            f"size_group: "
            f"{video_row.size_group}"
        )

        print(
            f"category: "
            f"{video_row.category}"
        )

        if video_df.empty:

            print(
                "ERROR: no documents found."
            )

            continue

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

        # ====================================================
        # Top liked
        # ====================================================

        print()
        print("-" * 100)
        print(
            f"TOP {TOP_LIKED_N} BY LIKE COUNT"
        )
        print("-" * 100)

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

        # ====================================================
        # Deterministic random sample
        # ====================================================

        print()
        print("-" * 100)
        print(
            f"RANDOM SAMPLE OF "
            f"{RANDOM_N}"
        )
        print("-" * 100)

        sample_size = min(
            RANDOM_N,
            len(video_df),
        )

        random_sample = (
            video_df
            .sample(
                n=sample_size,
                random_state=(
                    RANDOM_SEED
                    + position
                ),
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