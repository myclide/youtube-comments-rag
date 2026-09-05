from pathlib import Path

import pandas as pd


BY_VIDEO_DIR = (
    Path("data")
    / "structured_comments"
    / "by_video"
)


def load_corpus():
    frames = []

    for file in sorted(
        BY_VIDEO_DIR.glob("*.jsonl")
    ):
        if file.stat().st_size == 0:
            continue

        df = pd.read_json(
            file,
            lines=True,
        )

        if not df.empty:
            frames.append(df)

    if not frames:
        raise RuntimeError(
            "No comment data found."
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


def normalize_text(series):
    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )
    )


def main():
    df = load_corpus()

    df["normalized_text"] = (
        normalize_text(
            df["comment_text"]
        )
    )

    print("=" * 70)
    print("TEXT DUPLICATE AUDIT")
    print("=" * 70)

    print(
        f"Total comments: "
        f"{len(df):,}"
    )

    print()

    # ========================================================
    # Global duplicate text
    # ========================================================

    global_duplicate_mask = (
        df["normalized_text"]
        .duplicated(
            keep="first"
        )
    )

    global_duplicate_rows = (
        global_duplicate_mask.sum()
    )

    print("GLOBAL EXACT-TEXT DUPLICATES")
    print("-" * 70)

    print(
        f"Duplicate rows: "
        f"{global_duplicate_rows:,}"
    )

    print(
        "Duplicate percentage: "
        f"{global_duplicate_rows / len(df) * 100:.2f}%"
    )

    print()

    # ========================================================
    # Within-video duplicates
    # ========================================================

    within_duplicate_mask = (
        df.duplicated(
            subset=[
                "video_id",
                "normalized_text",
            ],
            keep="first",
        )
    )

    within_duplicate_rows = (
        within_duplicate_mask.sum()
    )

    print("WITHIN-VIDEO EXACT DUPLICATES")
    print("-" * 70)

    print(
        f"Duplicate rows: "
        f"{within_duplicate_rows:,}"
    )

    print(
        "Duplicate percentage: "
        f"{within_duplicate_rows / len(df) * 100:.2f}%"
    )

    print()

    # ========================================================
    # Cross-video duplicated texts
    # ========================================================

    text_video_counts = (
        df.groupby(
            "normalized_text"
        )["video_id"]
        .nunique()
    )

    cross_video_texts = (
        text_video_counts[
            text_video_counts > 1
        ]
    )

    cross_video_rows = (
        df[
            df["normalized_text"].isin(
                cross_video_texts.index
            )
        ]
    )

    print("CROSS-VIDEO REPEATED TEXT")
    print("-" * 70)

    print(
        f"Unique texts appearing "
        f"in >1 video: "
        f"{len(cross_video_texts):,}"
    )

    print(
        f"Rows belonging to these texts: "
        f"{len(cross_video_rows):,}"
    )

    print()

    # ========================================================
    # Most repeated texts globally
    # ========================================================

    counts = (
        df["normalized_text"]
        .value_counts()
    )

    print("TOP 30 MOST REPEATED TEXTS")
    print("-" * 70)

    for text, count in (
        counts.head(30).items()
    ):
        video_count = (
            df.loc[
                df["normalized_text"]
                == text,
                "video_id",
            ]
            .nunique()
        )

        print()
        print(
            f"COUNT={count} | "
            f"VIDEOS={video_count}"
        )

        print(
            repr(
                text[:300]
            )
        )

    # ========================================================
    # Videos with most internal duplicate rows
    # ========================================================

    duplicate_df = (
        df[
            within_duplicate_mask
        ]
    )

    print()
    print("=" * 70)
    print(
        "VIDEOS WITH MOST "
        "WITHIN-VIDEO DUPLICATES"
    )
    print("-" * 70)

    if duplicate_df.empty:
        print(
            "No within-video duplicates."
        )

    else:
        per_video = (
            duplicate_df
            .groupby(
                [
                    "video_id",
                    "video_title",
                ]
            )
            .size()
            .sort_values(
                ascending=False
            )
        )

        print(
            per_video
            .head(20)
            .to_string()
        )


if __name__ == "__main__":
    main()