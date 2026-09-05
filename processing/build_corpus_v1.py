import json
import re
from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

SOURCE_DIR = (
    Path("data")
    / "structured_comments"
    / "by_video"
)

OUTPUT_DIR = (
    Path("data")
    / "corpus_v1"
)

RAW_OUTPUT = (
    OUTPUT_DIR
    / "raw_comments_v1.parquet"
)

RETRIEVAL_OUTPUT = (
    OUTPUT_DIR
    / "retrieval_comments_v1.parquet"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "corpus_v1_summary.json"
)


# ============================================================
# Normalization
# ============================================================

def normalize_text(text):
    """
    Conservative normalization used only
    for within-video exact-text grouping.

    We intentionally DO NOT:
    - remove punctuation
    - remove emojis
    - stem
    - lemmatize
    - remove stop words

    This keeps deduplication conservative.
    """

    text = str(text)

    text = text.strip()

    text = text.lower()

    # Collapse:
    # spaces / tabs / internal newlines
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


# ============================================================
# Load raw corpus
# ============================================================

def load_raw_comments():
    frames = []

    files = sorted(
        SOURCE_DIR.glob(
            "*.jsonl"
        )
    )

    for file in files:

        # Legitimate zero-comment videos
        if file.stat().st_size == 0:
            continue

        df = pd.read_json(
            file,
            lines=True,
        )

        if df.empty:
            continue

        frames.append(df)

    if not frames:
        raise RuntimeError(
            "No comment records found."
        )

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    return df


# ============================================================
# Build retrieval corpus
# ============================================================

def build_retrieval_corpus(raw_df):
    df = raw_df.copy()

    df["normalized_text"] = (
        df["comment_text"]
        .map(normalize_text)
    )

    # --------------------------------------------
    # Each group represents:
    #
    # SAME VIDEO
    # +
    # SAME normalized text
    # --------------------------------------------

    grouped = (
        df.groupby(
            [
                "video_id",
                "normalized_text",
            ],
            sort=False,
            dropna=False,
        )
    )

    records = []

    for (
        video_id,
        normalized_text,
    ), group in grouped:

        # Representative comment:
        # keep the comment with the
        # highest like count.
        representative = (
            group.sort_values(
                "like_count",
                ascending=False,
            )
            .iloc[0]
        )

        published = pd.to_datetime(
            group["published_at"],
            errors="coerce",
            utc=True,
        )

        authors = (
            group["author"]
            .dropna()
            .astype(str)
        )

        record = {
            # --------------------------------
            # Core document identity
            # --------------------------------

            "video_id": (
                video_id
            ),

            "video_title": (
                representative[
                    "video_title"
                ]
            ),

            "source_video_url": (
                representative[
                    "source_video_url"
                ]
            ),

            # --------------------------------
            # Retrieval text
            # --------------------------------

            "comment_text": (
                representative[
                    "comment_text"
                ]
            ),

            "normalized_text": (
                normalized_text
            ),

            # --------------------------------
            # Representative source
            # --------------------------------

            "representative_comment_id": (
                representative[
                    "comment_id"
                ]
            ),

            "representative_author": (
                representative[
                    "author"
                ]
            ),

            # --------------------------------
            # Frequency metadata
            # --------------------------------

            "occurrence_count": (
                int(len(group))
            ),

            "unique_author_count": (
                int(
                    authors.nunique()
                )
            ),

            # --------------------------------
            # Engagement metadata
            # --------------------------------

            "total_like_count": (
                int(
                    group[
                        "like_count"
                    ].fillna(0).sum()
                )
            ),

            "max_like_count": (
                int(
                    group[
                        "like_count"
                    ].fillna(0).max()
                )
            ),

            # --------------------------------
            # Time metadata
            # --------------------------------

            "first_published_at": (
                published.min()
            ),

            "last_published_at": (
                published.max()
            ),

            # --------------------------------
            # Number of comments in the video
            # agreeing exactly with this text
            # --------------------------------

            "is_repeated_text": (
                len(group) > 1
            ),
        }

        records.append(
            record
        )

    retrieval_df = (
        pd.DataFrame(
            records
        )
    )

    return retrieval_df


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # RAW CORPUS
    # ========================================================

    raw_df = load_raw_comments()

    print("=" * 70)
    print("BUILDING CORPUS V1")
    print("=" * 70)

    print()

    print(
        f"Raw comments: "
        f"{len(raw_df):,}"
    )

    print(
        f"Raw videos: "
        f"{raw_df['video_id'].nunique():,}"
    )

    print(
        f"Unique comment IDs: "
        f"{raw_df['comment_id'].nunique():,}"
    )

    # Safety check
    duplicate_ids = (
        raw_df["comment_id"]
        .duplicated()
        .sum()
    )

    if duplicate_ids > 0:
        raise RuntimeError(
            f"Found {duplicate_ids} "
            f"duplicate comment IDs."
        )

    raw_df.to_parquet(
        RAW_OUTPUT,
        index=False,
    )

    # ========================================================
    # RETRIEVAL CORPUS
    # ========================================================

    retrieval_df = (
        build_retrieval_corpus(
            raw_df
        )
    )

    print()

    print(
        f"Retrieval documents: "
        f"{len(retrieval_df):,}"
    )

    removed_from_retrieval = (
        len(raw_df)
        - len(retrieval_df)
    )

    print(
        "Within-video duplicate rows "
        f"collapsed: "
        f"{removed_from_retrieval:,}"
    )

    print(
        "Reduction percentage: "
        f"{removed_from_retrieval / len(raw_df) * 100:.2f}%"
    )

    repeated_groups = (
        retrieval_df[
            "is_repeated_text"
        ].sum()
    )

    print(
        f"Repeated-text groups: "
        f"{repeated_groups:,}"
    )

    max_occurrence = (
        retrieval_df[
            "occurrence_count"
        ].max()
    )

    print(
        f"Maximum occurrence count "
        f"in one video: "
        f"{max_occurrence:,}"
    )

    retrieval_df.to_parquet(
        RETRIEVAL_OUTPUT,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {
        "raw_comments": (
            int(len(raw_df))
        ),

        "contributing_videos": (
            int(
                raw_df[
                    "video_id"
                ].nunique()
            )
        ),

        "unique_comment_ids": (
            int(
                raw_df[
                    "comment_id"
                ].nunique()
            )
        ),

        "retrieval_documents": (
            int(
                len(
                    retrieval_df
                )
            )
        ),

        "within_video_duplicate_rows_collapsed": (
            int(
                removed_from_retrieval
            )
        ),

        "retrieval_reduction_percentage": (
            round(
                removed_from_retrieval
                / len(raw_df)
                * 100,
                4,
            )
        ),

        "repeated_text_groups": (
            int(
                repeated_groups
            )
        ),

        "max_within_video_occurrence": (
            int(
                max_occurrence
            )
        ),

        "deduplication_scope": (
            "within_video_only"
        ),

        "global_text_deduplication": (
            False
        ),

        "minimum_length_filter": (
            None
        ),

        "language_filter": (
            None
        ),
    }

    with open(
        SUMMARY_OUTPUT,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 70)
    print("OUTPUT")
    print("=" * 70)

    print(
        f"Raw corpus:"
        f"\n{RAW_OUTPUT}"
    )

    print()

    print(
        f"Retrieval corpus:"
        f"\n{RETRIEVAL_OUTPUT}"
    )

    print()

    print(
        f"Summary:"
        f"\n{SUMMARY_OUTPUT}"
    )


if __name__ == "__main__":
    main()