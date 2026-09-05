import json
from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

DATA_ROOT = Path("data")

STRUCTURED_ROOT = (
    DATA_ROOT
    / "structured_comments"
)

BY_VIDEO_DIR = (
    STRUCTURED_ROOT
    / "by_video"
)

STATE_PATH = (
    STRUCTURED_ROOT
    / "collection_state.json"
)


# ============================================================
# Load state
# ============================================================

def load_state():
    with open(
        STATE_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# ============================================================
# Load all per-video files
# ============================================================

def load_video_files():
    files = sorted(
        BY_VIDEO_DIR.glob("*.jsonl")
    )

    frames = []

    for file in files:
        df = pd.read_json(
            file,
            lines=True,
        )

        if df.empty:
            print(
                f"WARNING: empty file: {file}"
            )
            continue

        df["_source_file"] = file.name

        frames.append(df)

    if not frames:
        raise RuntimeError(
            "No JSONL files found."
        )

    return (
        files,
        pd.concat(
            frames,
            ignore_index=True,
        ),
    )


# ============================================================
# Main
# ============================================================

def main():
    state = load_state()

    files, df = load_video_files()

    completed_ids = set(
        state["completed"].keys()
    )

    skipped_ids = set(
        state["skipped"].keys()
    )

    failed_ids = set(
        state["failed"].keys()
    )

    file_ids = {
        file.stem
        for file in files
    }

    # ========================================================
    # STATE / FILE CONSISTENCY
    # ========================================================

    missing_files = (
        completed_ids - file_ids
    )

    unexpected_files = (
        file_ids - completed_ids
    )

    print("=" * 70)
    print("CORPUS INTEGRITY AUDIT")
    print("=" * 70)

    print()
    print("STATE SUMMARY")
    print("-" * 70)

    print(
        f"Completed videos: "
        f"{len(completed_ids):,}"
    )

    print(
        f"Skipped videos: "
        f"{len(skipped_ids):,}"
    )

    print(
        f"Failed videos: "
        f"{len(failed_ids):,}"
    )

    print(
        f"JSONL files: "
        f"{len(files):,}"
    )

    print()

    print(
        f"Completed videos missing files: "
        f"{len(missing_files):,}"
    )

    print(
        f"Unexpected JSONL files: "
        f"{len(unexpected_files):,}"
    )

    if missing_files:
        print(
            "Missing file video IDs:"
        )

        for video_id in sorted(
            missing_files
        ):
            print(video_id)

    if unexpected_files:
        print(
            "Unexpected file video IDs:"
        )

        for video_id in sorted(
            unexpected_files
        ):
            print(video_id)

    # ========================================================
    # CORPUS SIZE
    # ========================================================

    print()
    print("CORPUS SIZE")
    print("-" * 70)

    print(
        f"Total comment records: "
        f"{len(df):,}"
    )

    print(
        f"Unique video IDs: "
        f"{df['video_id'].nunique():,}"
    )

    print(
        f"Unique comment IDs: "
        f"{df['comment_id'].nunique():,}"
    )

    global_duplicate_ids = (
        df["comment_id"]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate comment IDs: "
        f"{global_duplicate_ids:,}"
    )

    # ========================================================
    # MISSING VALUES
    # ========================================================

    print()
    print("MISSING VALUES")
    print("-" * 70)

    important_columns = [
        "video_id",
        "video_title",
        "comment_id",
        "comment_text",
        "published_at",
        "collected_at",
    ]

    for column in important_columns:
        if column in df.columns:
            print(
                f"{column}: "
                f"{df[column].isna().sum():,}"
            )

    if "author" in df.columns:
        print(
            f"author: "
            f"{df['author'].isna().sum():,}"
        )

    # ========================================================
    # EMPTY TEXT
    # ========================================================

    text = (
        df["comment_text"]
        .fillna("")
        .astype(str)
    )

    empty_text = (
        text.str.strip()
        .eq("")
        .sum()
    )

    print()
    print("TEXT QUALITY")
    print("-" * 70)

    print(
        f"Empty comments: "
        f"{empty_text:,}"
    )

    # ========================================================
    # MULTILINE
    # ========================================================

    multiline = (
        text
        .str.contains(
            "\n",
            regex=False,
        )
        .sum()
    )

    print(
        f"Multiline comments: "
        f"{multiline:,}"
    )

    print(
        "Multiline percentage: "
        f"{multiline / len(df) * 100:.2f}%"
    )

    # ========================================================
    # LENGTH DISTRIBUTION
    # ========================================================

    lengths = text.str.len()

    print()
    print("COMMENT LENGTH")
    print("-" * 70)

    print(
        f"Mean: "
        f"{lengths.mean():.2f}"
    )

    print(
        f"Median: "
        f"{lengths.median():.2f}"
    )

    print(
        f"Min: "
        f"{lengths.min():,}"
    )

    print(
        f"Max: "
        f"{lengths.max():,}"
    )

    print()

    print("Length percentiles:")

    print(
        lengths.quantile(
            [
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
    )

    # ========================================================
    # COMMENTS PER VIDEO
    # ========================================================

    comments_per_video = (
        df.groupby("video_id")
        .size()
        .sort_values(
            ascending=False
        )
    )

    print()
    print("COMMENTS PER VIDEO")
    print("-" * 70)

    print(
        f"Mean: "
        f"{comments_per_video.mean():.2f}"
    )

    print(
        f"Median: "
        f"{comments_per_video.median():.2f}"
    )

    print(
        f"Min: "
        f"{comments_per_video.min():,}"
    )

    print(
        f"Max: "
        f"{comments_per_video.max():,}"
    )

    print()

    print(
        "Videos at 500-comment cap: "
        f"{(comments_per_video == 500).sum():,}"
    )

    print(
        "Videos below cap: "
        f"{(comments_per_video < 500).sum():,}"
    )

    print(
        "Videos above cap (should be 0): "
        f"{(comments_per_video > 500).sum():,}"
    )

    # ========================================================
    # EXACT TEXT DUPLICATES
    # ========================================================

    normalized_text = (
        text
        .str.strip()
        .str.lower()
    )

    duplicate_text_rows = (
        normalized_text
        .duplicated()
        .sum()
    )

    print()
    print("EXACT TEXT DUPLICATES")
    print("-" * 70)

    print(
        f"Duplicate text rows: "
        f"{duplicate_text_rows:,}"
    )

    print(
        "Duplicate text percentage: "
        f"{duplicate_text_rows / len(df) * 100:.2f}%"
    )

    # ========================================================
    # TIME RANGE
    # ========================================================

    published = pd.to_datetime(
        df["published_at"],
        errors="coerce",
        utc=True,
    )

    print()
    print("COMMENT TIME RANGE")
    print("-" * 70)

    print(
        f"Earliest comment: "
        f"{published.min()}"
    )

    print(
        f"Latest comment: "
        f"{published.max()}"
    )

    print(
        f"Invalid timestamps: "
        f"{published.isna().sum():,}"
    )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    problems = []

    if missing_files:
        problems.append(
            "completed videos missing files"
        )

    if unexpected_files:
        problems.append(
            "unexpected JSONL files"
        )

    if failed_ids:
        problems.append(
            "failed videos remain"
        )

    if global_duplicate_ids > 0:
        problems.append(
            "duplicate comment IDs"
        )

    if empty_text > 0:
        problems.append(
            "empty comment text"
        )

    if (
        comments_per_video > 500
    ).any():
        problems.append(
            "video exceeds 500-comment cap"
        )

    print()
    print("=" * 70)
    print("FINAL STATUS")
    print("=" * 70)

    if problems:
        print(
            "AUDIT STATUS: REVIEW REQUIRED"
        )

        for problem in problems:
            print(
                f"- {problem}"
            )

    else:
        print(
            "AUDIT STATUS: PASSED"
        )


if __name__ == "__main__":
    main()