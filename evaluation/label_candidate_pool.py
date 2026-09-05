from pathlib import Path
import os
import shutil

import pandas as pd


# ============================================================
# Paths
# ============================================================

LABELING_PATH = Path(
    r"evaluation\candidate_pool_v1"
    r"\candidate_pool_pilot_labeling.csv"
)

BACKUP_PATH = Path(
    r"evaluation\candidate_pool_v1"
    r"\candidate_pool_pilot_labeling_backup.csv"
)


# ============================================================
# Valid labels
# ============================================================

VALID_LABELS = {
    "0",
    "1",
    "2",
}


# ============================================================
# Helpers
# ============================================================

def is_labeled(value):
    """
    Return True only when relevance is 0, 1, or 2.
    """

    return str(value).strip() in VALID_LABELS


def atomic_save(df):
    """
    Save safely using a temporary file first.

    This reduces the chance of corrupting the labeling file
    if PowerShell / Python closes during a write.
    """

    temp_path = LABELING_PATH.with_suffix(
        ".tmp.csv"
    )

    df.to_csv(
        temp_path,
        index=False,
        encoding="utf-8-sig",
    )

    os.replace(
        temp_path,
        LABELING_PATH,
    )


def print_help():
    print()
    print("=" * 80)
    print("LABEL COMMANDS")
    print("=" * 80)

    print(
        "2 = Highly relevant"
    )

    print(
        "1 = Relevant"
    )

    print(
        "0 = Not relevant"
    )

    print(
        "s = Skip this document for now"
    )

    print(
        "q = Save and quit"
    )

    print(
        "h = Show this help"
    )

    print()


def print_relevance_guide():
    print()
    print("=" * 80)
    print("RELEVANCE GUIDE")
    print("=" * 80)

    print()
    print(
        "2 = Highly relevant"
    )

    print(
        "    Directly helps answer the query."
    )

    print(
        "    Contains a concrete opinion, "
        "recommendation, experience, comparison, "
        "specification, or other useful information."
    )

    print()
    print(
        "1 = Relevant"
    )

    print(
        "    Clearly related to the query, "
        "but provides limited information."
    )

    print(
        "    Often a relevant question, "
        "brief mention, or weakly informative statement."
    )

    print()
    print(
        "0 = Not relevant"
    )

    print(
        "    Does not meaningfully help answer the query."
    )

    print(
        "    Keyword overlap alone is not enough."
    )

    print()
    print(
        "Important:"
    )

    print(
        "Judge retrieval relevance, "
        "not whether the commenter is factually correct."
    )

    print()


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 80)
    print("CANDIDATE POOL RELEVANCE LABELING")
    print("=" * 80)

    # --------------------------------------------------------
    # Check file
    # --------------------------------------------------------

    if not LABELING_PATH.exists():

        raise FileNotFoundError(
            f"Labeling file not found:\n"
            f"{LABELING_PATH}"
        )

    # --------------------------------------------------------
    # Create one untouched backup
    # --------------------------------------------------------

    if not BACKUP_PATH.exists():

        shutil.copy2(
            LABELING_PATH,
            BACKUP_PATH,
        )

        print()
        print(
            "Created backup:"
        )

        print(
            BACKUP_PATH
        )

    # --------------------------------------------------------
    # Load CSV
    #
    # keep_default_na=False makes empty cells remain ""
    # instead of becoming NaN.
    # --------------------------------------------------------

    df = pd.read_csv(
        LABELING_PATH,
        keep_default_na=False,
        dtype=str,
    )

    required_columns = {
        "label_id",
        "query_id",
        "video_id",
        "query_type",
        "query",
        "document_id",
        "representative_comment_id",
        "comment_text",
        "relevance",
        "notes",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:

        raise RuntimeError(
            "Labeling file missing columns: "
            f"{missing_columns}"
        )

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    labeled_mask = (
        df["relevance"]
        .apply(
            is_labeled
        )
    )

    labeled_count = int(
        labeled_mask.sum()
    )

    total_count = len(df)

    print()
    print(
        f"Total judgments: "
        f"{total_count}"
    )

    print(
        f"Already labeled: "
        f"{labeled_count}"
    )

    print(
        f"Remaining: "
        f"{total_count - labeled_count}"
    )

    # --------------------------------------------------------
    # Stop immediately if complete
    # --------------------------------------------------------

    if labeled_count == total_count:

        print()
        print(
            "All documents are already labeled."
        )

        return

    print_relevance_guide()
    print_help()

    # --------------------------------------------------------
    # Labeling loop
    # --------------------------------------------------------

    skipped_indices = set()

    while True:

        # Find next unlabeled row that was not skipped
        # during this session.

        next_index = None

        for idx in df.index:

            if is_labeled(
                df.at[
                    idx,
                    "relevance",
                ]
            ):
                continue

            if idx in skipped_indices:
                continue

            next_index = idx
            break

        # ----------------------------------------------------
        # No more available rows
        # ----------------------------------------------------

        if next_index is None:

            remaining_unlabeled = [
                idx
                for idx in df.index
                if not is_labeled(
                    df.at[
                        idx,
                        "relevance",
                    ]
                )
            ]

            print()
            print("=" * 80)

            if remaining_unlabeled:

                print(
                    "No more unskipped documents "
                    "in this session."
                )

                print(
                    f"Still unlabeled: "
                    f"{len(remaining_unlabeled)}"
                )

                print(
                    "Restart the script later "
                    "to revisit skipped documents."
                )

            else:

                print(
                    "ALL JUDGMENTS COMPLETE"
                )

            print("=" * 80)

            atomic_save(
                df
            )

            break

        row = df.loc[
            next_index
        ]

        # ----------------------------------------------------
        # Current progress
        # ----------------------------------------------------

        labeled_count = int(
            df["relevance"]
            .apply(
                is_labeled
            )
            .sum()
        )

        overall_position = (
            labeled_count + 1
        )

        query_id = row[
            "query_id"
        ]

        query_mask = (
            df["query_id"]
            == query_id
        )

        query_total = int(
            query_mask.sum()
        )

        query_labeled = int(
            df.loc[
                query_mask,
                "relevance",
            ]
            .apply(
                is_labeled
            )
            .sum()
        )

        # ----------------------------------------------------
        # Display judgment item
        # ----------------------------------------------------

        print()
        print("=" * 80)

        print(
            f"OVERALL: "
            f"{overall_position}/{total_count}"
        )

        print(
            f"QUERY: "
            f"{query_id} "
            f"({query_labeled + 1}/{query_total})"
        )

        print(
            f"TYPE: "
            f"{row['query_type']}"
        )

        print(
            f"LABEL ID: "
            f"{row['label_id']}"
        )

        print("-" * 80)

        print("QUESTION:")
        print()

        print(
            row[
                "query"
            ]
        )

        print()
        print("-" * 80)

        print("COMMENT:")
        print()

        print(
            row[
                "comment_text"
            ]
        )

        print()
        print("-" * 80)

        # ----------------------------------------------------
        # Ask for label
        # ----------------------------------------------------

        while True:

            command = input(
                "Label "
                "[0/1/2, s=skip, q=quit, h=help]: "
            ).strip().lower()

            # ------------------------------------------------
            # Label
            # ------------------------------------------------

            if command in VALID_LABELS:

                df.at[
                    next_index,
                    "relevance",
                ] = command

                # Save immediately after every judgment.
                atomic_save(
                    df
                )

                break

            # ------------------------------------------------
            # Skip
            # ------------------------------------------------

            elif command == "s":

                skipped_indices.add(
                    next_index
                )

                print(
                    "Skipped for this session."
                )

                break

            # ------------------------------------------------
            # Quit
            # ------------------------------------------------

            elif command == "q":

                atomic_save(
                    df
                )

                labeled_count = int(
                    df["relevance"]
                    .apply(
                        is_labeled
                    )
                    .sum()
                )

                print()
                print("=" * 80)

                print(
                    "Progress saved."
                )

                print(
                    f"Labeled: "
                    f"{labeled_count}/{total_count}"
                )

                print(
                    f"Remaining: "
                    f"{total_count - labeled_count}"
                )

                print("=" * 80)

                return

            # ------------------------------------------------
            # Help
            # ------------------------------------------------

            elif command == "h":

                print_relevance_guide()
                print_help()

            # ------------------------------------------------
            # Invalid command
            # ------------------------------------------------

            else:

                print(
                    "Invalid input. "
                    "Use 0, 1, 2, s, q, or h."
                )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    labeled_count = int(
        df["relevance"]
        .apply(
            is_labeled
        )
        .sum()
    )

    print()
    print(
        f"Final progress: "
        f"{labeled_count}/{total_count}"
    )

    if labeled_count == total_count:

        print()
        print(
            "Label distribution:"
        )

        distribution = (
            df[
                "relevance"
            ]
            .value_counts()
            .sort_index()
        )

        for label, count in (
            distribution.items()
        ):

            print(
                f"relevance={label}: "
                f"{count}"
            )


if __name__ == "__main__":
    main()