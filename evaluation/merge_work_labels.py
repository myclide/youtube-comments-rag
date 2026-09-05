from pathlib import Path
import json
import re

import pandas as pd


# ============================================================
# Paths
# ============================================================

BATCH_DIR = Path(
    r"evaluation\candidate_pool_v1\work_batches_v2"
)

REFERENCE_PATH = Path(
    r"evaluation\candidate_pool_v1"
    r"\candidate_pool_benchmark_v1_unlabeled.csv"
)

OUTPUT_DIR = Path(
    r"evaluation\candidate_pool_v1"
)

MERGED_PATH = (
    OUTPUT_DIR
    / "benchmark_v1_judgments_gpt_v2.csv"
)

UNCERTAIN_PATH = (
    OUTPUT_DIR
    / "benchmark_v1_uncertain_gpt_v2.csv"
)

QUERY_SUMMARY_PATH = (
    OUTPUT_DIR
    / "benchmark_v1_judgment_summary.csv"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "benchmark_v1_label_merge_report.json"
)


# ============================================================
# Expected benchmark properties
# ============================================================

EXPECTED_BATCHES = 6
EXPECTED_ROWS = 3339
EXPECTED_QUERIES = 60


# ============================================================
# Required columns
# ============================================================

BASE_COLUMNS = [
    "label_id",
    "query_id",
    "video_id",
    "query_type",
    "query",
    "document_id",
    "representative_comment_id",
    "comment_text",
]

LABEL_COLUMNS = [
    "gpt_relevance",
    "gpt_reason",
    "gpt_uncertain",
]


# ============================================================
# Helpers
# ============================================================

def query_number(query_id):
    """
    q001 -> 1
    """

    match = re.search(
        r"(\d+)",
        str(query_id),
    )

    if not match:
        return 999999

    return int(
        match.group(1)
    )


def normalize_relevance(value):
    """
    Normalize possible CSV representations:
        0
        1
        2
        0.0
        1.0
        2.0
    into strings:
        "0"
        "1"
        "2"
    """

    text = str(value).strip()

    mapping = {
        "0": "0",
        "1": "1",
        "2": "2",
        "0.0": "0",
        "1.0": "1",
        "2.0": "2",
    }

    if text not in mapping:
        return None

    return mapping[text]


def normalize_uncertain(value):
    """
    Normalize:
        TRUE / FALSE
        True / False
        true / false

    into:
        TRUE / FALSE
    """

    text = str(value).strip().upper()

    if text == "TRUE":
        return "TRUE"

    if text == "FALSE":
        return "FALSE"

    return None


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 80)
    print("MERGING BENCHMARK V1 WORK LABELS")
    print("=" * 80)

    # ========================================================
    # Find batch files
    # ========================================================

    batch_files = sorted(
        BATCH_DIR.glob(
            "*_v2_labeled.csv"
        )
    )

    print()
    print("Batch files found:")

    for path in batch_files:
        print(
            f"  {path.name}"
        )

    print()
    print(
        "Number of batch files:",
        len(batch_files),
    )

    if len(batch_files) != EXPECTED_BATCHES:

        raise RuntimeError(
            f"Expected {EXPECTED_BATCHES} labeled batches, "
            f"found {len(batch_files)}."
        )

    # ========================================================
    # Load original frozen candidate pool
    # ========================================================

    print()
    print("Loading frozen reference candidate pool...")

    reference = pd.read_csv(
        REFERENCE_PATH,
        dtype=str,
        keep_default_na=False,
    )

    missing_reference = (
        set(BASE_COLUMNS)
        - set(reference.columns)
    )

    if missing_reference:

        raise RuntimeError(
            "Reference file missing columns: "
            f"{missing_reference}"
        )

    print(
        "Reference rows:",
        f"{len(reference):,}",
    )

    print(
        "Reference queries:",
        reference[
            "query_id"
        ].nunique(),
    )

    # Keep original benchmark ordering.
    reference[
        "_reference_order"
    ] = range(
        len(reference)
    )

    # ========================================================
    # Load all labeled batches
    # ========================================================

    frames = []

    batch_stats = []

    for batch_index, path in enumerate(
        batch_files,
        start=1,
    ):

        print()
        print("-" * 80)

        print(
            f"Reading batch {batch_index}: "
            f"{path.name}"
        )

        df = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
        )

        required = (
            set(BASE_COLUMNS)
            | set(LABEL_COLUMNS)
        )

        missing = (
            required
            - set(df.columns)
        )

        if missing:

            raise RuntimeError(
                f"{path.name} missing columns: "
                f"{missing}"
            )

        # -----------------------------------------------
        # Normalize labels
        # -----------------------------------------------

        df[
            "gpt_relevance"
        ] = (
            df[
                "gpt_relevance"
            ]
            .apply(
                normalize_relevance
            )
        )

        df[
            "gpt_uncertain"
        ] = (
            df[
                "gpt_uncertain"
            ]
            .apply(
                normalize_uncertain
            )
        )

        invalid_relevance = (
            df[
                "gpt_relevance"
            ]
            .isna()
            .sum()
        )

        invalid_uncertain = (
            df[
                "gpt_uncertain"
            ]
            .isna()
            .sum()
        )

        empty_reason = (
            df[
                "gpt_reason"
            ]
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )

        duplicated_ids = (
            df[
                "label_id"
            ]
            .duplicated()
            .sum()
        )

        print(
            "Rows:",
            len(df),
        )

        print(
            "Queries:",
            df[
                "query_id"
            ].nunique(),
        )

        print(
            "Duplicate label_id:",
            duplicated_ids,
        )

        print(
            "Invalid relevance:",
            invalid_relevance,
        )

        print(
            "Invalid uncertain:",
            invalid_uncertain,
        )

        print(
            "Empty reasons:",
            empty_reason,
        )

        if duplicated_ids != 0:

            raise RuntimeError(
                f"Duplicate label_id inside {path.name}"
            )

        if invalid_relevance != 0:

            raise RuntimeError(
                f"Invalid gpt_relevance inside {path.name}"
            )

        if invalid_uncertain != 0:

            raise RuntimeError(
                f"Invalid gpt_uncertain inside {path.name}"
            )

        if empty_reason != 0:

            raise RuntimeError(
                f"Empty gpt_reason inside {path.name}"
            )

        counts = (
            df[
                "gpt_relevance"
            ]
            .value_counts()
            .to_dict()
        )

        uncertain_count = int(
            (
                df[
                    "gpt_uncertain"
                ]
                == "TRUE"
            )
            .sum()
        )

        batch_stats.append({
            "batch":
                batch_index,

            "file":
                path.name,

            "rows":
                len(df),

            "queries":
                df[
                    "query_id"
                ].nunique(),

            "relevance_0":
                int(
                    counts.get(
                        "0",
                        0,
                    )
                ),

            "relevance_1":
                int(
                    counts.get(
                        "1",
                        0,
                    )
                ),

            "relevance_2":
                int(
                    counts.get(
                        "2",
                        0,
                    )
                ),

            "uncertain_true":
                uncertain_count,
        })

        frames.append(
            df
        )

    # ========================================================
    # Merge batches
    # ========================================================

    merged_labels = pd.concat(
        frames,
        ignore_index=True,
    )

    print()
    print("=" * 80)
    print("GLOBAL INTEGRITY CHECKS")
    print("=" * 80)

    # ========================================================
    # Check total row count
    # ========================================================

    actual_rows = len(
        merged_labels
    )

    print(
        "Merged rows:",
        f"{actual_rows:,}",
    )

    if actual_rows != EXPECTED_ROWS:

        raise RuntimeError(
            f"Expected {EXPECTED_ROWS:,} rows, "
            f"found {actual_rows:,}."
        )

    # ========================================================
    # Check query count
    # ========================================================

    actual_queries = (
        merged_labels[
            "query_id"
        ]
        .nunique()
    )

    print(
        "Unique queries:",
        actual_queries,
    )

    if actual_queries != EXPECTED_QUERIES:

        raise RuntimeError(
            f"Expected {EXPECTED_QUERIES} queries, "
            f"found {actual_queries}."
        )

    # ========================================================
    # Check global label_id uniqueness
    # ========================================================

    duplicate_global = (
        merged_labels[
            "label_id"
        ]
        .duplicated()
        .sum()
    )

    print(
        "Duplicate label_id:",
        duplicate_global,
    )

    if duplicate_global != 0:

        duplicates = (
            merged_labels[
                merged_labels[
                    "label_id"
                ]
                .duplicated(
                    keep=False
                )
            ][
                [
                    "label_id",
                    "query_id",
                ]
            ]
        )

        print()
        print(
            duplicates.head(
                20
            )
        )

        raise RuntimeError(
            "Duplicate label_id across batches."
        )

    # ========================================================
    # Compare label IDs with original frozen pool
    # ========================================================

    reference_ids = set(
        reference[
            "label_id"
        ]
    )

    labeled_ids = set(
        merged_labels[
            "label_id"
        ]
    )

    missing_ids = (
        reference_ids
        - labeled_ids
    )

    extra_ids = (
        labeled_ids
        - reference_ids
    )

    print(
        "Missing label_id:",
        len(
            missing_ids
        ),
    )

    print(
        "Unexpected label_id:",
        len(
            extra_ids
        ),
    )

    if missing_ids:

        print(
            "Example missing IDs:",
            sorted(
                missing_ids
            )[:20],
        )

        raise RuntimeError(
            "Some frozen candidate-pool rows "
            "were not labeled."
        )

    if extra_ids:

        print(
            "Example extra IDs:",
            sorted(
                extra_ids
            )[:20],
        )

        raise RuntimeError(
            "Labeled files contain unexpected rows."
        )

    # ========================================================
    # Verify Work did not modify original benchmark content
    # ========================================================

    print()
    print("Checking original fields against frozen pool...")

    comparison_columns = [
        "query_id",
        "video_id",
        "query_type",
        "query",
        "document_id",
        "representative_comment_id",
        "comment_text",
    ]

    reference_compare = (
        reference[
            [
                "label_id",
                *comparison_columns,
            ]
        ]
        .copy()
    )

    labeled_compare = (
        merged_labels[
            [
                "label_id",
                *comparison_columns,
            ]
        ]
        .copy()
    )

    compare = reference_compare.merge(
        labeled_compare,
        on="label_id",
        how="inner",
        suffixes=(
            "_reference",
            "_labeled",
        ),
        validate="one_to_one",
    )

    altered_counts = {}

    for column in comparison_columns:

        reference_column = (
            f"{column}_reference"
        )

        labeled_column = (
            f"{column}_labeled"
        )

        mismatches = (
            compare[
                reference_column
            ]
            != compare[
                labeled_column
            ]
        )

        count = int(
            mismatches.sum()
        )

        altered_counts[
            column
        ] = count

        print(
            f"Changed {column}:",
            count,
        )

    total_changed = sum(
        altered_counts.values()
    )

    if total_changed != 0:

        raise RuntimeError(
            "Work modified one or more "
            "frozen benchmark fields."
        )

    # ========================================================
    # Restore original candidate-pool ordering
    # ========================================================

    labels_only = (
        merged_labels[
            [
                "label_id",
                "gpt_relevance",
                "gpt_reason",
                "gpt_uncertain",
            ]
        ]
    )

    # Use the frozen original rows as the canonical source.
    final_df = reference.merge(
        labels_only,
        on="label_id",
        how="left",
        validate="one_to_one",
    )

    final_df = (
        final_df
        .sort_values(
            "_reference_order"
        )
        .drop(
            columns=[
                "_reference_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # Final label validation
    # ========================================================

    missing_relevance = (
        final_df[
            "gpt_relevance"
        ]
        .isna()
        .sum()
    )

    missing_reason = (
        final_df[
            "gpt_reason"
        ]
        .isna()
        .sum()
    )

    missing_uncertain = (
        final_df[
            "gpt_uncertain"
        ]
        .isna()
        .sum()
    )

    print()
    print(
        "Missing relevance:",
        missing_relevance,
    )

    print(
        "Missing reason:",
        missing_reason,
    )

    print(
        "Missing uncertain:",
        missing_uncertain,
    )

    if (
        missing_relevance != 0
        or missing_reason != 0
        or missing_uncertain != 0
    ):

        raise RuntimeError(
            "Missing labels after merge."
        )

    # ========================================================
    # Label distribution
    # ========================================================

    relevance_counts = (
        final_df[
            "gpt_relevance"
        ]
        .value_counts()
        .sort_index()
    )

    uncertain_count = int(
        (
            final_df[
                "gpt_uncertain"
            ]
            == "TRUE"
        )
        .sum()
    )

    print()
    print("Relevance distribution:")

    for label in [
        "0",
        "1",
        "2",
    ]:

        print(
            f"  {label}:",
            int(
                relevance_counts.get(
                    label,
                    0,
                )
            ),
        )

    print(
        "Uncertain TRUE:",
        uncertain_count,
    )

    # ========================================================
    # Save merged judgments
    # ========================================================

    final_df.to_csv(
        MERGED_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Save uncertain-only review file
    # ========================================================

    uncertain_df = (
        final_df[
            final_df[
                "gpt_uncertain"
            ]
            == "TRUE"
        ]
        .copy()
    )

    uncertain_df.to_csv(
        UNCERTAIN_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Query-level summary
    # ========================================================

    summary_rows = []

    for query_id, group in (
        final_df.groupby(
            "query_id",
            sort=False,
        )
    ):

        counts = (
            group[
                "gpt_relevance"
            ]
            .value_counts()
            .to_dict()
        )

        summary_rows.append({
            "query_id":
                query_id,

            "video_id":
                group[
                    "video_id"
                ].iloc[0],

            "query_type":
                group[
                    "query_type"
                ].iloc[0],

            "query":
                group[
                    "query"
                ].iloc[0],

            "judged_docs":
                len(
                    group
                ),

            "relevance_0":
                int(
                    counts.get(
                        "0",
                        0,
                    )
                ),

            "relevance_1":
                int(
                    counts.get(
                        "1",
                        0,
                    )
                ),

            "relevance_2":
                int(
                    counts.get(
                        "2",
                        0,
                    )
                ),

            "relevant_ge_1":
                int(
                    (
                        group[
                            "gpt_relevance"
                        ]
                        .astype(int)
                        >= 1
                    )
                    .sum()
                ),

            "highly_relevant_2":
                int(
                    (
                        group[
                            "gpt_relevance"
                        ]
                        .astype(int)
                        == 2
                    )
                    .sum()
                ),

            "uncertain_true":
                int(
                    (
                        group[
                            "gpt_uncertain"
                        ]
                        == "TRUE"
                    )
                    .sum()
                ),
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df[
        "_query_number"
    ] = (
        summary_df[
            "query_id"
        ]
        .apply(
            query_number
        )
    )

    summary_df = (
        summary_df
        .sort_values(
            "_query_number"
        )
        .drop(
            columns=[
                "_query_number"
            ]
        )
    )

    summary_df.to_csv(
        QUERY_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Build report
    # ========================================================

    report = {
        "benchmark":
            "benchmark_v1",

        "expected_batches":
            EXPECTED_BATCHES,

        "actual_batches":
            len(
                batch_files
            ),

        "expected_rows":
            EXPECTED_ROWS,

        "actual_rows":
            len(
                final_df
            ),

        "expected_queries":
            EXPECTED_QUERIES,

        "actual_queries":
            int(
                final_df[
                    "query_id"
                ].nunique()
            ),

        "unique_label_ids":
            int(
                final_df[
                    "label_id"
                ].nunique()
            ),

        "missing_label_ids":
            len(
                missing_ids
            ),

        "unexpected_label_ids":
            len(
                extra_ids
            ),

        "modified_original_fields":
            altered_counts,

        "relevance_distribution": {
            "0":
                int(
                    relevance_counts.get(
                        "0",
                        0,
                    )
                ),

            "1":
                int(
                    relevance_counts.get(
                        "1",
                        0,
                    )
                ),

            "2":
                int(
                    relevance_counts.get(
                        "2",
                        0,
                    )
                ),
        },

        "uncertain_true":
            uncertain_count,

        "batches":
            batch_stats,

        "integrity_check":
            "PASS",
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # Final output
    # ========================================================

    print()
    print("=" * 80)
    print("MERGE COMPLETE")
    print("=" * 80)

    print(
        "Batches:",
        len(
            batch_files
        ),
    )

    print(
        "Rows:",
        f"{len(final_df):,}",
    )

    print(
        "Queries:",
        final_df[
            "query_id"
        ].nunique(),
    )

    print(
        "Unique label IDs:",
        final_df[
            "label_id"
        ].nunique(),
    )

    print(
        "Uncertain judgments:",
        uncertain_count,
    )

    print()
    print(
        "Merged judgments:"
    )

    print(
        MERGED_PATH
    )

    print()
    print(
        "Uncertain-only review file:"
    )

    print(
        UNCERTAIN_PATH
    )

    print()
    print(
        "Query summary:"
    )

    print(
        QUERY_SUMMARY_PATH
    )

    print()
    print(
        "Merge report:"
    )

    print(
        REPORT_PATH
    )

    print()
    print(
        "INTEGRITY CHECK: PASS"
    )


if __name__ == "__main__":
    main()