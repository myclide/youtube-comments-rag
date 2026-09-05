from pathlib import Path
import json

import pandas as pd


# ============================================================
# Paths
# ============================================================

BASE_JUDGMENTS_PATH = Path(
    r"evaluation\candidate_pool_v1"
    r"\benchmark_v1_judgments_final.csv"
)

BASE_QRELS_PATH = Path(
    r"evaluation\candidate_pool_v1"
    r"\benchmark_v1_qrels.csv"
)

CORPUS_PATH = Path(
    r"data\corpus_v1"
    r"\retrieval_comments_v1.parquet"
)

NOVEL_UNLABELED_PATH = Path(
    r"evaluation\qwen3_dense_v1"
    r"\qwen3_dense_novel_candidates_unlabeled.csv"
)

NOVEL_LABELED_PATH = Path(
    r"evaluation\qwen3_dense_v1"
    r"\qwen3_dense_novel_candidates_labeled.csv"
)

OUTPUT_DIR = Path(
    r"evaluation\expanded_qrels_v1"
)

EXPANDED_JUDGMENTS_PATH = (
    OUTPUT_DIR
    / "benchmark_v1_judgments_expanded_qwen3.csv"
)

EXPANDED_QRELS_PATH = (
    OUTPUT_DIR
    / "benchmark_v1_qrels_expanded_qwen3.csv"
)

QUERY_SUMMARY_PATH = (
    OUTPUT_DIR
    / "benchmark_v1_expanded_query_summary.csv"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "benchmark_v1_qrels_expansion_report.json"
)


# ============================================================
# Expected sizes
# ============================================================

EXPECTED_BASE_ROWS = 3339
EXPECTED_NOVEL_ROWS = 419
EXPECTED_EXPANDED_ROWS = 3758
EXPECTED_QUERIES = 60


# ============================================================
# Columns expected in blind Work files
# ============================================================

ORIGINAL_COLUMNS = [
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


# ============================================================
# Helpers
# ============================================================

def normalize_relevance(value):

    text = str(value).strip()

    mapping = {
        "0": 0,
        "1": 1,
        "2": 2,
        "0.0": 0,
        "1.0": 1,
        "2.0": 2,
    }

    return mapping.get(text)


def normalize_boolean(value):

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

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 90)
    print("EXPANDING BENCHMARK V1 QRELS WITH QWEN3 POOL")
    print("=" * 90)

    # ========================================================
    # Load files
    # ========================================================

    base_judgments = pd.read_csv(
        BASE_JUDGMENTS_PATH,
        dtype=str,
        keep_default_na=False,
    )

    base_qrels = pd.read_csv(
        BASE_QRELS_PATH,
        dtype={
            "query_id": str,
            "document_id": str,
        },
        keep_default_na=False,
    )

    novel_unlabeled = pd.read_csv(
        NOVEL_UNLABELED_PATH,
        dtype=str,
        keep_default_na=False,
    )

    novel_labeled = pd.read_csv(
        NOVEL_LABELED_PATH,
        dtype=str,
        keep_default_na=False,
    )

    # ========================================================
    # Restore video_title in legacy base judgments
    #
    # The original 3339-row benchmark file was created before
    # video_title became part of the judging schema.
    #
    # We recover it deterministically from the frozen corpus.
    # ========================================================

    print()
    print("VIDEO TITLE CHECK")
    print("-" * 90)

    corpus_titles = pd.read_parquet(
        CORPUS_PATH,
        columns=[
            "video_id",
            "video_title",
        ],
    )

    corpus_titles[
        "video_id"
    ] = (
        corpus_titles[
            "video_id"
        ]
        .astype(str)
    )

    # --------------------------------------------------------
    # Ensure each video_id maps to one title
    # --------------------------------------------------------

    title_counts = (
        corpus_titles
        .groupby(
            "video_id"
        )[
            "video_title"
        ]
        .nunique(
            dropna=False
        )
    )

    conflicts = (
        title_counts[
            title_counts > 1
        ]
    )

    if not conflicts.empty:

        raise RuntimeError(
            "Multiple video titles found for these video IDs: "
            f"{conflicts.index.tolist()}"
        )

    corpus_titles = (
        corpus_titles[
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
        .reset_index(
            drop=True
        )
    )

    title_map = dict(
        zip(
            corpus_titles[
                "video_id"
            ],
            corpus_titles[
                "video_title"
            ],
        )
    )

    base_judgments[
        "video_id"
    ] = (
        base_judgments[
            "video_id"
        ]
        .astype(str)
    )

    if (
        "video_title"
        not in base_judgments.columns
    ):

        print(
            "Base judgments do not contain video_title."
        )

        print(
            "Restoring video titles from corpus..."
        )

        base_judgments[
            "video_title"
        ] = (
            base_judgments[
                "video_id"
            ]
            .map(
                title_map
            )
        )

    else:

        print(
            "Base judgments already contain video_title."
        )

        blank_title_mask = (
            base_judgments[
                "video_title"
            ]
            .isna()
            |
            base_judgments[
                "video_title"
            ]
            .astype(str)
            .str.strip()
            .eq("")
        )

        if blank_title_mask.any():

            print(
                "Filling blank video titles from corpus..."
            )

            base_judgments.loc[
                blank_title_mask,
                "video_title",
            ] = (
                base_judgments.loc[
                    blank_title_mask,
                    "video_id",
                ]
                .map(
                    title_map
                )
            )

    missing_title_mask = (
        base_judgments[
            "video_title"
        ]
        .isna()
        |
        base_judgments[
            "video_title"
        ]
        .astype(str)
        .str.strip()
        .eq("")
    )

    missing_titles = int(
        missing_title_mask.sum()
    )

    print(
        "Missing restored video titles:",
        missing_titles,
    )

    if missing_titles != 0:

        missing_video_ids = (
            base_judgments.loc[
                missing_title_mask,
                "video_id",
            ]
            .drop_duplicates()
            .tolist()
        )

        raise RuntimeError(
            "Could not restore video_title for: "
            f"{missing_video_ids}"
        )

    print(
        "video_title restoration: PASS"
    )

    # ========================================================
    # Input size checks
    # ========================================================

    print()
    print("INPUT SIZES")
    print("-" * 90)

    print(
        "Base judgments:",
        f"{len(base_judgments):,}",
    )

    print(
        "Base qrels:",
        f"{len(base_qrels):,}",
    )

    print(
        "Novel unlabeled:",
        f"{len(novel_unlabeled):,}",
    )

    print(
        "Novel labeled:",
        f"{len(novel_labeled):,}",
    )

    if len(base_judgments) != EXPECTED_BASE_ROWS:

        raise RuntimeError(
            f"Expected {EXPECTED_BASE_ROWS} base judgments, "
            f"found {len(base_judgments)}."
        )

    if len(base_qrels) != EXPECTED_BASE_ROWS:

        raise RuntimeError(
            f"Expected {EXPECTED_BASE_ROWS} base qrels, "
            f"found {len(base_qrels)}."
        )

    if len(novel_unlabeled) != EXPECTED_NOVEL_ROWS:

        raise RuntimeError(
            f"Expected {EXPECTED_NOVEL_ROWS} novel unlabeled rows, "
            f"found {len(novel_unlabeled)}."
        )

    if len(novel_labeled) != EXPECTED_NOVEL_ROWS:

        raise RuntimeError(
            f"Expected {EXPECTED_NOVEL_ROWS} novel labeled rows, "
            f"found {len(novel_labeled)}."
        )

    # ========================================================
    # Required columns
    # ========================================================

    required_labeled_columns = (
        set(
            ORIGINAL_COLUMNS
        )
        | {
            "gpt_relevance",
            "gpt_reason",
            "gpt_uncertain",
        }
    )

    missing_labeled_columns = (
        required_labeled_columns
        - set(
            novel_labeled.columns
        )
    )

    if missing_labeled_columns:

        raise RuntimeError(
            "Novel labeled file missing columns: "
            f"{missing_labeled_columns}"
        )

    missing_unlabeled_columns = (
        set(
            ORIGINAL_COLUMNS
        )
        - set(
            novel_unlabeled.columns
        )
    )

    if missing_unlabeled_columns:

        raise RuntimeError(
            "Novel unlabeled file missing columns: "
            f"{missing_unlabeled_columns}"
        )

    # ========================================================
    # Novel label_id integrity
    # ========================================================

    print()
    print("ID INTEGRITY")
    print("-" * 90)

    for name, frame in [
        (
            "novel_unlabeled",
            novel_unlabeled,
        ),
        (
            "novel_labeled",
            novel_labeled,
        ),
    ]:

        duplicate_ids = int(
            frame[
                "label_id"
            ]
            .duplicated()
            .sum()
        )

        print(
            f"Duplicate label_id ({name}):",
            duplicate_ids,
        )

        if duplicate_ids != 0:

            raise RuntimeError(
                f"Duplicate label_id in {name}."
            )

    unlabeled_ids = set(
        novel_unlabeled[
            "label_id"
        ]
    )

    labeled_ids = set(
        novel_labeled[
            "label_id"
        ]
    )

    if unlabeled_ids != labeled_ids:

        missing_ids = (
            unlabeled_ids
            - labeled_ids
        )

        extra_ids = (
            labeled_ids
            - unlabeled_ids
        )

        raise RuntimeError(
            "Novel labeled/unlabeled ID mismatch. "
            f"Missing={len(missing_ids)}, "
            f"Extra={len(extra_ids)}"
        )

    print(
        "Novel ID sets match: PASS"
    )

    # ========================================================
    # Verify Work did not alter original fields
    # ========================================================

    print()
    print("ORIGINAL-FIELD INTEGRITY")
    print("-" * 90)

    comparison = (
        novel_unlabeled
        .merge(
            novel_labeled,
            on="label_id",
            how="inner",
            suffixes=(
                "_original",
                "_labeled",
            ),
            validate="one_to_one",
        )
    )

    total_changed = 0

    changed_fields = {}

    for column in ORIGINAL_COLUMNS:

        if column == "label_id":
            continue

        original_column = (
            f"{column}_original"
        )

        labeled_column = (
            f"{column}_labeled"
        )

        changed = int(
            (
                comparison[
                    original_column
                ]
                != comparison[
                    labeled_column
                ]
            )
            .sum()
        )

        changed_fields[
            column
        ] = changed

        total_changed += changed

        print(
            f"Changed {column}:",
            changed,
        )

    if total_changed != 0:

        raise RuntimeError(
            "Work modified one or more frozen fields."
        )

    print(
        "Original fields unchanged: PASS"
    )

    # ========================================================
    # Normalize labels
    # ========================================================

    novel_labeled[
        "expansion_relevance"
    ] = (
        novel_labeled[
            "gpt_relevance"
        ]
        .apply(
            normalize_relevance
        )
    )

    novel_labeled[
        "expansion_uncertain"
    ] = (
        novel_labeled[
            "gpt_uncertain"
        ]
        .apply(
            normalize_boolean
        )
    )

    invalid_relevance = int(
        novel_labeled[
            "expansion_relevance"
        ]
        .isna()
        .sum()
    )

    invalid_uncertain = int(
        novel_labeled[
            "expansion_uncertain"
        ]
        .isna()
        .sum()
    )

    empty_reason = int(
        novel_labeled[
            "gpt_reason"
        ]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    unresolved = int(
        (
            novel_labeled[
                "expansion_uncertain"
            ]
            == "TRUE"
        )
        .sum()
    )

    print()
    print("LABEL INTEGRITY")
    print("-" * 90)

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

    print(
        "Unresolved expansion judgments:",
        unresolved,
    )

    if invalid_relevance != 0:

        raise RuntimeError(
            "Invalid expansion relevance labels."
        )

    if invalid_uncertain != 0:

        raise RuntimeError(
            "Invalid expansion uncertainty values."
        )

    if empty_reason != 0:

        raise RuntimeError(
            "Empty expansion reasons."
        )

    if unresolved != 0:

        raise RuntimeError(
            "Expansion still contains uncertain judgments."
        )

    # ========================================================
    # Verify all Qwen3 candidates are genuinely novel
    # ========================================================

    base_qrels[
        "query_id"
    ] = (
        base_qrels[
            "query_id"
        ]
        .astype(str)
    )

    base_qrels[
        "document_id"
    ] = (
        base_qrels[
            "document_id"
        ]
        .astype(str)
    )

    base_pairs = set(
        zip(
            base_qrels[
                "query_id"
            ],
            base_qrels[
                "document_id"
            ],
        )
    )

    novel_pairs = list(
        zip(
            novel_labeled[
                "query_id"
            ],
            novel_labeled[
                "document_id"
            ],
        )
    )

    existing_novel_pairs = [
        pair
        for pair
        in novel_pairs
        if pair in base_pairs
    ]

    duplicate_novel_pairs = (
        len(
            novel_pairs
        )
        - len(
            set(
                novel_pairs
            )
        )
    )

    print()
    print("PAIR INTEGRITY")
    print("-" * 90)

    print(
        "Novel pairs already in base qrels:",
        len(
            existing_novel_pairs
        ),
    )

    print(
        "Duplicate novel query-document pairs:",
        duplicate_novel_pairs,
    )

    if existing_novel_pairs:

        raise RuntimeError(
            "Qwen3 novel pool contains pairs "
            "already present in base qrels."
        )

    if duplicate_novel_pairs != 0:

        raise RuntimeError(
            "Qwen3 novel pool contains duplicate pairs."
        )

    # ========================================================
    # Verify base judgment/qrel pair sets agree
    # ========================================================

    base_judgment_pairs = set(
        zip(
            base_judgments[
                "query_id"
            ].astype(str),
            base_judgments[
                "document_id"
            ].astype(str),
        )
    )

    if base_judgment_pairs != base_pairs:

        raise RuntimeError(
            "Base judgments and base qrels "
            "do not contain the same query-document pairs."
        )

    print(
        "Base judgments/qrels pair sets match: PASS"
    )

    # ========================================================
    # Build expansion rows
    # ========================================================

    expansion = (
        novel_labeled
        .copy()
    )

    expansion[
        "final_relevance"
    ] = (
        expansion[
            "expansion_relevance"
        ]
        .astype(int)
    )

    expansion[
        "final_label_source"
    ] = (
        "QWEN3_POOL_EXPANSION_GPT"
    )

    expansion[
        "final_uncertain"
    ] = "FALSE"

    # ========================================================
    # Harmonize base and expansion schemas
    # ========================================================

    common_columns = [
        "label_id",
        "query_id",
        "video_id",
        "video_title",
        "query_type",
        "query",
        "document_id",
        "representative_comment_id",
        "comment_text",
        "final_relevance",
        "final_label_source",
        "final_uncertain",
    ]

    missing_base_columns = (
        set(
            common_columns
        )
        - set(
            base_judgments.columns
        )
    )

    if missing_base_columns:

        raise RuntimeError(
            "Base judgments missing common fields: "
            f"{missing_base_columns}"
        )

    missing_expansion_columns = (
        set(
            common_columns
        )
        - set(
            expansion.columns
        )
    )

    if missing_expansion_columns:

        raise RuntimeError(
            "Expansion judgments missing common fields: "
            f"{missing_expansion_columns}"
        )

    base_common = (
        base_judgments[
            common_columns
        ]
        .copy()
    )

    expansion_common = (
        expansion[
            common_columns
        ]
        .copy()
    )

    base_common[
        "final_relevance"
    ] = (
        base_common[
            "final_relevance"
        ]
        .astype(int)
    )

    expansion_common[
        "final_relevance"
    ] = (
        expansion_common[
            "final_relevance"
        ]
        .astype(int)
    )

    # ========================================================
    # Combine
    # ========================================================

    expanded = pd.concat(
        [
            base_common,
            expansion_common,
        ],
        ignore_index=True,
    )

    # ========================================================
    # Expanded integrity
    # ========================================================

    print()
    print("EXPANDED POOL INTEGRITY")
    print("-" * 90)

    print(
        "Expanded rows:",
        f"{len(expanded):,}",
    )

    print(
        "Expanded queries:",
        expanded[
            "query_id"
        ].nunique(),
    )

    duplicate_expanded_pairs = int(
        expanded[
            [
                "query_id",
                "document_id",
            ]
        ]
        .duplicated()
        .sum()
    )

    duplicate_expanded_label_ids = int(
        expanded[
            "label_id"
        ]
        .duplicated()
        .sum()
    )

    print(
        "Duplicate expanded query-document pairs:",
        duplicate_expanded_pairs,
    )

    print(
        "Duplicate expanded label_id:",
        duplicate_expanded_label_ids,
    )

    if len(expanded) != EXPECTED_EXPANDED_ROWS:

        raise RuntimeError(
            f"Expected {EXPECTED_EXPANDED_ROWS} expanded rows, "
            f"found {len(expanded)}."
        )

    if (
        expanded[
            "query_id"
        ]
        .nunique()
        != EXPECTED_QUERIES
    ):

        raise RuntimeError(
            "Expanded benchmark does not contain 60 queries."
        )

    if duplicate_expanded_pairs != 0:

        raise RuntimeError(
            "Duplicate expanded query-document pairs."
        )

    if duplicate_expanded_label_ids != 0:

        raise RuntimeError(
            "Duplicate expanded label IDs."
        )

    # ========================================================
    # Expanded distribution
    # ========================================================

    expanded_counts = (
        expanded[
            "final_relevance"
        ]
        .value_counts()
        .sort_index()
    )

    print()
    print("=" * 90)
    print("EXPANDED RELEVANCE DISTRIBUTION")
    print("=" * 90)

    for relevance in [
        0,
        1,
        2,
    ]:

        count = int(
            expanded_counts.get(
                relevance,
                0,
            )
        )

        print(
            f"relevance {relevance}: "
            f"{count:,}"
        )

    print()
    print(
        "Total:",
        f"{len(expanded):,}",
    )

    # ========================================================
    # Novel-only distribution
    # ========================================================

    novel_counts = (
        expansion[
            "final_relevance"
        ]
        .value_counts()
        .sort_index()
    )

    print()
    print("QWEN3 NOVEL JUDGMENTS")
    print("-" * 90)

    for relevance in [
        0,
        1,
        2,
    ]:

        print(
            f"relevance {relevance}:",
            int(
                novel_counts.get(
                    relevance,
                    0,
                )
            ),
        )

    # ========================================================
    # Query-level summary
    # ========================================================

    summary_rows = []

    for query_id, group in expanded.groupby(
        "query_id",
        sort=False,
    ):

        rel0 = int(
            (
                group[
                    "final_relevance"
                ]
                == 0
            )
            .sum()
        )

        rel1 = int(
            (
                group[
                    "final_relevance"
                ]
                == 1
            )
            .sum()
        )

        rel2 = int(
            (
                group[
                    "final_relevance"
                ]
                == 2
            )
            .sum()
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
                rel0,

            "relevance_1":
                rel1,

            "relevance_2":
                rel2,

            "relevant_ge_1":
                rel1 + rel2,
        })

    summary = pd.DataFrame(
        summary_rows
    )

    zero_relevant_queries = (
        summary[
            summary[
                "relevant_ge_1"
            ]
            == 0
        ]
    )

    print()
    print(
        "Queries with zero relevant docs:",
        len(
            zero_relevant_queries
        ),
    )

    if not zero_relevant_queries.empty:

        print(
            zero_relevant_queries[
                [
                    "query_id",
                    "query",
                ]
            ]
            .to_string(
                index=False
            )
        )

        raise RuntimeError(
            "At least one query has no relevant documents."
        )

    # ========================================================
    # Build expanded qrels
    #
    # Keep relevance=0 rows.
    #
    # This preserves:
    #   judged non-relevant
    # vs
    #   unjudged
    # ========================================================

    expanded_qrels = (
        expanded[
            [
                "query_id",
                "document_id",
                "final_relevance",
            ]
        ]
        .rename(
            columns={
                "final_relevance":
                    "relevance"
            }
        )
        .copy()
    )

    # ========================================================
    # Save files
    # ========================================================

    expanded.to_csv(
        EXPANDED_JUDGMENTS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    expanded_qrels.to_csv(
        EXPANDED_QRELS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        QUERY_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # JSON report
    # ========================================================

    report = {
        "base_judgments":
            len(
                base_judgments
            ),

        "qwen3_novel_judgments":
            len(
                expansion
            ),

        "expanded_judgments":
            len(
                expanded
            ),

        "queries":
            int(
                expanded[
                    "query_id"
                ].nunique()
            ),

        "novel_relevance_distribution": {
            str(label):
                int(
                    novel_counts.get(
                        label,
                        0,
                    )
                )
            for label
            in [
                0,
                1,
                2,
            ]
        },

        "expanded_relevance_distribution": {
            str(label):
                int(
                    expanded_counts.get(
                        label,
                        0,
                    )
                )
            for label
            in [
                0,
                1,
                2,
            ]
        },

        "changed_work_fields":
            changed_fields,

        "unresolved":
            unresolved,

        "integrity":
            "PASS",
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # Final output
    # ========================================================

    print()
    print("=" * 90)
    print("EXPANDED QRELS CREATED")
    print("=" * 90)

    print()
    print(
        "Expanded judgments:"
    )
    print(
        EXPANDED_JUDGMENTS_PATH
    )

    print()
    print(
        "Expanded qrels:"
    )
    print(
        EXPANDED_QRELS_PATH
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
        "Expansion report:"
    )
    print(
        REPORT_PATH
    )

    print()
    print(
        "QRELS EXPANSION: PASS"
    )


if __name__ == "__main__":
    main()