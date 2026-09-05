from pathlib import Path

import pandas as pd


# ============================================================
# Paths
# ============================================================

ORIGINAL_PATH = Path(
    r"evaluation\candidate_pool_v1"
    r"\benchmark_v1_judgments_gpt_v2.csv"
)

MANIFEST_PATH = Path(
    r"evaluation\candidate_pool_v1"
    r"\benchmark_v1_adjudication_manifest.csv"
)

ADJUDICATION_PATH = Path(
    r"evaluation\candidate_pool_v1"
    r"\benchmark_v1_adjudication_blind_labeled.csv"
)

OUTPUT_DIR = Path(
    r"evaluation\candidate_pool_v1"
)

FINAL_PATH = (
    OUTPUT_DIR
    / "benchmark_v1_judgments_post_adjudication.csv"
)

TRANSITION_PATH = (
    OUTPUT_DIR
    / "benchmark_v1_adjudication_transitions.csv"
)

QUERY_SUMMARY_PATH = (
    OUTPUT_DIR
    / "benchmark_v1_post_adjudication_query_summary.csv"
)

PENDING_PATH = (
    OUTPUT_DIR
    / "benchmark_v1_final_uncertain_review.csv"
)


EXPECTED_TOTAL_ROWS = 3339
EXPECTED_ADJUDICATED_ROWS = 556
EXPECTED_QUERIES = 60


TEXT_COLUMNS = [
    "query_id",
    "video_id",
    "video_title",
    "query_type",
    "query",
    "document_id",
    "representative_comment_id",
    "comment_text",
]


FOCUS_QUERIES = [
    "q011",
    "q012",
    "q018",
    "q030",
    "q036",
    "q039",
    "q050",
    "q054",
    "q059",
]


# ============================================================
# Helpers
# ============================================================

def normalize_relevance(value):

    text = str(value).strip()

    mapping = {
        "0": "0",
        "1": "1",
        "2": "2",
        "0.0": "0",
        "1.0": "1",
        "2.0": "2",
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

    print("=" * 90)
    print("APPLYING BENCHMARK V1 BLIND ADJUDICATION")
    print("=" * 90)

    # ========================================================
    # Load files
    # ========================================================

    original = pd.read_csv(
        ORIGINAL_PATH,
        dtype=str,
        keep_default_na=False,
    )

    manifest = pd.read_csv(
        MANIFEST_PATH,
        dtype=str,
        keep_default_na=False,
    )

    adjudicated = pd.read_csv(
        ADJUDICATION_PATH,
        dtype=str,
        keep_default_na=False,
    )

    print()
    print("Original judgments:", len(original))
    print("Manifest rows:", len(manifest))
    print("Adjudicated rows:", len(adjudicated))

    # ========================================================
    # Basic expected sizes
    # ========================================================

    if len(original) != EXPECTED_TOTAL_ROWS:

        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_ROWS} original rows, "
            f"found {len(original)}."
        )

    if len(manifest) != EXPECTED_ADJUDICATED_ROWS:

        raise RuntimeError(
            f"Expected {EXPECTED_ADJUDICATED_ROWS} manifest rows, "
            f"found {len(manifest)}."
        )

    if len(adjudicated) != EXPECTED_ADJUDICATED_ROWS:

        raise RuntimeError(
            f"Expected {EXPECTED_ADJUDICATED_ROWS} adjudicated rows, "
            f"found {len(adjudicated)}."
        )

    # ========================================================
    # Check label IDs
    # ========================================================

    for name, df in [
        ("original", original),
        ("manifest", manifest),
        ("adjudicated", adjudicated),
    ]:

        duplicates = (
            df["label_id"]
            .duplicated()
            .sum()
        )

        print(
            f"Duplicate label_id ({name}):",
            duplicates,
        )

        if duplicates != 0:

            raise RuntimeError(
                f"Duplicate label_id in {name}."
            )

    manifest_ids = set(
        manifest["label_id"]
    )

    adjudicated_ids = set(
        adjudicated["label_id"]
    )

    if manifest_ids != adjudicated_ids:

        missing = (
            manifest_ids
            - adjudicated_ids
        )

        extra = (
            adjudicated_ids
            - manifest_ids
        )

        raise RuntimeError(
            "Adjudication IDs do not match manifest. "
            f"Missing={len(missing)}, Extra={len(extra)}"
        )

    print(
        "Adjudication ID set matches manifest: PASS"
    )

    # ========================================================
    # Validate adjudication labels
    # ========================================================

    adjudicated[
        "adjudicated_relevance"
    ] = (
        adjudicated[
            "adjudicated_relevance"
        ]
        .apply(
            normalize_relevance
        )
    )

    adjudicated[
        "adjudication_uncertain"
    ] = (
        adjudicated[
            "adjudication_uncertain"
        ]
        .apply(
            normalize_boolean
        )
    )

    invalid_relevance = (
        adjudicated[
            "adjudicated_relevance"
        ]
        .isna()
        .sum()
    )

    invalid_uncertain = (
        adjudicated[
            "adjudication_uncertain"
        ]
        .isna()
        .sum()
    )

    empty_reason = (
        adjudicated[
            "adjudication_reason"
        ]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    print()
    print(
        "Invalid adjudicated relevance:",
        invalid_relevance,
    )

    print(
        "Invalid adjudication uncertain:",
        invalid_uncertain,
    )

    print(
        "Empty adjudication reason:",
        empty_reason,
    )

    if (
        invalid_relevance != 0
        or invalid_uncertain != 0
        or empty_reason != 0
    ):

        raise RuntimeError(
            "Invalid adjudication output."
        )

    # ========================================================
    # Verify Work did not modify original fields
    # ========================================================

    print()
    print(
        "Checking blind adjudication "
        "against internal manifest..."
    )

    comparison = manifest.merge(
        adjudicated,
        on="label_id",
        how="inner",
        suffixes=(
            "_manifest",
            "_adjudicated",
        ),
        validate="one_to_one",
    )

    total_changed = 0

    for column in TEXT_COLUMNS:

        left = (
            f"{column}_manifest"
        )

        right = (
            f"{column}_adjudicated"
        )

        mismatches = int(
            (
                comparison[left]
                != comparison[right]
            )
            .sum()
        )

        print(
            f"Changed {column}:",
            mismatches,
        )

        total_changed += mismatches

    if total_changed != 0:

        raise RuntimeError(
            "Work modified one or more "
            "original benchmark fields."
        )

    print(
        "Original-field integrity: PASS"
    )

    # ========================================================
    # Create transition table
    # ========================================================

    transition_data = (
        manifest[
            [
                "label_id",
                "query_id",
                "gpt_relevance",
            ]
        ]
        .merge(
            adjudicated[
                [
                    "label_id",
                    "adjudicated_relevance",
                    "adjudication_uncertain",
                ]
            ],
            on="label_id",
            how="inner",
            validate="one_to_one",
        )
    )

    transition_data[
        "gpt_relevance"
    ] = (
        transition_data[
            "gpt_relevance"
        ]
        .apply(
            normalize_relevance
        )
    )

    transition_table = (
        transition_data
        .groupby(
            [
                "gpt_relevance",
                "adjudicated_relevance",
            ]
        )
        .size()
        .reset_index(
            name="count"
        )
        .sort_values(
            [
                "gpt_relevance",
                "adjudicated_relevance",
            ]
        )
    )

    transition_table.to_csv(
        TRANSITION_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 90)
    print("LABEL TRANSITIONS")
    print("=" * 90)

    matrix = pd.crosstab(
        transition_data[
            "gpt_relevance"
        ],
        transition_data[
            "adjudicated_relevance"
        ],
        rownames=[
            "V2"
        ],
        colnames=[
            "V3"
        ],
    )

    print()
    print(
        matrix.to_string()
    )

    changed_labels = int(
        (
            transition_data[
                "gpt_relevance"
            ]
            != transition_data[
                "adjudicated_relevance"
            ]
        )
        .sum()
    )

    unchanged_labels = (
        len(transition_data)
        - changed_labels
    )

    print()
    print(
        "Changed labels:",
        changed_labels,
    )

    print(
        "Unchanged labels:",
        unchanged_labels,
    )

    print(
        "Change rate:",
        f"{changed_labels / len(transition_data) * 100:.2f}%",
    )

    # ========================================================
    # Apply adjudication to all 3339 judgments
    # ========================================================

    adjudication_labels = (
        adjudicated[
            [
                "label_id",
                "adjudicated_relevance",
                "adjudication_reason",
                "adjudication_uncertain",
            ]
        ]
    )

    final = original.merge(
        adjudication_labels,
        on="label_id",
        how="left",
        validate="one_to_one",
    )

    final[
        "was_adjudicated"
    ] = (
        final[
            "adjudicated_relevance"
        ]
        .notna()
    )

    final[
        "final_relevance"
    ] = final[
        "gpt_relevance"
    ]

    mask = (
        final[
            "was_adjudicated"
        ]
    )

    final.loc[
        mask,
        "final_relevance",
    ] = (
        final.loc[
            mask,
            "adjudicated_relevance",
        ]
    )

    final[
        "final_label_source"
    ] = (
        "GPT_JUDGE_V2"
    )

    final.loc[
        mask,
        "final_label_source",
    ] = (
        "BLIND_ADJUDICATION_V3"
    )

    # Every original uncertain row was included in adjudication.
    #
    # Therefore final uncertainty should come from V3 for
    # adjudicated rows, and V2 for untouched rows.
    final[
        "final_uncertain"
    ] = (
        final[
            "gpt_uncertain"
        ]
    )

    final.loc[
        mask,
        "final_uncertain",
    ] = (
        final.loc[
            mask,
            "adjudication_uncertain",
        ]
    )

    # ========================================================
    # Final integrity
    # ========================================================

    if len(final) != EXPECTED_TOTAL_ROWS:

        raise RuntimeError(
            "Final row count changed."
        )

    if (
        final["label_id"]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate final label IDs."
        )

    if (
        final["query_id"]
        .nunique()
        != EXPECTED_QUERIES
    ):

        raise RuntimeError(
            "Final query count is not 60."
        )

    if (
        final[
            "final_relevance"
        ]
        .isna()
        .any()
    ):

        raise RuntimeError(
            "Missing final relevance."
        )

    final[
        "final_relevance"
    ] = (
        final[
            "final_relevance"
        ]
        .astype(int)
    )

    # ========================================================
    # Final distribution
    # ========================================================

    counts = (
        final[
            "final_relevance"
        ]
        .value_counts()
        .sort_index()
    )

    final_uncertain_count = int(
        (
            final[
                "final_uncertain"
            ]
            .str.upper()
            == "TRUE"
        )
        .sum()
    )

    print()
    print("=" * 90)
    print("POST-ADJUDICATION GLOBAL SUMMARY")
    print("=" * 90)

    print()
    print(
        "Rows:",
        len(final),
    )

    print(
        "Queries:",
        final[
            "query_id"
        ].nunique(),
    )

    print(
        "Adjudicated rows:",
        int(
            final[
                "was_adjudicated"
            ]
            .sum()
        ),
    )

    print()

    for relevance in [
        0,
        1,
        2,
    ]:

        count = int(
            counts.get(
                relevance,
                0,
            )
        )

        print(
            f"final relevance {relevance}: "
            f"{count}"
        )

    print()
    print(
        "Final uncertain:",
        final_uncertain_count,
    )

    # ========================================================
    # Focus-query statistics
    # ========================================================

    print()
    print("=" * 90)
    print("FOCUS QUERY RESULTS")
    print("=" * 90)

    focus_rows = []

    for query_id in FOCUS_QUERIES:

        group = final[
            final[
                "query_id"
            ]
            == query_id
        ]

        if group.empty:
            continue

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

        uncertain = int(
            (
                group[
                    "final_uncertain"
                ]
                .str.upper()
                == "TRUE"
            )
            .sum()
        )

        focus_rows.append({
            "query_id":
                query_id,

            "rows":
                len(group),

            "rel0":
                rel0,

            "rel1":
                rel1,

            "rel2":
                rel2,

            "relevant_ge_1":
                rel1 + rel2,

            "uncertain":
                uncertain,
        })

    focus_df = pd.DataFrame(
        focus_rows
    )

    print()
    print(
        focus_df.to_string(
            index=False
        )
    )

    # ========================================================
    # Query summary
    # ========================================================

    summary_rows = []

    for query_id, group in final.groupby(
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

        uncertain = int(
            (
                group[
                    "final_uncertain"
                ]
                .str.upper()
                == "TRUE"
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
                len(group),

            "relevance_0":
                rel0,

            "relevance_1":
                rel1,

            "relevance_2":
                rel2,

            "relevant_ge_1":
                rel1 + rel2,

            "uncertain":
                uncertain,
        })

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        QUERY_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Remaining uncertain rows
    # ========================================================

    pending = (
        final[
            final[
                "final_uncertain"
            ]
            .str.upper()
            == "TRUE"
        ]
        .copy()
    )

    pending.to_csv(
        PENDING_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Save post-adjudication judgments
    # ========================================================

    final.to_csv(
        FINAL_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Outputs
    # ========================================================

    print()
    print("=" * 90)
    print("OUTPUT FILES")
    print("=" * 90)

    print()
    print(
        "Post-adjudication judgments:"
    )
    print(
        FINAL_PATH
    )

    print()
    print(
        "Transition table:"
    )
    print(
        TRANSITION_PATH
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
        "Remaining uncertain review:"
    )
    print(
        PENDING_PATH
    )

    print()
    print(
        "ADJUDICATION APPLICATION: PASS"
    )


if __name__ == "__main__":
    main()