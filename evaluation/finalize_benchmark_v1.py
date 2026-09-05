from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    r"evaluation\candidate_pool_v1"
    r"\benchmark_v1_judgments_post_adjudication.csv"
)

OUTPUT_DIR = Path(
    r"evaluation\candidate_pool_v1"
)

FINAL_PATH = (
    OUTPUT_DIR
    / "benchmark_v1_judgments_final.csv"
)

QRELS_PATH = (
    OUTPUT_DIR
    / "benchmark_v1_qrels.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "benchmark_v1_final_query_summary.csv"
)


# ============================================================
# Final human adjudication
# ============================================================

HUMAN_FINAL = {

    "q023_064": {
        "relevance": 0,
        "reason": (
            "Compares ChatGPT with Google but does not "
            "explicitly discuss dictionaries, vocabulary "
            "learning, or learning words from context."
        ),
    },

    "q027_049": {
        "relevance": 0,
        "reason": (
            "'My favorite' does not identify which animal "
            "or provide a usable statement about the "
            "langurs, chital deer, or tiger."
        ),
    },

    "q031_050": {
        "relevance": 1,
        "reason": (
            "Appears to praise the girl's Uzi/gameplay, "
            "but provides no comparison with her attitude."
        ),
    },

    "q033_034": {
        "relevance": 0,
        "reason": (
            "Mentions Loneranger but does not express an "
            "intelligible opinion or useful question."
        ),
    },

    "q036_002": {
        "relevance": 1,
        "reason": (
            "Asks about OnePlus Lite and may refer to the "
            "Nord-related device, but the target entity is "
            "not explicit enough for full relevance."
        ),
    },

    "q037_037": {
        "relevance": 0,
        "reason": (
            "'Nada snake' is too ambiguous to establish "
            "a real-versus-fake judgment."
        ),
    },

    "q038_001": {
        "relevance": 2,
        "reason": (
            "The comment is a misspelled Haitian Creole "
            "expression meaning approximately 'look at the "
            "size of the anaconda,' directly addressing size."
        ),
    },

    "q048_038": {
        "relevance": 1,
        "reason": (
            "Clearly praises a performer as a top Nati "
            "performer, but does not identify whether the "
            "target is Rajeev Negi or Birbal Musafir."
        ),
    },

    "q048_045": {
        "relevance": 1,
        "reason": (
            "Criticizes the other singer's singing but "
            "does not identify which target performer is meant."
        ),
    },
}


EXPECTED_ROWS = 3339
EXPECTED_QUERIES = 60
EXPECTED_HUMAN_ROWS = 9


def main():

    print("=" * 90)
    print("FINALIZING BENCHMARK V1 JUDGMENTS")
    print("=" * 90)

    df = pd.read_csv(
        INPUT_PATH,
        dtype=str,
        keep_default_na=False,
    )

    print()
    print(
        "Input rows:",
        len(df),
    )

    print(
        "Queries:",
        df["query_id"].nunique(),
    )

    if len(df) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} rows."
        )

    if df["query_id"].nunique() != EXPECTED_QUERIES:
        raise RuntimeError(
            f"Expected {EXPECTED_QUERIES} queries."
        )

    if len(HUMAN_FINAL) != EXPECTED_HUMAN_ROWS:
        raise RuntimeError(
            "Human adjudication dictionary "
            "does not contain exactly 9 rows."
        )

    # ========================================================
    # Verify all target IDs exist exactly once
    # ========================================================

    all_ids = set(
        df["label_id"]
    )

    missing_ids = (
        set(HUMAN_FINAL)
        - all_ids
    )

    if missing_ids:
        raise RuntimeError(
            f"Missing adjudication IDs: {missing_ids}"
        )

    for label_id in HUMAN_FINAL:

        count = int(
            (
                df["label_id"]
                == label_id
            )
            .sum()
        )

        if count != 1:
            raise RuntimeError(
                f"{label_id} occurs {count} times."
            )

    # ========================================================
    # Preserve pre-human label for provenance
    # ========================================================

    df[
        "pre_human_final_relevance"
    ] = (
        df[
            "final_relevance"
        ]
    )

    df[
        "human_final_reason"
    ] = ""

    # ========================================================
    # Apply final 9 decisions
    # ========================================================

    print()
    print("=" * 90)
    print("FINAL HUMAN ADJUDICATION")
    print("=" * 90)

    changed = 0

    for label_id, decision in HUMAN_FINAL.items():

        mask = (
            df["label_id"]
            == label_id
        )

        old_value = int(
            df.loc[
                mask,
                "final_relevance",
            ].iloc[0]
        )

        new_value = int(
            decision["relevance"]
        )

        if old_value != new_value:
            changed += 1

        print(
            f"{label_id}: "
            f"{old_value} -> {new_value}"
        )

        df.loc[
            mask,
            "final_relevance",
        ] = str(
            new_value
        )

        df.loc[
            mask,
            "final_label_source",
        ] = (
            "HUMAN_FINAL_ADJUDICATION"
        )

        df.loc[
            mask,
            "final_uncertain",
        ] = "FALSE"

        df.loc[
            mask,
            "human_final_reason",
        ] = decision[
            "reason"
        ]

    print()
    print(
        "Human-reviewed rows:",
        len(HUMAN_FINAL),
    )

    print(
        "Labels changed:",
        changed,
    )

    # ========================================================
    # Final normalization
    # ========================================================

    df[
        "final_relevance"
    ] = (
        df[
            "final_relevance"
        ]
        .astype(int)
    )

    # ========================================================
    # Integrity checks
    # ========================================================

    if (
        df["label_id"]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Duplicate label_id."
        )

    if (
        df["final_relevance"]
        .isna()
        .any()
    ):
        raise RuntimeError(
            "Missing final relevance."
        )

    invalid = (
        ~df[
            "final_relevance"
        ]
        .isin(
            [0, 1, 2]
        )
    )

    if invalid.any():
        raise RuntimeError(
            "Invalid final relevance value."
        )

    unresolved = int(
        (
            df[
                "final_uncertain"
            ]
            .str.upper()
            == "TRUE"
        )
        .sum()
    )

    print()
    print(
        "Remaining uncertain:",
        unresolved,
    )

    if unresolved != 0:
        raise RuntimeError(
            "Benchmark still has unresolved judgments."
        )

    # ========================================================
    # Final relevance distribution
    # ========================================================

    counts = (
        df[
            "final_relevance"
        ]
        .value_counts()
        .sort_index()
    )

    print()
    print("=" * 90)
    print("FINAL DISTRIBUTION")
    print("=" * 90)

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
            f"relevance {relevance}: "
            f"{count}"
        )

    print()
    print(
        "Total:",
        len(df),
    )

    # ========================================================
    # Query coverage
    # ========================================================

    summary_rows = []

    for query_id, group in df.groupby(
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
                len(group),

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

    zero_relevant = (
        summary[
            summary[
                "relevant_ge_1"
            ]
            == 0
        ]
    )

    print()
    print(
        "Queries with >=1 relevant:",
        int(
            (
                summary[
                    "relevant_ge_1"
                ]
                >= 1
            )
            .sum()
        ),
        "/",
        len(summary),
    )

    if not zero_relevant.empty:
        raise RuntimeError(
            "At least one query has no relevant documents."
        )

    # ========================================================
    # Save final judgments
    # ========================================================

    df.to_csv(
        FINAL_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Save qrels
    #
    # Keep relevance=0 rows too.
    #
    # This preserves the distinction between:
    #   judged non-relevant
    # and
    #   unjudged
    # ========================================================

    qrels = (
        df[
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

    qrels.to_csv(
        QRELS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Final output
    # ========================================================

    print()
    print("=" * 90)
    print("BENCHMARK V1 JUDGMENTS FINALIZED")
    print("=" * 90)

    print()
    print(
        "Final judgments:"
    )
    print(
        FINAL_PATH
    )

    print()
    print(
        "Qrels:"
    )
    print(
        QRELS_PATH
    )

    print()
    print(
        "Query summary:"
    )
    print(
        SUMMARY_PATH
    )

    print()
    print(
        "FINALIZATION: PASS"
    )


if __name__ == "__main__":
    main()