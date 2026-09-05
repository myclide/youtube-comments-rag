from pathlib import Path

import pandas as pd


# ============================================================
# Paths
# ============================================================

JUDGMENTS_PATH = Path(
    r"evaluation\candidate_pool_v1"
    r"\benchmark_v1_judgments_gpt_v2.csv"
)

SUMMARY_PATH = Path(
    r"evaluation\candidate_pool_v1"
    r"\benchmark_v1_judgment_summary.csv"
)

UNCERTAIN_PATH = Path(
    r"evaluation\candidate_pool_v1"
    r"\benchmark_v1_uncertain_gpt_v2.csv"
)

OUTPUT_DIR = Path(
    r"evaluation\candidate_pool_v1"
)

AUDIT_OUTPUT_PATH = (
    OUTPUT_DIR
    / "benchmark_v1_query_audit.csv"
)


EXPECTED_ROWS = 3339
EXPECTED_QUERIES = 60


def main():

    print("=" * 90)
    print("BENCHMARK V1 JUDGMENT AUDIT")
    print("=" * 90)

    # ========================================================
    # Load data
    # ========================================================

    judgments = pd.read_csv(
        JUDGMENTS_PATH,
        dtype=str,
        keep_default_na=False,
    )

    summary = pd.read_csv(
        SUMMARY_PATH,
    )

    uncertain = pd.read_csv(
        UNCERTAIN_PATH,
        dtype=str,
        keep_default_na=False,
    )

    judgments["gpt_relevance"] = (
        judgments["gpt_relevance"]
        .astype(int)
    )

    # ========================================================
    # Basic checks
    # ========================================================

    print()
    print("GLOBAL")
    print("-" * 90)

    print(
        "Judgments:",
        f"{len(judgments):,}",
    )

    print(
        "Queries:",
        judgments["query_id"].nunique(),
    )

    print(
        "Videos:",
        judgments["video_id"].nunique(),
    )

    print(
        "Uncertain:",
        f"{len(uncertain):,}",
    )

    if len(judgments) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} judgments."
        )

    if judgments["query_id"].nunique() != EXPECTED_QUERIES:
        raise RuntimeError(
            f"Expected {EXPECTED_QUERIES} queries."
        )

    # ========================================================
    # Overall relevance
    # ========================================================

    relevance_counts = (
        judgments["gpt_relevance"]
        .value_counts()
        .sort_index()
    )

    print()
    print("RELEVANCE DISTRIBUTION")
    print("-" * 90)

    for label in [0, 1, 2]:

        count = int(
            relevance_counts.get(
                label,
                0,
            )
        )

        pct = (
            count
            / len(judgments)
            * 100
        )

        print(
            f"relevance={label}: "
            f"{count:,} "
            f"({pct:.2f}%)"
        )

    relevant_total = int(
        (
            judgments[
                "gpt_relevance"
            ]
            >= 1
        )
        .sum()
    )

    print()
    print(
        "Relevant >= 1:",
        f"{relevant_total:,}",
        f"({relevant_total / len(judgments) * 100:.2f}%)",
    )

    # ========================================================
    # Query-level statistics
    # ========================================================

    query_stats = []

    for query_id, group in judgments.groupby(
        "query_id",
        sort=False,
    ):

        relevance_0 = int(
            (
                group["gpt_relevance"]
                == 0
            )
            .sum()
        )

        relevance_1 = int(
            (
                group["gpt_relevance"]
                == 1
            )
            .sum()
        )

        relevance_2 = int(
            (
                group["gpt_relevance"]
                == 2
            )
            .sum()
        )

        relevant_ge_1 = (
            relevance_1
            + relevance_2
        )

        uncertain_count = int(
            (
                group[
                    "gpt_uncertain"
                ]
                .str.upper()
                == "TRUE"
            )
            .sum()
        )

        query_stats.append({
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
                relevance_0,

            "relevance_1":
                relevance_1,

            "relevance_2":
                relevance_2,

            "relevant_ge_1":
                relevant_ge_1,

            "relevant_ratio":
                relevant_ge_1
                / len(group),

            "uncertain_true":
                uncertain_count,

            "uncertain_ratio":
                uncertain_count
                / len(group),
        })

    audit = pd.DataFrame(
        query_stats
    )

    audit.to_csv(
        AUDIT_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Coverage summary
    # ========================================================

    print()
    print("PER-QUERY RELEVANCE COVERAGE")
    print("-" * 90)

    for column in [
        "judged_docs",
        "relevant_ge_1",
        "relevance_2",
        "uncertain_true",
    ]:

        print()
        print(column)

        print(
            "  min:",
            int(
                audit[column].min()
            )
        )

        print(
            "  median:",
            round(
                audit[column].median(),
                2,
            )
        )

        print(
            "  mean:",
            round(
                audit[column].mean(),
                2,
            )
        )

        print(
            "  max:",
            int(
                audit[column].max()
            )
        )

    # ========================================================
    # Critical flags
    # ========================================================

    zero_relevant = audit[
        audit[
            "relevant_ge_1"
        ]
        == 0
    ]

    zero_highly_relevant = audit[
        audit[
            "relevance_2"
        ]
        == 0
    ]

    low_relevant = audit[
        audit[
            "relevant_ge_1"
        ]
        <= 3
    ]

    high_uncertainty = (
        audit.sort_values(
            "uncertain_true",
            ascending=False,
        )
        .head(10)
    )

    print()
    print("=" * 90)
    print("CRITICAL FLAGS")
    print("=" * 90)

    print()
    print(
        "Queries with ZERO relevant documents:",
        len(zero_relevant),
    )

    if not zero_relevant.empty:

        print(
            zero_relevant[
                [
                    "query_id",
                    "query",
                    "judged_docs",
                ]
            ]
            .to_string(
                index=False
            )
        )

    print()
    print(
        "Queries with ZERO highly-relevant "
        "(label 2) documents:",
        len(zero_highly_relevant),
    )

    if not zero_highly_relevant.empty:

        print(
            zero_highly_relevant[
                [
                    "query_id",
                    "query",
                    "relevant_ge_1",
                ]
            ]
            .to_string(
                index=False
            )
        )

    print()
    print(
        "Queries with <= 3 relevant documents:",
        len(low_relevant),
    )

    if not low_relevant.empty:

        print(
            low_relevant[
                [
                    "query_id",
                    "query_type",
                    "relevant_ge_1",
                    "relevance_2",
                    "judged_docs",
                ]
            ]
            .sort_values(
                "relevant_ge_1"
            )
            .to_string(
                index=False
            )
        )

    # ========================================================
    # Most uncertain queries
    # ========================================================

    print()
    print("=" * 90)
    print("TOP 10 QUERIES BY UNCERTAINTY")
    print("=" * 90)

    print(
        high_uncertainty[
            [
                "query_id",
                "query_type",
                "uncertain_true",
                "judged_docs",
                "relevant_ge_1",
                "relevance_2",
                "query",
            ]
        ]
        .to_string(
            index=False
        )
    )

    # ========================================================
    # Query type analysis
    # ========================================================

    print()
    print("=" * 90)
    print("BY QUERY TYPE")
    print("=" * 90)

    type_summary = (
        audit.groupby(
            "query_type"
        )
        .agg(
            queries=(
                "query_id",
                "count",
            ),
            avg_judged_docs=(
                "judged_docs",
                "mean",
            ),
            avg_relevant=(
                "relevant_ge_1",
                "mean",
            ),
            avg_highly_relevant=(
                "relevance_2",
                "mean",
            ),
            avg_uncertain=(
                "uncertain_true",
                "mean",
            ),
        )
        .round(2)
    )

    print()
    print(
        type_summary.to_string()
    )

    # ========================================================
    # Final assessment
    # ========================================================

    print()
    print("=" * 90)
    print("AUDIT SUMMARY")
    print("=" * 90)

    print(
        "Queries:",
        len(audit),
    )

    print(
        "Queries with >=1 relevant document:",
        int(
            (
                audit[
                    "relevant_ge_1"
                ]
                >= 1
            )
            .sum()
        ),
    )

    print(
        "Queries with >=1 highly relevant document:",
        int(
            (
                audit[
                    "relevance_2"
                ]
                >= 1
            )
            .sum()
        ),
    )

    print(
        "Queries containing uncertain judgments:",
        int(
            (
                audit[
                    "uncertain_true"
                ]
                > 0
            )
            .sum()
        ),
    )

    print()
    print(
        "Audit file:"
    )

    print(
        AUDIT_OUTPUT_PATH
    )

    print()
    print(
        "AUDIT COMPLETE"
    )


if __name__ == "__main__":
    main()