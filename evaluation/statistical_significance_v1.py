from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

BOOTSTRAP_ITERATIONS = 10000

PERMUTATION_ITERATIONS = 100000

CONFIDENCE_LEVEL = 0.95

RANDOM_SEED = 20260904

EPSILON = 1e-12


# ============================================================
# Paths
# ============================================================

EXPANDED_RESULTS_PATH = Path(
    r"evaluation\expanded_results_v1"
    r"\expanded_metrics_per_query.csv"
)

RERANKER_ABLATION_PATH = Path(
    r"evaluation\reranker_candidate_ablation_v1"
    r"\reranker_candidate_ablation_per_query.csv"
)

OUTPUT_DIR = Path(
    r"evaluation\statistical_significance_v1"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "statistical_significance_summary.csv"
)

BOOTSTRAP_PATH = (
    OUTPUT_DIR
    / "bootstrap_distributions.csv"
)

QUERY_DELTA_PATH = (
    OUTPUT_DIR
    / "paired_query_deltas.csv"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "statistical_significance_report.json"
)


# ============================================================
# Metrics
# ============================================================

METRICS = [
    "ndcg_at_10",
    "mrr_at_10",
    "precision_at_10",
]


# ============================================================
# Comparisons
# ============================================================

COMPARISONS = [
    {
        "comparison":
            "qwen3_bge_vs_minilm",

        "baseline_label":
            "MiniLM dense",

        "system_label":
            "Qwen3 Top25 + BGE reranker",
    },

    {
        "comparison":
            "triple_bge_vs_qwen3_bge",

        "baseline_label":
            "Qwen3 Top25 + BGE reranker",

        "system_label":
            "BM25 + MiniLM + Qwen3 + BGE reranker",
    },
]


# ============================================================
# Statistical helpers
# ============================================================

def bootstrap_paired(
    baseline,
    system,
    iterations,
    rng,
):
    """
    Paired bootstrap over queries.

    Each bootstrap sample resamples query indices WITH
    replacement.

    For every resample we calculate:

        mean(system)
        mean(baseline)
        absolute difference
        relative improvement
    """

    baseline = np.asarray(
        baseline,
        dtype=float,
    )

    system = np.asarray(
        system,
        dtype=float,
    )

    if len(baseline) != len(system):

        raise ValueError(
            "Paired arrays must have equal length."
        )

    n = len(
        baseline
    )

    differences = np.empty(
        iterations,
        dtype=float,
    )

    relative_improvements = np.empty(
        iterations,
        dtype=float,
    )

    baseline_means = np.empty(
        iterations,
        dtype=float,
    )

    system_means = np.empty(
        iterations,
        dtype=float,
    )

    for i in range(
        iterations
    ):

        indices = rng.integers(
            0,
            n,
            size=n,
        )

        baseline_sample = (
            baseline[
                indices
            ]
        )

        system_sample = (
            system[
                indices
            ]
        )

        baseline_mean = float(
            baseline_sample.mean()
        )

        system_mean = float(
            system_sample.mean()
        )

        difference = (
            system_mean
            - baseline_mean
        )

        if abs(
            baseline_mean
        ) > EPSILON:

            relative = (
                difference
                / baseline_mean
            )

        else:

            relative = np.nan

        baseline_means[i] = (
            baseline_mean
        )

        system_means[i] = (
            system_mean
        )

        differences[i] = (
            difference
        )

        relative_improvements[i] = (
            relative
        )

    return {
        "baseline_mean":
            baseline_means,

        "system_mean":
            system_means,

        "difference":
            differences,

        "relative_improvement":
            relative_improvements,
    }


def percentile_ci(
    values,
    confidence_level,
):
    """
    Percentile bootstrap confidence interval.
    """

    alpha = (
        1.0
        - confidence_level
    )

    lower_percentile = (
        100
        * alpha
        / 2
    )

    upper_percentile = (
        100
        * (
            1
            - alpha / 2
        )
    )

    lower = float(
        np.nanpercentile(
            values,
            lower_percentile,
        )
    )

    upper = float(
        np.nanpercentile(
            values,
            upper_percentile,
        )
    )

    return (
        lower,
        upper,
    )


def paired_sign_flip_test(
    baseline,
    system,
    iterations,
    rng,
):
    """
    Monte-Carlo paired randomization test.

    Null hypothesis:
        system and baseline are exchangeable within each query.

    We preserve the magnitude of each paired difference but
    randomly flip its sign.

    Test statistic:
        absolute mean paired difference.

    Returns an approximate two-sided p-value.
    """

    baseline = np.asarray(
        baseline,
        dtype=float,
    )

    system = np.asarray(
        system,
        dtype=float,
    )

    differences = (
        system
        - baseline
    )

    observed = abs(
        float(
            differences.mean()
        )
    )

    exceed_count = 0

    # Process in chunks so memory stays small.
    chunk_size = 5000

    completed = 0

    while completed < iterations:

        current_size = min(
            chunk_size,
            iterations - completed,
        )

        signs = rng.choice(
            np.array(
                [
                    -1.0,
                    1.0,
                ]
            ),
            size=(
                current_size,
                len(
                    differences
                ),
            ),
        )

        randomized_means = (
            signs
            * differences
        ).mean(
            axis=1
        )

        exceed_count += int(
            (
                np.abs(
                    randomized_means
                )
                >= observed
            )
            .sum()
        )

        completed += current_size

    # +1 correction prevents zero Monte-Carlo p-values.
    p_value = (
        exceed_count
        + 1
    ) / (
        iterations
        + 1
    )

    return float(
        p_value
    )


# ============================================================
# Dataset preparation
# ============================================================

def prepare_comparison_data(
    expanded,
    ablation,
    comparison_name,
):

    if (
        comparison_name
        == "qwen3_bge_vs_minilm"
    ):

        baseline = (
            expanded[
                expanded[
                    "system"
                ]
                == "minilm_dense"
            ]
            .copy()
        )

        system = (
            ablation[
                ablation[
                    "configuration"
                ]
                == "qwen3"
            ]
            .copy()
        )

    elif (
        comparison_name
        == "triple_bge_vs_qwen3_bge"
    ):

        baseline = (
            ablation[
                ablation[
                    "configuration"
                ]
                == "qwen3"
            ]
            .copy()
        )

        system = (
            ablation[
                ablation[
                    "configuration"
                ]
                == "bm25_minilm_qwen3"
            ]
            .copy()
        )

    else:

        raise ValueError(
            f"Unknown comparison: "
            f"{comparison_name}"
        )

    baseline_columns = [
        "query_id",
        "query_type",
        "query",
        *METRICS,
    ]

    system_columns = [
        "query_id",
        *METRICS,
    ]

    baseline = (
        baseline[
            baseline_columns
        ]
        .copy()
    )

    system = (
        system[
            system_columns
        ]
        .copy()
    )

    baseline = (
        baseline.rename(
            columns={
                metric:
                    f"baseline_{metric}"
                for metric
                in METRICS
            }
        )
    )

    system = (
        system.rename(
            columns={
                metric:
                    f"system_{metric}"
                for metric
                in METRICS
            }
        )
    )

    paired = (
        baseline
        .merge(
            system,
            on="query_id",
            how="inner",
            validate="one_to_one",
        )
        .sort_values(
            "query_id"
        )
        .reset_index(
            drop=True
        )
    )

    if len(
        paired
    ) != 60:

        raise RuntimeError(
            f"{comparison_name}: expected "
            f"60 paired queries, found "
            f"{len(paired)}."
        )

    return paired


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 90)
    print("BENCHMARK V1 — STATISTICAL SIGNIFICANCE")
    print("=" * 90)

    # ========================================================
    # Load data
    # ========================================================

    expanded = pd.read_csv(
        EXPANDED_RESULTS_PATH
    )

    ablation = pd.read_csv(
        RERANKER_ABLATION_PATH
    )

    # ========================================================
    # Input checks
    # ========================================================

    print()
    print("INPUT INTEGRITY")
    print("-" * 90)

    print(
        "Expanded metrics rows:",
        len(
            expanded
        ),
    )

    print(
        "Ablation metrics rows:",
        len(
            ablation
        ),
    )

    minilm_count = int(
        (
            expanded[
                "system"
            ]
            == "minilm_dense"
        )
        .sum()
    )

    qwen_bge_count = int(
        (
            ablation[
                "configuration"
            ]
            == "qwen3"
        )
        .sum()
    )

    triple_count = int(
        (
            ablation[
                "configuration"
            ]
            == "bm25_minilm_qwen3"
        )
        .sum()
    )

    print(
        "MiniLM query rows:",
        minilm_count,
    )

    print(
        "Qwen3+BGE query rows:",
        qwen_bge_count,
    )

    print(
        "Triple+BGE query rows:",
        triple_count,
    )

    if (
        minilm_count != 60
        or qwen_bge_count != 60
        or triple_count != 60
    ):

        raise RuntimeError(
            "Each target system must have "
            "exactly 60 query rows."
        )

    # ========================================================
    # Random generators
    #
    # Separate generators make different statistical procedures
    # reproducible independently.
    # ========================================================

    bootstrap_rng = (
        np.random.default_rng(
            RANDOM_SEED
        )
    )

    permutation_rng = (
        np.random.default_rng(
            RANDOM_SEED + 1
        )
    )

    # ========================================================
    # Outputs
    # ========================================================

    summary_rows = []

    bootstrap_rows = []

    paired_query_rows = []

    # ========================================================
    # Run comparisons
    # ========================================================

    for comparison_config in COMPARISONS:

        comparison_name = (
            comparison_config[
                "comparison"
            ]
        )

        baseline_label = (
            comparison_config[
                "baseline_label"
            ]
        )

        system_label = (
            comparison_config[
                "system_label"
            ]
        )

        print()
        print("=" * 90)

        print(
            comparison_name
        )

        print("=" * 90)

        paired = (
            prepare_comparison_data(
                expanded,
                ablation,
                comparison_name,
            )
        )

        # ====================================================
        # Save paired query deltas
        # ====================================================

        for metric in METRICS:

            paired[
                f"delta_{metric}"
            ] = (
                paired[
                    f"system_{metric}"
                ]
                - paired[
                    f"baseline_{metric}"
                ]
            )

        paired_output = (
            paired.copy()
        )

        paired_output.insert(
            0,
            "comparison",
            comparison_name,
        )

        paired_query_rows.append(
            paired_output
        )

        # ====================================================
        # Metrics
        # ====================================================

        for metric in METRICS:

            baseline_values = (
                paired[
                    f"baseline_{metric}"
                ]
                .to_numpy(
                    dtype=float
                )
            )

            system_values = (
                paired[
                    f"system_{metric}"
                ]
                .to_numpy(
                    dtype=float
                )
            )

            differences = (
                system_values
                - baseline_values
            )

            baseline_mean = float(
                baseline_values.mean()
            )

            system_mean = float(
                system_values.mean()
            )

            absolute_difference = (
                system_mean
                - baseline_mean
            )

            relative_improvement = (
                absolute_difference
                / baseline_mean
            )

            # ------------------------------------------------
            # Query-level wins / ties / losses
            # ------------------------------------------------

            wins = int(
                (
                    differences
                    > EPSILON
                )
                .sum()
            )

            losses = int(
                (
                    differences
                    < -EPSILON
                )
                .sum()
            )

            ties = (
                len(
                    differences
                )
                - wins
                - losses
            )

            # ------------------------------------------------
            # Paired bootstrap
            # ------------------------------------------------

            bootstrap = (
                bootstrap_paired(
                    baseline_values,
                    system_values,
                    BOOTSTRAP_ITERATIONS,
                    bootstrap_rng,
                )
            )

            difference_ci = (
                percentile_ci(
                    bootstrap[
                        "difference"
                    ],
                    CONFIDENCE_LEVEL,
                )
            )

            relative_ci = (
                percentile_ci(
                    bootstrap[
                        "relative_improvement"
                    ],
                    CONFIDENCE_LEVEL,
                )
            )

            probability_positive = float(
                (
                    bootstrap[
                        "difference"
                    ]
                    > 0
                )
                .mean()
            )

            # ------------------------------------------------
            # Paired sign-flip randomization test
            # ------------------------------------------------

            p_value = (
                paired_sign_flip_test(
                    baseline_values,
                    system_values,
                    PERMUTATION_ITERATIONS,
                    permutation_rng,
                )
            )

            # ------------------------------------------------
            # Interpretation
            # ------------------------------------------------

            if (
                difference_ci[0]
                > 0
            ):

                ci_interpretation = (
                    "positive"
                )

            elif (
                difference_ci[1]
                < 0
            ):

                ci_interpretation = (
                    "negative"
                )

            else:

                ci_interpretation = (
                    "includes_zero"
                )

            summary_rows.append({
                "comparison":
                    comparison_name,

                "baseline":
                    baseline_label,

                "system":
                    system_label,

                "metric":
                    metric,

                "queries":
                    len(
                        paired
                    ),

                "baseline_mean":
                    baseline_mean,

                "system_mean":
                    system_mean,

                "absolute_difference":
                    absolute_difference,

                "relative_improvement":
                    relative_improvement,

                "ci_level":
                    CONFIDENCE_LEVEL,

                "difference_ci_lower":
                    difference_ci[0],

                "difference_ci_upper":
                    difference_ci[1],

                "relative_ci_lower":
                    relative_ci[0],

                "relative_ci_upper":
                    relative_ci[1],

                "bootstrap_probability_positive":
                    probability_positive,

                "randomization_p_value":
                    p_value,

                "wins":
                    wins,

                "ties":
                    ties,

                "losses":
                    losses,

                "ci_interpretation":
                    ci_interpretation,
            })

            # ------------------------------------------------
            # Save complete bootstrap distribution
            # ------------------------------------------------

            for iteration in range(
                BOOTSTRAP_ITERATIONS
            ):

                bootstrap_rows.append({
                    "comparison":
                        comparison_name,

                    "metric":
                        metric,

                    "iteration":
                        iteration + 1,

                    "baseline_mean":
                        bootstrap[
                            "baseline_mean"
                        ][
                            iteration
                        ],

                    "system_mean":
                        bootstrap[
                            "system_mean"
                        ][
                            iteration
                        ],

                    "difference":
                        bootstrap[
                            "difference"
                        ][
                            iteration
                        ],

                    "relative_improvement":
                        bootstrap[
                            "relative_improvement"
                        ][
                            iteration
                        ],
                })

            # ------------------------------------------------
            # Console
            # ------------------------------------------------

            print()
            print(
                metric
            )

            print(
                "  baseline mean:",
                f"{baseline_mean:.4f}",
            )

            print(
                "  system mean:",
                f"{system_mean:.4f}",
            )

            print(
                "  absolute delta:",
                f"{absolute_difference:+.4f}",
            )

            print(
                "  relative improvement:",
                f"{relative_improvement * 100:+.2f}%",
            )

            print(
                "  95% CI absolute:",
                (
                    f"[{difference_ci[0]:+.4f}, "
                    f"{difference_ci[1]:+.4f}]"
                ),
            )

            print(
                "  95% CI relative:",
                (
                    f"[{relative_ci[0] * 100:+.2f}%, "
                    f"{relative_ci[1] * 100:+.2f}%]"
                ),
            )

            print(
                "  bootstrap P(delta > 0):",
                f"{probability_positive:.4f}",
            )

            print(
                "  randomization p-value:",
                f"{p_value:.6f}",
            )

            print(
                "  wins / ties / losses:",
                f"{wins} / {ties} / {losses}",
            )

    # ========================================================
    # Build output dataframes
    # ========================================================

    summary = pd.DataFrame(
        summary_rows
    )

    bootstrap_df = pd.DataFrame(
        bootstrap_rows
    )

    paired_queries = pd.concat(
        paired_query_rows,
        ignore_index=True,
    )

    # ========================================================
    # Save
    # ========================================================

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    bootstrap_df.to_csv(
        BOOTSTRAP_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    paired_queries.to_csv(
        QUERY_DELTA_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # JSON report
    # ========================================================

    report = {
        "queries":
            60,

        "bootstrap_iterations":
            BOOTSTRAP_ITERATIONS,

        "randomization_iterations":
            PERMUTATION_ITERATIONS,

        "confidence_level":
            CONFIDENCE_LEVEL,

        "random_seed":
            RANDOM_SEED,

        "comparisons":
            [],
    }

    for comparison_name in [
        config[
            "comparison"
        ]
        for config
        in COMPARISONS
    ]:

        comparison_rows = (
            summary[
                summary[
                    "comparison"
                ]
                == comparison_name
            ]
        )

        comparison_report = {
            "comparison":
                comparison_name,

            "metrics":
                {},
        }

        for row in comparison_rows.itertuples(
            index=False
        ):

            comparison_report[
                "metrics"
            ][
                row.metric
            ] = {
                "baseline_mean":
                    row.baseline_mean,

                "system_mean":
                    row.system_mean,

                "absolute_difference":
                    row.absolute_difference,

                "relative_improvement":
                    row.relative_improvement,

                "difference_95ci": [
                    row.difference_ci_lower,
                    row.difference_ci_upper,
                ],

                "relative_95ci": [
                    row.relative_ci_lower,
                    row.relative_ci_upper,
                ],

                "bootstrap_probability_positive":
                    row.bootstrap_probability_positive,

                "randomization_p_value":
                    row.randomization_p_value,

                "wins":
                    int(
                        row.wins
                    ),

                "ties":
                    int(
                        row.ties
                    ),

                "losses":
                    int(
                        row.losses
                    ),
            }

        report[
            "comparisons"
        ].append(
            comparison_report
        )

    REPORT_PATH.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # Compact final conclusion
    # ========================================================

    print()
    print("=" * 90)
    print("SIGNIFICANCE SUMMARY")
    print("=" * 90)

    for comparison_name in [
        config[
            "comparison"
        ]
        for config
        in COMPARISONS
    ]:

        print()
        print(
            comparison_name
        )

        rows = (
            summary[
                summary[
                    "comparison"
                ]
                == comparison_name
            ]
        )

        for row in rows.itertuples(
            index=False
        ):

            print(
                f"  {row.metric}: "
                f"delta={row.absolute_difference:+.4f}, "
                f"95% CI="
                f"[{row.difference_ci_lower:+.4f}, "
                f"{row.difference_ci_upper:+.4f}], "
                f"p={row.randomization_p_value:.6f}"
            )

    print()
    print("=" * 90)
    print("OUTPUT FILES")
    print("=" * 90)

    print()
    print(
        SUMMARY_PATH
    )

    print(
        BOOTSTRAP_PATH
    )

    print(
        QUERY_DELTA_PATH
    )

    print(
        REPORT_PATH
    )

    print()
    print(
        "STATISTICAL ANALYSIS: PASS"
    )


if __name__ == "__main__":
    main()