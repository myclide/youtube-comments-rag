from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    r"evaluation\candidate_pool_v1"
    r"\candidate_pool_benchmark_v1_unlabeled.csv"
)

OUTPUT_DIR = Path(
    r"evaluation\candidate_pool_v1"
    r"\work_batches"
)

QUERIES_PER_BATCH = 10


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        INPUT_PATH,
        keep_default_na=False,
        dtype=str,
    )

    required = {
        "label_id",
        "query_id",
        "video_id",
        "query_type",
        "query",
        "document_id",
        "representative_comment_id",
        "comment_text",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing columns: {missing}"
        )

    query_ids = (
        df["query_id"]
        .drop_duplicates()
        .tolist()
    )

    print("=" * 80)
    print("SPLITTING WORK LABELING BATCHES")
    print("=" * 80)

    print(
        f"Total rows: {len(df):,}"
    )

    print(
        f"Total queries: {len(query_ids)}"
    )

    batch_number = 0
    total_written = 0

    for start in range(
        0,
        len(query_ids),
        QUERIES_PER_BATCH,
    ):

        batch_number += 1

        batch_query_ids = query_ids[
            start:
            start + QUERIES_PER_BATCH
        ]

        batch_df = (
            df[
                df["query_id"].isin(
                    batch_query_ids
                )
            ]
            .copy()
        )

        first_query = batch_query_ids[0]
        last_query = batch_query_ids[-1]

        output_path = (
            OUTPUT_DIR
            / (
                f"benchmark_v1_batch_"
                f"{batch_number:02d}_"
                f"{first_query}_to_{last_query}.csv"
            )
        )

        batch_df.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
        )

        total_written += len(
            batch_df
        )

        print()
        print(
            f"Batch {batch_number:02d}"
        )

        print(
            f"Queries: "
            f"{first_query} -> {last_query}"
        )

        print(
            f"Rows: {len(batch_df)}"
        )

        print(
            f"Saved: {output_path}"
        )

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(
        f"Batches: {batch_number}"
    )

    print(
        f"Rows written: "
        f"{total_written:,}"
    )

    if total_written != len(df):
        raise RuntimeError(
            "Row-count mismatch after splitting."
        )

    print(
        "Integrity check: PASS"
    )


if __name__ == "__main__":
    main()