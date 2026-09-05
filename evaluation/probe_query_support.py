import re

import pandas as pd


CORPUS_PATH = (
    r"data\corpus_v1\retrieval_comments_v1.parquet"
)

VIDEO_ID = "dxPhJ0wfp0g"


# ------------------------------------------------------------
# These are inspection probes only.
# They are NOT relevance labels and NOT benchmark retrievers.
#
# Each entry:
#     display label -> regex
#
# Word boundaries prevent:
#     "mac" matching "machine"
#     "ram" matching unrelated longer words
# ------------------------------------------------------------

QUERY_PROBES = {

    "q001": {
        "recommend": (
            r"\brecommend"
            r"(?:s|ed|ing|ation|ations)?\b"
        ),
        "laptop": r"\blaptops?\b",
        "linux": r"\blinux\b",
        "thinkpad": r"\bthinkpads?\b",
        "dell": r"\bdell\b",
        "xps": r"\bxps\b",
        "pc": r"\bpc\b",
        "azure": r"\bazure\b",
    },

    "q002": {
        "processor": r"\bprocessors?\b",
        "cpu": r"\bcpus?\b",
        "gpu": r"\bgpus?\b",
        "graphics": r"\bgraphics?\b",
        "ryzen": r"\bryzen\b",
        "intel": r"\bintel\b",
        "i3": r"\bi3\b",
        "i5": r"\bi5\b",
        "i7": r"\bi7\b",
        "ram": r"\bram\b",
        "ssd": r"\bssd\b",
    },

    "q003": {
        "macbook": (
            r"\bmac\s*book"
            r"(?:\s+air|\s+pro)?\b"
        ),
        "mac": r"\bmac\b",
    },
}


def get_matched_terms(
    text,
    patterns,
):
    matched = []

    for label, pattern in patterns.items():

        if re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):
            matched.append(label)

    return matched


def main():

    df = pd.read_parquet(
        CORPUS_PATH
    )

    df = (
        df[
            df["video_id"] == VIDEO_ID
        ]
        .copy()
        .reset_index(drop=True)
    )

    if df.empty:
        raise RuntimeError(
            f"No documents found for video "
            f"{VIDEO_ID}"
        )

    print("=" * 80)
    print("QUERY SUPPORT PROBE")
    print("=" * 80)

    print(
        f"Documents in video: {len(df)}"
    )

    for query_id, patterns in (
        QUERY_PROBES.items()
    ):

        matched_terms = (
            df["comment_text"]
            .fillna("")
            .astype(str)
            .apply(
                lambda x: get_matched_terms(
                    x,
                    patterns,
                )
            )
        )

        mask = matched_terms.apply(
            lambda x: len(x) > 0
        )

        matches = (
            df[mask]
            .copy()
        )

        matches["matched_terms"] = (
            matched_terms[mask]
        )

        print()
        print("=" * 80)
        print(query_id)
        print("=" * 80)

        print(
            "Keyword-assisted candidates:",
            len(matches),
        )

        print()

        print("MATCH COUNTS BY TERM")

        for label in patterns:

            count = matches[
                "matched_terms"
            ].apply(
                lambda terms:
                label in terms
            ).sum()

            print(
                f"{label:15s}: {count}"
            )

        if matches.empty:
            continue

        sample = matches.sample(
            n=min(
                20,
                len(matches),
            ),
            random_state=42,
        )

        print()
        print("SAMPLE")
        print("-" * 80)

        for i, row in enumerate(
            sample.itertuples(),
            start=1,
        ):

            print()

            print(
                f"[{i}] "
                f"matched="
                f"{row.matched_terms} | "
                f"occurrences="
                f"{row.occurrence_count}"
            )

            print(
                repr(
                    row.comment_text
                )
            )


if __name__ == "__main__":
    main()