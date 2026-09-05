import re
from pathlib import Path

import numpy as np
import pandas as pd

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ============================================================
# Paths
# ============================================================

CORPUS_PATH = Path(
    r"data\corpus_v1\retrieval_comments_v1.parquet"
)

QUERY_PATH = Path(
    r"evaluation\benchmark_v1_queries.csv"
)

OUTPUT_PATH = Path(
    r"evaluation\benchmark_query_validation.csv"
)

DETAIL_PATH = Path(
    r"evaluation\benchmark_query_validation_details.txt"
)


# ============================================================
# Model / retrieval settings
# ============================================================

MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

BM25_TOP_K = 20
DENSE_TOP_K = 20


# ============================================================
# Thresholds
#
# These are diagnostic thresholds, not ground-truth rules.
# ============================================================

PASS_MIN_UNION = 18
PASS_MIN_OVERLAP = 3

REVIEW_MIN_UNION = 10


# ============================================================
# Helpers
# ============================================================

def tokenize(text):
    return re.findall(
        r"\b\w+\b",
        str(text).lower(),
        flags=re.UNICODE,
    )


def classify_support(
    union_count,
    overlap_count,
    corpus_size,
):
    """
    Diagnostic classification only.

    PASS:
        retrieval systems find a reasonably sized
        candidate set and have some agreement.

    REVIEW:
        potentially usable, but manually inspect.

    WEAK:
        support looks too sparse or unstable.
    """

    # Very small corpora need slightly softer thresholds.
    if corpus_size < 120:

        if (
            union_count >= 14
            and overlap_count >= 2
        ):
            return "PASS"

        if union_count >= 8:
            return "REVIEW"

        return "WEAK"

    if (
        union_count >= PASS_MIN_UNION
        and overlap_count >= PASS_MIN_OVERLAP
    ):
        return "PASS"

    if union_count >= REVIEW_MIN_UNION:
        return "REVIEW"

    return "WEAK"


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 80)
    print("VALIDATING BENCHMARK QUERIES")
    print("=" * 80)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    corpus = pd.read_parquet(
        CORPUS_PATH
    )

    queries = pd.read_csv(
        QUERY_PATH
    )

    print(
        f"Corpus documents: {len(corpus):,}"
    )

    print(
        f"Queries: {len(queries):,}"
    )

    if len(queries) != 60:
        print(
            "WARNING: expected 60 queries, "
            f"found {len(queries)}."
        )

    # --------------------------------------------------------
    # Load dense model
    # --------------------------------------------------------

    print()
    print("Loading model:")
    print(MODEL_NAME)

    model = SentenceTransformer(
        MODEL_NAME
    )

    # --------------------------------------------------------
    # Cache per-video representations
    # --------------------------------------------------------

    video_cache = {}

    results = []

    detail_lines = []

    # ========================================================
    # Process each query
    # ========================================================

    for position, row in enumerate(
        queries.itertuples(index=False),
        start=1,
    ):

        query_id = str(row.query_id)
        video_id = str(row.video_id)
        query_type = str(row.query_type)
        query_text = str(row.query)

        print()
        print(
            f"[{position}/{len(queries)}] "
            f"{query_id}"
        )

        # ----------------------------------------------------
        # Build video cache if needed
        # ----------------------------------------------------

        if video_id not in video_cache:

            video_df = (
                corpus[
                    corpus["video_id"].astype(str)
                    == video_id
                ]
                .copy()
                .reset_index(drop=True)
            )

            if video_df.empty:
                raise RuntimeError(
                    f"No documents found for "
                    f"video {video_id}"
                )

            texts = (
                video_df["comment_text"]
                .fillna("")
                .astype(str)
                .tolist()
            )

            tokenized_corpus = [
                tokenize(text)
                for text in texts
            ]

            bm25 = BM25Okapi(
                tokenized_corpus
            )

            print(
                f"Encoding video {video_id}: "
                f"{len(video_df)} docs"
            )

            embeddings = model.encode(
                texts,
                batch_size=64,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )

            video_cache[video_id] = {
                "df": video_df,
                "texts": texts,
                "bm25": bm25,
                "embeddings": embeddings,
            }

        cache = video_cache[video_id]

        video_df = cache["df"]
        texts = cache["texts"]
        bm25 = cache["bm25"]
        doc_embeddings = cache["embeddings"]

        corpus_size = len(video_df)

        # ====================================================
        # BM25
        # ====================================================

        query_tokens = tokenize(
            query_text
        )

        bm25_scores = np.asarray(
            bm25.get_scores(
                query_tokens
            ),
            dtype=float,
        )

        bm25_k = min(
            BM25_TOP_K,
            corpus_size,
        )

        bm25_indices = np.argsort(
            -bm25_scores
        )[:bm25_k]

        # ====================================================
        # Dense
        # ====================================================

        query_embedding = model.encode(
            query_text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        dense_scores = (
            doc_embeddings
            @ query_embedding
        )

        dense_k = min(
            DENSE_TOP_K,
            corpus_size,
        )

        dense_indices = np.argsort(
            -dense_scores
        )[:dense_k]

        # ====================================================
        # Candidate-set diagnostics
        # ====================================================

        bm25_set = set(
            bm25_indices.tolist()
        )

        dense_set = set(
            dense_indices.tolist()
        )

        union_set = (
            bm25_set
            | dense_set
        )

        overlap_set = (
            bm25_set
            & dense_set
        )

        union_count = len(
            union_set
        )

        overlap_count = len(
            overlap_set
        )

        overlap_ratio = (
            overlap_count
            / BM25_TOP_K
        )

        # ====================================================
        # Additional lexical signal
        #
        # Count how many documents contain at least one
        # reasonably informative query token.
        # ====================================================

        stop_words = {
            "what",
            "do",
            "does",
            "did",
            "the",
            "a",
            "an",
            "about",
            "say",
            "viewers",
            "viewer",
            "how",
            "which",
            "or",
            "and",
            "for",
            "of",
            "to",
            "in",
            "is",
            "are",
            "with",
        }

        meaningful_tokens = [
            token
            for token in query_tokens
            if (
                len(token) >= 3
                and token not in stop_words
            )
        ]

        lexical_mask = pd.Series(
            False,
            index=video_df.index,
        )

        lowered_text = (
            video_df["comment_text"]
            .fillna("")
            .astype(str)
            .str.lower()
        )

        for token in meaningful_tokens:

            pattern = (
                r"\b"
                + re.escape(token)
                + r"\b"
            )

            lexical_mask = (
                lexical_mask
                |
                lowered_text.str.contains(
                    pattern,
                    regex=True,
                    na=False,
                )
            )

        lexical_support = int(
            lexical_mask.sum()
        )

        # ====================================================
        # Classification
        # ====================================================

        status = classify_support(
            union_count=union_count,
            overlap_count=overlap_count,
            corpus_size=corpus_size,
        )

        # ----------------------------------------------------
        # Flag suspiciously low lexical support
        # ----------------------------------------------------

        if (
            status == "PASS"
            and lexical_support < 3
        ):
            status = "REVIEW"

        if lexical_support == 0:
            status = "WEAK"

        # ====================================================
        # Save summary row
        # ====================================================

        results.append({
            "query_id": query_id,
            "video_id": video_id,
            "query_type": query_type,
            "query": query_text,
            "video_docs": corpus_size,
            "bm25_top_k": len(bm25_set),
            "dense_top_k": len(dense_set),
            "union_candidates": union_count,
            "bm25_dense_overlap": overlap_count,
            "overlap_ratio": round(
                overlap_ratio,
                4,
            ),
            "lexical_support_docs": lexical_support,
            "validation_status": status,
        })

        # ====================================================
        # Detail report
        # ====================================================

        detail_lines.append(
            "=" * 100
        )

        detail_lines.append(
            f"{query_id} | "
            f"{status}"
        )

        detail_lines.append(
            f"video_id: {video_id}"
        )

        detail_lines.append(
            f"query_type: {query_type}"
        )

        detail_lines.append(
            f"query: {query_text}"
        )

        detail_lines.append(
            f"video_docs: {corpus_size}"
        )

        detail_lines.append(
            f"BM25/Dense overlap: "
            f"{overlap_count}"
        )

        detail_lines.append(
            f"Union candidates: "
            f"{union_count}"
        )

        detail_lines.append(
            f"Lexical support docs: "
            f"{lexical_support}"
        )

        detail_lines.append("")

        # ----------------------------------------------------
        # Top 8 BM25
        # ----------------------------------------------------

        detail_lines.append(
            "TOP BM25:"
        )

        for rank, idx in enumerate(
            bm25_indices[:8],
            start=1,
        ):

            text = texts[idx].replace(
                "\n",
                " ",
            )

            detail_lines.append(
                f"[B{rank}] "
                f"{text[:500]}"
            )

        detail_lines.append("")

        # ----------------------------------------------------
        # Top 8 Dense
        # ----------------------------------------------------

        detail_lines.append(
            "TOP DENSE:"
        )

        for rank, idx in enumerate(
            dense_indices[:8],
            start=1,
        ):

            text = texts[idx].replace(
                "\n",
                " ",
            )

            detail_lines.append(
                f"[D{rank}] "
                f"{text[:500]}"
            )

        detail_lines.append("")

        print(
            f"status={status} | "
            f"union={union_count} | "
            f"overlap={overlap_count} | "
            f"lexical={lexical_support}"
        )

    # ========================================================
    # Save summary CSV
    # ========================================================

    result_df = pd.DataFrame(
        results
    )

    result_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Save detailed inspection report
    # ========================================================

    DETAIL_PATH.write_text(
        "\n".join(
            detail_lines
        ),
        encoding="utf-8",
    )

    # ========================================================
    # Final summary
    # ========================================================

    print()
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    status_counts = (
        result_df[
            "validation_status"
        ]
        .value_counts()
    )

    for status in [
        "PASS",
        "REVIEW",
        "WEAK",
    ]:

        count = int(
            status_counts.get(
                status,
                0,
            )
        )

        print(
            f"{status}: {count}"
        )

    print()
    print("Queries requiring attention:")

    attention = result_df[
        result_df[
            "validation_status"
        ] != "PASS"
    ]

    if attention.empty:

        print(
            "None"
        )

    else:

        for row in attention.itertuples(
            index=False
        ):

            print(
                f"{row.query_id}: "
                f"{row.validation_status} | "
                f"lexical="
                f"{row.lexical_support_docs} | "
                f"overlap="
                f"{row.bm25_dense_overlap}"
            )

    print()
    print("Summary CSV:")
    print(
        OUTPUT_PATH
    )

    print()
    print("Detailed report:")
    print(
        DETAIL_PATH
    )


if __name__ == "__main__":
    main()