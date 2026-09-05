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

OUTPUT_DIR = Path(
    r"evaluation\candidate_pool_v1"
)

RAW_POOL_PATH = (
    OUTPUT_DIR
    / "candidate_pool_benchmark_v1.csv"
)

UNLABELED_PATH = (
    OUTPUT_DIR
    / "candidate_pool_benchmark_v1_unlabeled.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "candidate_pool_benchmark_v1_summary.csv"
)


# ============================================================
# Retrieval settings
# ============================================================

MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

BM25_TOP_K = 25
DENSE_TOP_K = 25
LEXICAL_TOP_K = 25

RANDOM_SAMPLES = 10

RANDOM_SEED = 42


# ============================================================
# Generic lexical stop words
#
# These words are too generic to be useful for candidate
# discovery.
#
# Important:
# lexical candidate discovery is NOT a benchmark retriever.
# It only broadens the human/LLM judgment pool.
# ============================================================

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "about",
    "be",
    "been",
    "being",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "i",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "say",
    "says",
    "saying",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "would",

    # Benchmark-specific generic terms
    "viewer",
    "viewers",
    "comment",
    "comments",
    "video",
    "videos",
}


# ============================================================
# Helpers
# ============================================================

def tokenize(text):
    """
    Simple Unicode-aware tokenization.

    This is deliberately lightweight because BM25 here is a
    baseline lexical retriever rather than a highly tuned
    production implementation.
    """

    return re.findall(
        r"\b\w+\b",
        str(text).lower(),
        flags=re.UNICODE,
    )


def get_meaningful_query_tokens(text):
    """
    Extract lexical terms that are informative enough for
    generic candidate-pool expansion.

    Example:

    Query:
        What hardware specifications do viewers consider
        important for programming or machine learning?

    Could yield:
        hardware
        specifications
        important
        programming
        machine
        learning
    """

    tokens = tokenize(
        text
    )

    meaningful = []

    seen = set()

    for token in tokens:

        if token in STOP_WORDS:
            continue

        # Keep short but meaningful technical tokens.
        if (
            len(token) < 3
            and token
            not in {
                "ai",
                "ml",
                "pc",
                "vr",
            }
        ):
            continue

        if token in seen:
            continue

        seen.add(
            token
        )

        meaningful.append(
            token
        )

    return meaningful


def find_matched_query_tokens(
    text,
    query_tokens,
):
    """
    Return meaningful query tokens found as whole words in
    the document.

    Whole-word matching prevents problems such as:
        mac -> machine
        ram -> programming
    """

    text = str(
        text
    ).lower()

    matched = []

    for token in query_tokens:

        pattern = (
            r"\b"
            + re.escape(token)
            + r"\b"
        )

        if re.search(
            pattern,
            text,
            flags=re.UNICODE,
        ):
            matched.append(
                token
            )

    return matched


def add_rank_column(
    dataframe,
    indices,
    column_name,
):
    """
    Add Top-K rank.

    Rank begins at 1.

    Documents outside Top-K remain NaN.
    """

    dataframe[
        column_name
    ] = np.nan

    for rank, idx in enumerate(
        indices,
        start=1,
    ):

        dataframe.loc[
            idx,
            column_name,
        ] = rank


def query_number(
    query_id,
):
    """
    q001 -> 1
    q060 -> 60

    Used only for deterministic random seeds.
    """

    digits = re.sub(
        r"\D",
        "",
        str(query_id),
    )

    if not digits:
        return 0

    return int(
        digits
    )


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("BUILDING BENCHMARK V1 CANDIDATE POOL")
    print("=" * 80)

    # ========================================================
    # Load corpus
    # ========================================================

    print()
    print("Loading retrieval corpus...")

    corpus = pd.read_parquet(
        CORPUS_PATH
    )

    base_required_columns = {
        "video_id",
        "representative_comment_id",
        "comment_text",
    }

    missing_columns = (
        base_required_columns
        - set(
            corpus.columns
        )
    )

    if missing_columns:

        raise RuntimeError(
            "Corpus missing columns: "
            f"{missing_columns}"
        )

    corpus[
        "video_id"
    ] = (
        corpus[
            "video_id"
        ]
        .astype(str)
    )

    corpus[
        "representative_comment_id"
    ] = (
        corpus[
            "representative_comment_id"
        ]
        .astype(str)
    )

    # ========================================================
    # Stable retrieval document ID
    # ========================================================

    corpus[
        "document_id"
    ] = (
        corpus[
            "video_id"
        ]
        + "::"
        + corpus[
            "representative_comment_id"
        ]
    )

    if (
        corpus[
            "document_id"
        ]
        .duplicated()
        .any()
    ):

        duplicates = (
            corpus[
                corpus[
                    "document_id"
                ]
                .duplicated(
                    keep=False
                )
            ]
        )

        raise RuntimeError(
            "Duplicate document_id detected. "
            "Example: "
            f"{duplicates['document_id'].iloc[0]}"
        )

    print(
        "Corpus documents:",
        f"{len(corpus):,}",
    )

    print(
        "Unique document IDs:",
        f"{corpus['document_id'].nunique():,}",
    )

    # ========================================================
    # Load benchmark queries
    # ========================================================

    print()
    print("Loading benchmark queries...")

    queries = pd.read_csv(
        QUERY_PATH
    )

    query_required_columns = {
        "query_id",
        "video_id",
        "query_type",
        "query",
    }

    missing_query_columns = (
        query_required_columns
        - set(
            queries.columns
        )
    )

    if missing_query_columns:

        raise RuntimeError(
            "Query file missing columns: "
            f"{missing_query_columns}"
        )

    queries[
        "query_id"
    ] = (
        queries[
            "query_id"
        ]
        .astype(str)
    )

    queries[
        "video_id"
    ] = (
        queries[
            "video_id"
        ]
        .astype(str)
    )

    if (
        queries[
            "query_id"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate query_id detected."
        )

    print(
        "Queries:",
        len(
            queries
        ),
    )

    print(
        "Benchmark videos:",
        queries[
            "video_id"
        ].nunique(),
    )

    if len(
        queries
    ) != 60:

        print(
            "WARNING: Benchmark V1 "
            "is expected to contain 60 queries."
        )

    if (
        queries[
            "video_id"
        ]
        .nunique()
        != 20
    ):

        print(
            "WARNING: Benchmark V1 "
            "is expected to contain 20 videos."
        )

    # ========================================================
    # Load old dense baseline model
    # ========================================================

    print()
    print("Loading dense baseline model:")

    print(
        MODEL_NAME
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    # ========================================================
    # Cache
    #
    # Each benchmark video has 3 queries.
    # Encode each video's documents only once.
    # ========================================================

    video_cache = {}

    all_pools = []

    summary_rows = []

    # ========================================================
    # Process queries
    # ========================================================

    for position, query_row in enumerate(
        queries.itertuples(
            index=False
        ),
        start=1,
    ):

        query_id = str(
            query_row.query_id
        )

        video_id = str(
            query_row.video_id
        )

        query_type = str(
            query_row.query_type
        )

        query_text = str(
            query_row.query
        )

        print()
        print("=" * 80)

        print(
            f"[{position}/{len(queries)}] "
            f"{query_id}"
        )

        print(
            query_text
        )

        print("=" * 80)

        # ====================================================
        # Build video cache
        # ====================================================

        if (
            video_id
            not in video_cache
        ):

            video_df = (
                corpus[
                    corpus[
                        "video_id"
                    ]
                    == video_id
                ]
                .copy()
                .reset_index(
                    drop=True
                )
            )

            if video_df.empty:

                raise RuntimeError(
                    "No corpus documents "
                    f"for video {video_id}"
                )

            texts = (
                video_df[
                    "comment_text"
                ]
                .fillna("")
                .astype(str)
                .tolist()
            )

            # -----------------------------------------------
            # BM25
            # -----------------------------------------------

            tokenized_corpus = [
                tokenize(
                    text
                )
                for text in texts
            ]

            bm25 = BM25Okapi(
                tokenized_corpus
            )

            # -----------------------------------------------
            # Dense embeddings
            # -----------------------------------------------

            print(
                "Encoding video:",
                video_id,
            )

            print(
                "Documents:",
                len(
                    video_df
                ),
            )

            embeddings = model.encode(
                texts,
                batch_size=64,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )

            video_cache[
                video_id
            ] = {
                "df": video_df,
                "texts": texts,
                "bm25": bm25,
                "embeddings": embeddings,
            }

        cache = (
            video_cache[
                video_id
            ]
        )

        video_df = (
            cache[
                "df"
            ]
            .copy()
        )

        texts = (
            cache[
                "texts"
            ]
        )

        bm25 = (
            cache[
                "bm25"
            ]
        )

        doc_embeddings = (
            cache[
                "embeddings"
            ]
        )

        corpus_size = len(
            video_df
        )

        print(
            "Video documents:",
            corpus_size,
        )

        # ====================================================
        # 1. BM25 retrieval
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

        bm25_indices = (
            np.argsort(
                -bm25_scores
            )[
                :bm25_k
            ]
        )

        video_df[
            "bm25_score"
        ] = bm25_scores

        add_rank_column(
            video_df,
            bm25_indices,
            "bm25_rank",
        )

        # ====================================================
        # 2. MiniLM dense baseline
        # ====================================================

        query_embedding = (
            model.encode(
                query_text,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        )

        # Both query and documents are normalized.
        #
        # Therefore:
        # dot product == cosine similarity
        dense_scores = (
            doc_embeddings
            @ query_embedding
        )

        dense_k = min(
            DENSE_TOP_K,
            corpus_size,
        )

        dense_indices = (
            np.argsort(
                -dense_scores
            )[
                :dense_k
            ]
        )

        video_df[
            "dense_score"
        ] = dense_scores

        add_rank_column(
            video_df,
            dense_indices,
            "dense_rank",
        )

        # ====================================================
        # 3. Generic lexical candidate discovery
        #
        # This is NOT evaluated as a retriever.
        #
        # It exists only to broaden relevance judgment pools.
        # ====================================================

        meaningful_tokens = (
            get_meaningful_query_tokens(
                query_text
            )
        )

        video_df[
            "lexical_terms"
        ] = (
            video_df[
                "comment_text"
            ]
            .fillna("")
            .astype(str)
            .apply(
                lambda text:
                find_matched_query_tokens(
                    text,
                    meaningful_tokens,
                )
            )
        )

        video_df[
            "lexical_match_count"
        ] = (
            video_df[
                "lexical_terms"
            ]
            .apply(
                len
            )
        )

        lexical_candidates = (
            video_df[
                video_df[
                    "lexical_match_count"
                ]
                > 0
            ]
            .copy()
        )

        # More distinct matched query concepts first.
        #
        # Dense score serves only as a deterministic
        # secondary tie-breaker.
        lexical_candidates = (
            lexical_candidates
            .sort_values(
                by=[
                    "lexical_match_count",
                    "dense_score",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
        )

        lexical_indices = (
            lexical_candidates
            .head(
                LEXICAL_TOP_K
            )
            .index
            .tolist()
        )

        video_df[
            "lexical_selected"
        ] = False

        if lexical_indices:

            video_df.loc[
                lexical_indices,
                "lexical_selected",
            ] = True

        # ====================================================
        # 4. Union of retrieval/discovery systems
        # ====================================================

        selected_indices = set(
            bm25_indices.tolist()
        )

        selected_indices.update(
            dense_indices.tolist()
        )

        selected_indices.update(
            lexical_indices
        )

        # ====================================================
        # 5. Deterministic random sample
        #
        # Important:
        # These are NOT assumed to be negatives.
        #
        # Some may be relevant.
        # ====================================================

        remaining_indices = [
            idx
            for idx
            in video_df.index
            if idx
            not in selected_indices
        ]

        seed = (
            RANDOM_SEED
            + query_number(
                query_id
            )
        )

        rng = (
            np.random.default_rng(
                seed
            )
        )

        random_count = min(
            RANDOM_SAMPLES,
            len(
                remaining_indices
            ),
        )

        if random_count > 0:

            random_indices = (
                rng.choice(
                    remaining_indices,
                    size=random_count,
                    replace=False,
                )
                .tolist()
            )

        else:

            random_indices = []

        selected_indices.update(
            random_indices
        )

        video_df[
            "random_sample"
        ] = False

        if random_indices:

            video_df.loc[
                random_indices,
                "random_sample",
            ] = True

        # ====================================================
        # 6. Build query candidate pool
        # ====================================================

        pool = (
            video_df.loc[
                sorted(
                    selected_indices
                )
            ]
            .copy()
        )

        pool[
            "query_id"
        ] = query_id

        pool[
            "query_type"
        ] = query_type

        pool[
            "query"
        ] = query_text

        # ----------------------------------------------------
        # Record candidate provenance
        # ----------------------------------------------------

        def get_candidate_source(
            row,
        ):

            sources = []

            if pd.notna(
                row[
                    "bm25_rank"
                ]
            ):

                sources.append(
                    "bm25"
                )

            if pd.notna(
                row[
                    "dense_rank"
                ]
            ):

                sources.append(
                    "dense"
                )

            if bool(
                row[
                    "lexical_selected"
                ]
            ):

                sources.append(
                    "lexical"
                )

            if bool(
                row[
                    "random_sample"
                ]
            ):

                sources.append(
                    "random"
                )

            return "|".join(
                sources
            )

        pool[
            "candidate_source"
        ] = (
            pool.apply(
                get_candidate_source,
                axis=1,
            )
        )

        # ====================================================
        # Diagnostics
        # ====================================================

        bm25_set = set(
            bm25_indices.tolist()
        )

        dense_set = set(
            dense_indices.tolist()
        )

        lexical_set = set(
            lexical_indices
        )

        bm25_dense_overlap = len(
            bm25_set
            & dense_set
        )

        bm25_lexical_overlap = len(
            bm25_set
            & lexical_set
        )

        dense_lexical_overlap = len(
            dense_set
            & lexical_set
        )

        print(
            f"BM25 top-{BM25_TOP_K}:",
            len(
                bm25_set
            ),
        )

        print(
            f"Dense top-{DENSE_TOP_K}:",
            len(
                dense_set
            ),
        )

        print(
            "Lexical selected:",
            len(
                lexical_set
            ),
        )

        print(
            "Random samples:",
            len(
                random_indices
            ),
        )

        print(
            "Unique pooled candidates:",
            len(
                pool
            ),
        )

        print(
            "BM25 / Dense overlap:",
            bm25_dense_overlap,
        )

        # ====================================================
        # Summary
        # ====================================================

        summary_rows.append({
            "query_id":
                query_id,

            "video_id":
                video_id,

            "query_type":
                query_type,

            "query":
                query_text,

            "video_docs":
                corpus_size,

            "bm25_candidates":
                len(
                    bm25_set
                ),

            "dense_candidates":
                len(
                    dense_set
                ),

            "lexical_candidates":
                len(
                    lexical_set
                ),

            "random_samples":
                len(
                    random_indices
                ),

            "unique_pool_size":
                len(
                    pool
                ),

            "bm25_dense_overlap":
                bm25_dense_overlap,

            "bm25_lexical_overlap":
                bm25_lexical_overlap,

            "dense_lexical_overlap":
                dense_lexical_overlap,

            "meaningful_query_tokens":
                "|".join(
                    meaningful_tokens
                ),
        })

        all_pools.append(
            pool
        )

    # ========================================================
    # Combine all query pools
    # ========================================================

    if not all_pools:

        raise RuntimeError(
            "No candidate pools generated."
        )

    full_pool = pd.concat(
        all_pools,
        ignore_index=True,
    )

    # Convert token lists to CSV-friendly strings.
    full_pool[
        "lexical_terms"
    ] = (
        full_pool[
            "lexical_terms"
        ]
        .apply(
            lambda values:
            "|".join(
                values
            )
            if isinstance(
                values,
                list,
            )
            else ""
        )
    )

    # ========================================================
    # Save internal raw pool
    #
    # Contains ranking / system provenance.
    #
    # Do NOT send this version to Work for blind judging.
    # ========================================================

    raw_columns = [
        "query_id",
        "video_id",
        "query_type",
        "query",

        "document_id",
        "representative_comment_id",

        "comment_text",

        "candidate_source",

        "bm25_rank",
        "bm25_score",

        "dense_rank",
        "dense_score",

        "lexical_match_count",
        "lexical_terms",

        "random_sample",
    ]

    optional_columns = [
        column
        for column
        in [
            "occurrence_count",
            "unique_author_count",
            "total_like_count",
            "max_like_count",
            "first_published_at",
            "last_published_at",
        ]
        if column
        in full_pool.columns
    ]

    raw_columns.extend(
        optional_columns
    )

    full_pool[
        raw_columns
    ].to_csv(
        RAW_POOL_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Build fully blinded Work-labeling file
    #
    # No:
    # - scores
    # - ranks
    # - candidate provenance
    # - likes
    # - occurrence counts
    # ========================================================

    labeling_parts = []

    for query_id, group in (
        full_pool.groupby(
            "query_id",
            sort=False,
        )
    ):

        shuffled = (
            group.sample(
                frac=1,
                random_state=(
                    RANDOM_SEED
                    + query_number(
                        query_id
                    )
                ),
            )
            .reset_index(
                drop=True
            )
        )

        shuffled[
            "label_id"
        ] = [
            (
                f"{query_id}_"
                f"{i:03d}"
            )
            for i in range(
                1,
                len(
                    shuffled
                ) + 1,
            )
        ]

        labeling_parts.append(
            shuffled
        )

    unlabeled = pd.concat(
        labeling_parts,
        ignore_index=True,
    )

    unlabeled_columns = [
        "label_id",

        "query_id",
        "video_id",
        "query_type",
        "query",

        "document_id",
        "representative_comment_id",

        "comment_text",
    ]

    unlabeled[
        unlabeled_columns
    ].to_csv(
        UNLABELED_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Save query-level pooling summary
    # ========================================================

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Integrity checks
    # ========================================================

    duplicated_pairs = (
        full_pool[
            [
                "query_id",
                "document_id",
            ]
        ]
        .duplicated()
        .sum()
    )

    if duplicated_pairs != 0:

        raise RuntimeError(
            "Duplicate query-document pairs "
            f"detected: {duplicated_pairs}"
        )

    if (
        unlabeled[
            "label_id"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate label_id detected."
        )

    # ========================================================
    # Final summary
    # ========================================================

    print()
    print("=" * 80)
    print("BENCHMARK V1 POOL SUMMARY")
    print("=" * 80)

    print(
        "Queries:",
        summary_df[
            "query_id"
        ].nunique(),
    )

    print(
        "Videos:",
        summary_df[
            "video_id"
        ].nunique(),
    )

    print(
        "Total query-document judgments:",
        f"{len(full_pool):,}",
    )

    print(
        "Unique retrieval documents represented:",
        f"{full_pool['document_id'].nunique():,}",
    )

    print(
        "Average pool size/query:",
        round(
            summary_df[
                "unique_pool_size"
            ].mean(),
            2,
        ),
    )

    print(
        "Minimum pool size:",
        int(
            summary_df[
                "unique_pool_size"
            ].min()
        ),
    )

    print(
        "Maximum pool size:",
        int(
            summary_df[
                "unique_pool_size"
            ].max()
        ),
    )

    print()
    print("Internal raw pool:")
    print(
        RAW_POOL_PATH
    )

    print()
    print("Blinded Work-labeling file:")
    print(
        UNLABELED_PATH
    )

    print()
    print("Pooling summary:")
    print(
        SUMMARY_PATH
    )


if __name__ == "__main__":
    main()