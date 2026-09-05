from pathlib import Path

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

TOP_K = 25

BATCH_SIZE = 16

# 512 is plenty for this comment-retrieval benchmark and keeps
# CPU inference practical.
MAX_SEQ_LENGTH = 512


# ============================================================
# Paths
# ============================================================

CORPUS_PATH = Path(
    r"data\corpus_v1"
    r"\retrieval_comments_v1.parquet"
)

QUERY_PATH = Path(
    r"evaluation\benchmark_v1_queries.csv"
)

QRELS_PATH = Path(
    r"evaluation\candidate_pool_v1"
    r"\benchmark_v1_qrels.csv"
)

OUTPUT_DIR = Path(
    r"evaluation\qwen3_dense_v1"
)

TOP25_PATH = (
    OUTPUT_DIR
    / "qwen3_dense_top25.csv"
)

NOVEL_PATH = (
    OUTPUT_DIR
    / "qwen3_dense_novel_candidates_unlabeled.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "qwen3_dense_pool_summary.csv"
)


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 90)
    print("QWEN3 EMBEDDING — BENCHMARK V1 POOL EXPANSION")
    print("=" * 90)

    # ========================================================
    # Load corpus
    # ========================================================

    print()
    print("Loading corpus...")

    corpus = pd.read_parquet(
        CORPUS_PATH
    )

    corpus["video_id"] = (
        corpus["video_id"]
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

    corpus["document_id"] = (
        corpus["video_id"]
        + "::"
        + corpus[
            "representative_comment_id"
        ]
    )

    if corpus["document_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate corpus document_id."
        )

    # ========================================================
    # Load queries
    # ========================================================

    queries = pd.read_csv(
        QUERY_PATH,
        dtype=str,
        keep_default_na=False,
    )

    if len(queries) != 60:
        raise RuntimeError(
            f"Expected 60 queries, found {len(queries)}."
        )

    # ========================================================
    # Load existing frozen qrels
    # ========================================================

    qrels = pd.read_csv(
        QRELS_PATH,
        dtype={
            "query_id": str,
            "document_id": str,
        },
    )

    existing_pairs = set(
        zip(
            qrels["query_id"],
            qrels["document_id"],
        )
    )

    print(
        "Existing qrel pairs:",
        f"{len(existing_pairs):,}",
    )

    # ========================================================
    # Load modern embedding model
    # ========================================================

    print()
    print("Loading model:")
    print(MODEL_NAME)

    model = SentenceTransformer(
        MODEL_NAME
    )

    model.max_seq_length = (
        MAX_SEQ_LENGTH
    )

    # ========================================================
    # Video embedding cache
    #
    # Each benchmark video has 3 queries.
    # Documents only need encoding once.
    # ========================================================

    video_cache = {}

    result_rows = []
    summary_rows = []

    # ========================================================
    # Retrieve
    # ========================================================

    for position, row in enumerate(
        queries.itertuples(
            index=False
        ),
        start=1,
    ):

        query_id = str(
            row.query_id
        )

        video_id = str(
            row.video_id
        )

        query_type = str(
            row.query_type
        )

        query_text = str(
            row.query
        )

        print()
        print(
            f"[{position:02d}/60] "
            f"{query_id} "
            f"video={video_id}"
        )

        # ====================================================
        # Encode this video's documents once
        # ====================================================

        if video_id not in video_cache:

            video_df = (
                corpus[
                    corpus["video_id"]
                    == video_id
                ]
                .copy()
                .reset_index(
                    drop=True
                )
            )

            if video_df.empty:
                raise RuntimeError(
                    f"No documents for video {video_id}."
                )

            texts = (
                video_df["comment_text"]
                .fillna("")
                .astype(str)
                .tolist()
            )

            print(
                "  Encoding documents:",
                len(texts),
            )

            doc_embeddings = (
                model.encode(
                    texts,
                    batch_size=BATCH_SIZE,
                    show_progress_bar=True,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
            )

            video_cache[video_id] = {
                "df":
                    video_df,

                "embeddings":
                    doc_embeddings,
            }

        cached = (
            video_cache[video_id]
        )

        video_df = (
            cached["df"]
            .copy()
        )

        doc_embeddings = (
            cached["embeddings"]
        )

        # ====================================================
        # Encode query
        #
        # Qwen's official SentenceTransformers usage recommends
        # the model's built-in "query" prompt.
        # ====================================================

        query_embedding = model.encode(
            query_text,
            prompt_name="query",
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        # ====================================================
        # Cosine similarity
        #
        # Because both sides are L2 normalized:
        #
        # dot product == cosine similarity
        # ====================================================

        scores = (
            doc_embeddings
            @ query_embedding
        )

        top_k = min(
            TOP_K,
            len(video_df),
        )

        top_indices = (
            np.argsort(
                -scores
            )[:top_k]
        )

        top_df = (
            video_df.loc[
                top_indices
            ]
            .copy()
        )

        top_df[
            "qwen3_score"
        ] = (
            scores[
                top_indices
            ]
        )

        top_df[
            "qwen3_rank"
        ] = range(
            1,
            len(top_df) + 1,
        )

        top_df[
            "query_id"
        ] = query_id

        top_df[
            "query_type"
        ] = query_type

        top_df[
            "query"
        ] = query_text

        top_df[
            "is_novel"
        ] = [
            (
                query_id,
                document_id,
            )
            not in existing_pairs
            for document_id
            in top_df[
                "document_id"
            ]
        ]

        novel_count = int(
            top_df[
                "is_novel"
            ]
            .sum()
        )

        judged_count = (
            len(top_df)
            - novel_count
        )

        print(
            f"  Top-{top_k}: "
            f"existing={judged_count}, "
            f"novel={novel_count}"
        )

        summary_rows.append({
            "query_id":
                query_id,

            "video_id":
                video_id,

            "query_type":
                query_type,

            "query":
                query_text,

            "video_documents":
                len(video_df),

            "top_k":
                len(top_df),

            "already_judged":
                judged_count,

            "novel_candidates":
                novel_count,

            "novel_ratio":
                novel_count
                / len(top_df),
        })

        result_rows.append(
            top_df
        )

    # ========================================================
    # Combine
    # ========================================================

    results = pd.concat(
        result_rows,
        ignore_index=True,
    )

    # Exactly 60 * 25 expected here.
    print()
    print("=" * 90)
    print("POOL EXPANSION SUMMARY")
    print("=" * 90)

    print(
        "Top25 rows:",
        f"{len(results):,}",
    )

    print(
        "Queries:",
        results[
            "query_id"
        ].nunique(),
    )

    # ========================================================
    # Check query-document uniqueness
    # ========================================================

    duplicate_pairs = (
        results[
            [
                "query_id",
                "document_id",
            ]
        ]
        .duplicated()
        .sum()
    )

    print(
        "Duplicate query-document pairs:",
        duplicate_pairs,
    )

    if duplicate_pairs != 0:
        raise RuntimeError(
            "Duplicate Qwen retrieval pairs."
        )

    # ========================================================
    # Save all Qwen Top25 results
    # ========================================================

    top25_columns = [
        "query_id",
        "video_id",
        "video_title",
        "query_type",
        "query",
        "document_id",
        "representative_comment_id",
        "comment_text",
        "qwen3_rank",
        "qwen3_score",
        "is_novel",
    ]

    results[
        top25_columns
    ].to_csv(
        TOP25_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Build blinded novel-candidate labeling file
    # ========================================================

    novel = (
        results[
            results[
                "is_novel"
            ]
        ]
        .copy()
    )

    # Stable label ID for expansion judgment.
    novel[
        "label_id"
    ] = [
        (
            f"{query_id}_qwen3_"
            f"{rank:02d}"
        )
        for query_id, rank
        in zip(
            novel[
                "query_id"
            ],
            novel[
                "qwen3_rank"
            ],
        )
    ]

    # Do NOT expose qwen rank or score to Work.
    novel_columns = [
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

    novel_blind = (
        novel[
            novel_columns
        ]
        .sample(
            frac=1,
            random_state=20260904,
        )
        .reset_index(
            drop=True
        )
    )

    novel_blind.to_csv(
        NOVEL_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Save summary
    # ========================================================

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    total_novel = len(
        novel
    )

    print()
    print(
        "Already judged Top25 pairs:",
        f"{len(results) - total_novel:,}",
    )

    print(
        "Novel Top25 pairs:",
        f"{total_novel:,}",
    )

    print(
        "Novel ratio:",
        f"{total_novel / len(results) * 100:.2f}%",
    )

    print()
    print(
        "Queries with at least one novel candidate:",
        int(
            (
                summary[
                    "novel_candidates"
                ]
                > 0
            )
            .sum()
        ),
    )

    print()
    print(
        "Mean novel candidates/query:",
        f"{summary['novel_candidates'].mean():.2f}",
    )

    print(
        "Median novel candidates/query:",
        f"{summary['novel_candidates'].median():.2f}",
    )

    print(
        "Max novel candidates/query:",
        int(
            summary[
                "novel_candidates"
            ].max()
        ),
    )

    print()
    print("All Qwen Top25:")
    print(
        TOP25_PATH
    )

    print()
    print("Blind novel-candidate Work file:")
    print(
        NOVEL_PATH
    )

    print()
    print("Summary:")
    print(
        SUMMARY_PATH
    )

    print()
    print(
        "QWEN3 POOL EXPANSION: PASS"
    )


if __name__ == "__main__":
    main()