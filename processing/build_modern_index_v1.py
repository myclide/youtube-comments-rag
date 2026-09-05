from __future__ import annotations

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

MAX_SEQ_LENGTH = 512

BATCH_SIZE = 16


# ============================================================
# Paths
# ============================================================

ROOT_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

CORPUS_PATH = (
    ROOT_DIR
    / "data"
    / "corpus_v1"
    / "retrieval_comments_v1.parquet"
)

OUTPUT_DIR = (
    ROOT_DIR
    / "data"
    / "modern_index_v1"
)

EMBEDDINGS_PATH = (
    OUTPUT_DIR
    / "document_embeddings.npy"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "metadata.parquet"
)

MANIFEST_PATH = (
    OUTPUT_DIR
    / "manifest.json"
)


# ============================================================
# Helpers
# ============================================================

def choose_device() -> str:

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def build_document_id(
    frame: pd.DataFrame,
) -> pd.Series:

    return (
        frame["video_id"].astype(str)
        + "::"
        + frame[
            "representative_comment_id"
        ].astype(str)
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 90)
    print("BUILD MODERN QWEN3 SERVING INDEX")
    print("=" * 90)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Load corpus
    # ========================================================

    print()
    print("Loading retrieval corpus:")
    print(CORPUS_PATH)

    if not CORPUS_PATH.exists():

        raise FileNotFoundError(
            f"Corpus not found: {CORPUS_PATH}"
        )

    corpus = pd.read_parquet(
        CORPUS_PATH
    )

    print(
        "Retrieval documents:",
        f"{len(corpus):,}",
    )

    # ========================================================
    # Required columns
    # ========================================================

    required_columns = {
        "video_id",
        "video_title",
        "source_video_url",
        "comment_text",
        "representative_comment_id",
    }

    missing_columns = (
        required_columns
        - set(corpus.columns)
    )

    if missing_columns:

        raise RuntimeError(
            "Corpus missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # ========================================================
    # Stable document IDs
    # ========================================================

    corpus[
        "document_id"
    ] = build_document_id(
        corpus
    )

    duplicate_document_ids = int(
        corpus[
            "document_id"
        ]
        .duplicated()
        .sum()
    )

    print(
        "Duplicate document IDs:",
        duplicate_document_ids,
    )

    if duplicate_document_ids != 0:

        raise RuntimeError(
            "Retrieval corpus contains duplicate document IDs."
        )

    # Preserve embedding row position explicitly.
    corpus[
        "embedding_row"
    ] = np.arange(
        len(corpus),
        dtype=np.int64,
    )

    # ========================================================
    # Clean text
    # ========================================================

    texts = (
        corpus[
            "comment_text"
        ]
        .fillna("")
        .astype(str)
        .tolist()
    )

    empty_comments = sum(
        text.strip() == ""
        for text in texts
    )

    print(
        "Empty comments:",
        empty_comments,
    )

    if empty_comments != 0:

        raise RuntimeError(
            "Retrieval corpus contains empty comment text."
        )

    # ========================================================
    # Model
    # ========================================================

    device = choose_device()

    print()
    print("Model:")
    print(MODEL_NAME)

    print(
        "Device:",
        device,
    )

    print(
        "Max sequence length:",
        MAX_SEQ_LENGTH,
    )

    print(
        "Batch size:",
        BATCH_SIZE,
    )

    model = SentenceTransformer(
        MODEL_NAME,
        device=device,
    )

    model.max_seq_length = (
        MAX_SEQ_LENGTH
    )

    # ========================================================
    # Encode documents
    #
    # IMPORTANT:
    # Documents do NOT use prompt_name="query".
    #
    # This matches the evaluated Qwen3 configuration.
    # ========================================================

    print()
    print("=" * 90)
    print("ENCODING DOCUMENTS")
    print("=" * 90)

    start = time.perf_counter()

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    elapsed_seconds = (
        time.perf_counter()
        - start
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    # ========================================================
    # Integrity checks
    # ========================================================

    if (
        embeddings.shape[0]
        != len(corpus)
    ):

        raise RuntimeError(
            "Embedding row count does not match corpus."
        )

    if embeddings.ndim != 2:

        raise RuntimeError(
            "Expected 2D embedding matrix."
        )

    norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    max_norm_error = float(
        np.max(
            np.abs(
                norms - 1.0
            )
        )
    )

    print()
    print(
        "Embedding shape:",
        embeddings.shape,
    )

    print(
        "Encoding seconds:",
        f"{elapsed_seconds:.2f}",
    )

    print(
        "Documents/sec:",
        f"{len(corpus) / elapsed_seconds:.2f}",
    )

    print(
        "Maximum L2 norm error:",
        f"{max_norm_error:.8f}",
    )

    # ========================================================
    # Save
    # ========================================================

    print()
    print("=" * 90)
    print("SAVING INDEX")
    print("=" * 90)

    np.save(
        EMBEDDINGS_PATH,
        embeddings,
    )

    metadata_columns = [
        column
        for column in [
            "embedding_row",
            "document_id",
            "video_id",
            "video_title",
            "source_video_url",
            "comment_text",
            "representative_comment_id",
            "representative_author",
            "occurrence_count",
            "unique_author_count",
            "total_like_count",
            "max_like_count",
            "first_published_at",
            "last_published_at",
            "is_repeated_text",
        ]
        if column in corpus.columns
    ]

    metadata = (
        corpus[
            metadata_columns
        ]
        .copy()
    )

    metadata.to_parquet(
        METADATA_PATH,
        index=False,
    )

    manifest = {
        "version":
            "modern_index_v1",

        "model":
            MODEL_NAME,

        "documents":
            int(
                len(corpus)
            ),

        "embedding_dimension":
            int(
                embeddings.shape[1]
            ),

        "embedding_dtype":
            str(
                embeddings.dtype
            ),

        "normalized":
            True,

        "document_prompt":
            None,

        "query_prompt_name":
            "query",

        "max_seq_length":
            MAX_SEQ_LENGTH,

        "source_corpus":
            str(
                CORPUS_PATH
            ),

        "embedding_file":
            EMBEDDINGS_PATH.name,

        "metadata_file":
            METADATA_PATH.name,

        "encoding_seconds":
            elapsed_seconds,

        "documents_per_second":
            (
                len(corpus)
                / elapsed_seconds
            ),
    }

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "Embeddings:"
    )
    print(
        EMBEDDINGS_PATH
    )

    print()
    print(
        "Metadata:"
    )
    print(
        METADATA_PATH
    )

    print()
    print(
        "Manifest:"
    )
    print(
        MANIFEST_PATH
    )

    print()
    print(
        "MODERN INDEX BUILD: PASS"
    )


if __name__ == "__main__":
    main()