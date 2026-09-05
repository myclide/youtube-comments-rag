from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sentence_transformers import (
    CrossEncoder,
    SentenceTransformer,
)


# ============================================================
# Configuration
# ============================================================

EMBEDDING_MODEL_NAME = (
    "Qwen/Qwen3-Embedding-0.6B"
)

RERANKER_MODEL_NAME = (
    "BAAI/bge-reranker-v2-m3"
)

DEFAULT_CANDIDATE_DEPTH = 25

DEFAULT_TOP_K = 10

MAX_SEQ_LENGTH = 512

RERANK_BATCH_SIZE = 16


# ============================================================
# Result object
# ============================================================

@dataclass
class RetrievedComment:

    text: str

    video_title: str

    video_link: Optional[str]

    score: float

    extra: Dict[str, Any]


# ============================================================
# Pipeline
# ============================================================

class ModernRAGPipeline:

    """
    Evaluated retrieval pipeline:

        Qwen3-Embedding-0.6B
            ->
        normalized dense retrieval
            ->
        Top-N candidates
            ->
        BGE-reranker-v2-m3
            ->
        final Top-k comments

    Document embeddings are built OFFLINE by:

        processing/build_modern_index_v1.py

    Query embeddings use:

        prompt_name="query"

    matching the benchmark configuration.
    """

    def __init__(
        self,
        device: str = "auto",
        data_dir: Optional[Path] = None,
    ) -> None:

        self.device = (
            self._resolve_device(
                device
            )
        )

        if data_dir is None:

            root_dir = (
                Path(__file__)
                .resolve()
                .parent
                .parent
            )

            data_dir = (
                root_dir
                / "data"
                / "modern_index_v1"
            )

        self.data_dir = Path(
            data_dir
        )

        embeddings_path = (
            self.data_dir
            / "document_embeddings.npy"
        )

        metadata_path = (
            self.data_dir
            / "metadata.parquet"
        )

        if not embeddings_path.exists():

            raise FileNotFoundError(
                "Modern embeddings not found at "
                f"{embeddings_path}. "
                "Run processing/build_modern_index_v1.py first."
            )

        if not metadata_path.exists():

            raise FileNotFoundError(
                "Modern metadata not found at "
                f"{metadata_path}. "
                "Run processing/build_modern_index_v1.py first."
            )

        # ====================================================
        # Metadata
        # ====================================================

        print(
            f"Loading modern retrieval metadata "
            f"from {metadata_path} ..."
        )

        self.comments_df = (
            pd.read_parquet(
                metadata_path
            )
            .reset_index(
                drop=True
            )
        )

        # ====================================================
        # Embeddings
        #
        # mmap keeps startup memory behavior more manageable.
        # Per-video retrieval only materializes selected rows.
        # ====================================================

        print(
            f"Memory-mapping document embeddings "
            f"from {embeddings_path} ..."
        )

        self.document_embeddings = (
            np.load(
                embeddings_path,
                mmap_mode="r",
            )
        )

        if (
            len(
                self.comments_df
            )
            != self.document_embeddings.shape[0]
        ):

            raise RuntimeError(
                "Metadata row count does not match "
                "embedding matrix."
            )

        # ====================================================
        # Video lookup
        # ====================================================

        self.video_row_indices = {}

        for video_id, group in (
            self.comments_df
            .groupby(
                "video_id",
                sort=False,
            )
        ):

            self.video_row_indices[
                str(
                    video_id
                )
            ] = (
                group.index
                .to_numpy(
                    dtype=np.int64
                )
            )

        # ====================================================
        # Models
        # ====================================================

        print(
            "Loading Qwen3 embedding model "
            f"on device={self.device} ..."
        )

        self.embedder = (
            SentenceTransformer(
                EMBEDDING_MODEL_NAME,
                device=self.device,
            )
        )

        self.embedder.max_seq_length = (
            MAX_SEQ_LENGTH
        )

        print(
            "Loading BGE reranker "
            f"on device={self.device} ..."
        )

        self.reranker = (
            CrossEncoder(
                RERANKER_MODEL_NAME,
                device=self.device,
                max_length=MAX_SEQ_LENGTH,
            )
        )

        print(
            "ModernRAGPipeline initialized."
        )

    # ========================================================
    # Device
    # ========================================================

    @staticmethod
    def _resolve_device(
        device: str,
    ) -> str:

        if device != "auto":
            return device

        if torch.cuda.is_available():
            return "cuda"

        return "cpu"

    # ========================================================
    # Query embedding
    # ========================================================

    def _encode_query(
        self,
        query: str,
    ) -> np.ndarray:

        embedding = self.embedder.encode(
            query,
            prompt_name="query",
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return np.asarray(
            embedding,
            dtype=np.float32,
        )

    # ========================================================
    # Candidate search
    # ========================================================

    @staticmethod
    def _top_indices(
        scores: np.ndarray,
        k: int,
    ) -> np.ndarray:

        n = len(
            scores
        )

        if n == 0:

            return np.array(
                [],
                dtype=np.int64,
            )

        k = min(
            int(k),
            n,
        )

        if k == n:

            return np.argsort(
                -scores
            )

        candidate_indices = np.argpartition(
            -scores,
            kth=k - 1,
        )[:k]

        sorted_local = (
            candidate_indices[
                np.argsort(
                    -scores[
                        candidate_indices
                    ]
                )
            ]
        )

        return sorted_local

    # ========================================================
    # Dense candidate generation
    # ========================================================

    def _dense_candidates(
        self,
        query_vector: np.ndarray,
        candidate_depth: int,
        video_id_filter: Optional[str],
    ) -> pd.DataFrame:

        # ----------------------------------------------------
        # Per-video retrieval
        #
        # This matches the benchmark retrieval scope.
        # ----------------------------------------------------

        if video_id_filter is not None:

            video_id = str(
                video_id_filter
            )

            row_indices = (
                self.video_row_indices
                .get(
                    video_id
                )
            )

            if row_indices is None:

                return pd.DataFrame()

            local_embeddings = np.asarray(
                self.document_embeddings[
                    row_indices
                ],
                dtype=np.float32,
            )

            scores = (
                local_embeddings
                @ query_vector
            )

            local_top = self._top_indices(
                scores,
                candidate_depth,
            )

            global_rows = (
                row_indices[
                    local_top
                ]
            )

            dense_scores = (
                scores[
                    local_top
                ]
            )

        # ----------------------------------------------------
        # Global retrieval
        #
        # NOTE:
        # The 60-query benchmark evaluated single-video
        # retrieval. Global mode is supported by the app but
        # is not covered by those reported benchmark metrics.
        # ----------------------------------------------------

        else:

            scores = (
                self.document_embeddings
                @ query_vector
            )

            global_rows = self._top_indices(
                scores,
                candidate_depth,
            )

            dense_scores = (
                scores[
                    global_rows
                ]
            )

        candidates = (
            self.comments_df
            .iloc[
                global_rows
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        candidates[
            "dense_score"
        ] = np.asarray(
            dense_scores,
            dtype=float,
        )

        return candidates

    # ========================================================
    # Reranking
    # ========================================================

    def _rerank(
        self,
        query: str,
        candidates: pd.DataFrame,
    ) -> pd.DataFrame:

        if candidates.empty:

            return candidates

        pairs = [
            (
                query,
                str(
                    comment
                ),
            )
            for comment
            in candidates[
                "comment_text"
            ]
        ]

        reranker_scores = (
            self.reranker.predict(
                pairs,
                batch_size=RERANK_BATCH_SIZE,
                show_progress_bar=False,
            )
        )

        candidates = (
            candidates.copy()
        )

        candidates[
            "reranker_score"
        ] = np.asarray(
            reranker_scores
        ).reshape(
            -1
        )

        candidates = (
            candidates
            .sort_values(
                by=[
                    "reranker_score",
                    "document_id",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
            .reset_index(
                drop=True
            )
        )

        return candidates

    # ========================================================
    # Public retrieval API
    # ========================================================

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        video_id_filter: Optional[str] = None,
        candidate_depth: int = DEFAULT_CANDIDATE_DEPTH,
    ) -> List[RetrievedComment]:

        query = (
            query.strip()
        )

        if not query:
            return []

        candidate_depth = max(
            int(candidate_depth),
            int(top_k),
        )

        query_vector = (
            self._encode_query(
                query
            )
        )

        candidates = (
            self._dense_candidates(
                query_vector=query_vector,
                candidate_depth=candidate_depth,
                video_id_filter=video_id_filter,
            )
        )

        if candidates.empty:
            return []

        reranked = (
            self._rerank(
                query=query,
                candidates=candidates,
            )
        )

        final_results = (
            reranked
            .head(
                min(
                    int(top_k),
                    len(
                        reranked
                    ),
                )
            )
        )

        results: List[
            RetrievedComment
        ] = []

        for row in final_results.itertuples(
            index=False
        ):

            extra = {
                "document_id":
                    getattr(
                        row,
                        "document_id",
                        None,
                    ),

                "video_id":
                    getattr(
                        row,
                        "video_id",
                        None,
                    ),

                "dense_score":
                    float(
                        getattr(
                            row,
                            "dense_score",
                            0.0,
                        )
                    ),

                "reranker_score":
                    float(
                        getattr(
                            row,
                            "reranker_score",
                            0.0,
                        )
                    ),

                "occurrence_count":
                    getattr(
                        row,
                        "occurrence_count",
                        None,
                    ),

                "unique_author_count":
                    getattr(
                        row,
                        "unique_author_count",
                        None,
                    ),

                "total_like_count":
                    getattr(
                        row,
                        "total_like_count",
                        None,
                    ),
            }

            results.append(
                RetrievedComment(
                    text=str(
                        getattr(
                            row,
                            "comment_text",
                            "",
                        )
                    ),

                    video_title=str(
                        getattr(
                            row,
                            "video_title",
                            "",
                        )
                    ),

                    video_link=(
                        getattr(
                            row,
                            "source_video_url",
                            None,
                        )
                    ),

                    score=float(
                        getattr(
                            row,
                            "reranker_score",
                            0.0,
                        )
                    ),

                    extra=extra,
                )
            )

        return results