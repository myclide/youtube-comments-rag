from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# Must match the model used in preprocessing / index building
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class RetrievedComment:
    text: str
    video_title: str
    video_link: Optional[str]
    score: float
    extra: Dict[str, Any]


class RAGPipeline:
    """
    Loads:
      - comments metadata (data/comments.parquet)
      - FAISS index over comment embeddings (data/comments.index)
      - embedding model for queries (and per-video retrieval)

    Provides:
      - retrieve(query, top_k, video_title_filter=None)
    """

    def __init__(self, device: str = "cuda", data_dir: Optional[Path] = None) -> None:
        self.device = device

        if data_dir is None:
            # youtube-rag/
            #   app/
            #   data/
            base_dir = Path(__file__).resolve().parent.parent
            data_dir = base_dir / "data"

        self.data_dir = data_dir
        comments_path = self.data_dir / "comments.parquet"
        index_path = self.data_dir / "comments.index"

        if not comments_path.exists():
            raise FileNotFoundError(f"Comments parquet not found at {comments_path}")
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {index_path}")

        print(f"Loading comments metadata from {comments_path} ...")
        self.comments_df = pd.read_parquet(comments_path)

        print(f"Loading FAISS index from {index_path} ...")
        self.index = faiss.read_index(str(index_path))

        print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' on device={device} ...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
        print("RAGPipeline initialized.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _encode_and_normalize(self, texts: List[str]) -> np.ndarray:
        """Encode texts and L2-normalize to match the index."""
        emb = self.embedder.encode(
            texts,
            batch_size=128,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
        return emb / norms

    def _build_results_from_indices(
        self, row_indices: np.ndarray, scores: np.ndarray
    ) -> List[RetrievedComment]:
        """Create RetrievedComment objects from DataFrame indices + scores."""
        results: List[RetrievedComment] = []
        for idx, score in zip(row_indices, scores):
            if idx < 0 or idx >= len(self.comments_df):
                continue
            row = self.comments_df.iloc[int(idx)]
            extra = {
                "views": row.get("views"),
                "uploaded_date": row.get("uploaded_date"),
                "likes_on_video": row.get("likes_on_video"),
                "dislikes_on_video": row.get("dislikes_on_video"),
            }
            rc = RetrievedComment(
                text=row.get("comment_text", ""),
                video_title=row.get("video_title", ""),
                video_link=row.get("video_link"),
                score=float(score),
                extra=extra,
            )
            results.append(rc)
        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        video_title_filter: Optional[str] = None,
    ) -> List[RetrievedComment]:
        """
        Retrieve comments for a query.

        - If video_title_filter is None: global search using FAISS.
        - If video_title_filter is provided: restrict to that video's comments first,
          then rank **within that video only** using fresh embeddings.
        """
        query = query.strip()
        if not query:
            return []

        # 1) Encode query
        q_emb = self._encode_and_normalize([query])  # shape (1, dim)

        # ------------------------------------------------------------------
        # Case A: global search (all videos) via FAISS
        # ------------------------------------------------------------------
        if video_title_filter is None:
            # search deeper than top_k so we have room to filter/unique etc.
            search_k = min(self.index.ntotal, max(top_k * 20, top_k))
            D, I = self.index.search(q_emb, search_k)  # shapes (1, search_k)

            # Flatten and filter out -1 indices, keep unique order
            indices = I[0]
            scores = D[0]
            valid_mask = indices >= 0
            indices = indices[valid_mask]
            scores = scores[valid_mask]

            # Keep first top_k
            indices = indices[:top_k]
            scores = scores[:top_k]

            return self._build_results_from_indices(indices, scores)

        # ------------------------------------------------------------------
        # Case B: per-video search
        # ------------------------------------------------------------------
        # Subset comments to the selected video title
        mask = self.comments_df["video_title"] == video_title_filter
        subset_df = self.comments_df[mask]

        if subset_df.empty:
            print(f"[RAGPipeline] No comments found for video title '{video_title_filter}'.")
            return []

        # Encode all comments for this video (per-video retrieval)
        texts = subset_df["comment_text"].tolist()
        comment_embs = self._encode_and_normalize(texts)  # (N, dim)

        # Cosine similarity via dot product of normalized vectors
        q_vec = q_emb[0]  # (dim,)
        sims = comment_embs @ q_vec  # (N,)

        # Top-k within this video
        k = min(top_k, len(subset_df))
        top_indices_local = np.argsort(-sims)[:k]  # indices within subset_df

        # Map back to global row indices
        subset_row_indices = subset_df.index.to_numpy()
        global_indices = subset_row_indices[top_indices_local]
        scores = sims[top_indices_local]

        return self._build_results_from_indices(global_indices, scores)
