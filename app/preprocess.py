"""
preprocess.py

This script will:
- Read `data/Final Result.csv`
- Split the big Comments field into individual comments
- Build embeddings for each comment (on GPU if available)
- Build a FAISS index for similarity search
"""

import pandas as pd
import numpy as np
from pathlib import Path

from sentence_transformers import SentenceTransformer
import faiss


# Paths
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_CSV_PATH = DATA_DIR / "Final Result.csv"
COMMENTS_PARQUET = DATA_DIR / "comments.parquet"
FAISS_INDEX_PATH = DATA_DIR / "comments.index"


def load_raw_csv() -> pd.DataFrame:
    """
    Load the original CSV file and return a pandas DataFrame.
    """
    if not RAW_CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found at {RAW_CSV_PATH}")
    df = pd.read_csv(RAW_CSV_PATH)

    # Drop useless unnamed index column if it exists
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    return df


def explode_comments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split the 'Comments' column into individual comments.

    Output columns:
    - comment_text
    - video_title
    - video_link
    - views
    - uploaded_date
    - likes_on_video
    - dislikes_on_video
    """
    if "Comments" not in df.columns:
        raise KeyError("Column 'Comments' not found in CSV columns: "
                       f"{list(df.columns)}")

    rows = []

    for _, row in df.iterrows():
        video_title = row.get("Video Title", "")
        video_link = row.get("Video Link", "")
        views = row.get("Views", None)
        uploaded_date = row.get("Uploaded Date", "")
        likes = row.get("Likes on Video", None)
        dislikes = row.get("Dislikes on Video", None)

        raw_comments = row["Comments"]

        if pd.isna(raw_comments):
            continue

        # Split by newline into individual comments
        for c in str(raw_comments).split("\n"):
            c = c.strip()
            # Skip empty or very short comments
            if len(c) < 5:
                continue

            rows.append(
                {
                    "comment_text": c,
                    "video_title": video_title,
                    "video_link": video_link,
                    "views": views,
                    "uploaded_date": uploaded_date,
                    "likes_on_video": likes,
                    "dislikes_on_video": dislikes,
                }
            )

    comments_df = pd.DataFrame(rows)
    return comments_df


def build_embeddings_and_index(
    comments_df: pd.DataFrame,
    device: str = "cuda",
    batch_size: int = 128,
):
    """
    Build embeddings for each comment and a FAISS index.
    Embeddings are computed on the specified device (GPU if 'cuda').
    """
    texts = comments_df["comment_text"].astype(str).tolist()
    print(f"Number of comments to embed: {len(texts)}")

    # Load embedding model on GPU (if available)
    print(f"Loading embedding model on device: {device}")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

    print("Encoding comments to embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    # Normalize for inner-product similarity (cosine after normalization)
    print("Normalizing embeddings...")
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    print(f"Embedding dimension: {dim}")

    index = faiss.IndexFlatIP(dim)
    print("Adding embeddings to FAISS index...")
    index.add(embeddings.astype("float32"))

    # Save index
    print(f"Saving FAISS index to {FAISS_INDEX_PATH}")
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    # Save comments metadata (aligned with embeddings)
    print(f"Saving comments metadata to {COMMENTS_PARQUET}")
    comments_df.to_parquet(COMMENTS_PARQUET, index=False)

    print("Done building embeddings and index.")


def main():
    print(f"Looking for CSV at: {RAW_CSV_PATH}")
    df = load_raw_csv()
    print("CSV loaded successfully!")
    print("Number of video rows:", len(df))
    print("Video-level columns:", list(df.columns))

    comments_df = explode_comments(df)
    print("\nAfter exploding into individual comments:")
    print("Number of comment rows:", len(comments_df))
    print("Comment-level columns:", list(comments_df.columns)[:10])

    # Build embeddings + index
    build_embeddings_and_index(comments_df, device="cuda")


if __name__ == "__main__":
    main()
