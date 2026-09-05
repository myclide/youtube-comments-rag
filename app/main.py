from __future__ import annotations

import pandas as pd
import streamlit as st
import torch

from models import (
    generate_answer,
    load_llm,
)

from modern_retrieval import (
    ModernRAGPipeline,
)



# ============================================================
# Streamlit configuration
# ============================================================

st.set_page_config(
    page_title="YouTube Comments RAG",
    layout="wide",
)


# ============================================================
# Device
# ============================================================

def get_device() -> str:

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


DEVICE = get_device()


# ============================================================
# Cached pipelines
# ============================================================

@st.cache_resource
def get_modern_pipeline():

    return ModernRAGPipeline(
        device=DEVICE
    )


@st.cache_resource
def get_legacy_pipeline():

    # Import lazily so FAISS is only required when
    # the legacy MiniLM + FAISS backend is selected.
    from rag_pipeline import (
        RAGPipeline as LegacyRAGPipeline,
    )

    return LegacyRAGPipeline(
        device=DEVICE
    )

@st.cache_resource
def get_llm():

    return load_llm(
        device=DEVICE
    )


# ============================================================
# Prompt construction
# ============================================================

def build_rag_prompt(
    question: str,
    comments,
    max_comments: int = 20,
) -> str:

    selected = (
        comments[
            :max_comments
        ]
    )

    lines: list[str] = []

    lines.append(
        "You summarize YouTube comments for a user.\n"
        "You will be given a question and a list of numbered comments.\n"
        "Answer the question using ONLY information from those comments.\n"
        "Do NOT use outside knowledge or invent new facts.\n"
        "If the comments do not contain information needed to answer part "
        "of the question, write: 'The comments do not mention this.'\n"
        "Write your answer in two parts:\n"
        "1) One short paragraph of 2-4 sentences summarizing the main view.\n"
        "2) 3-5 bullet points with specific supporting details.\n"
        "Do not copy comments word-for-word; paraphrase briefly.\n"
        "Stay under 180 words in total."
    )

    lines.append(
        "\nUser question:\n"
    )

    lines.append(
        question.strip()
    )

    lines.append(
        "\n\nRetrieved comments:\n"
    )

    for i, comment in enumerate(
        selected,
        start=1,
    ):

        text = (
            comment.text
            .replace(
                "\n",
                " ",
            )
            .strip()
        )

        if len(text) > 300:

            text = (
                text[:300]
                + "..."
            )

        title = (
            comment.video_title
            .replace(
                "\n",
                " ",
            )
            .strip()
        )

        lines.append(
            f"Comment {i} "
            f"[Video: {title}]: "
            f"{text}"
        )

    lines.append(
        "\n\nNow write the answer.\n"
        "Start immediately with the answer.\n"
        "### ANSWER START\n"
    )

    return "\n".join(
        lines
    )


# ============================================================
# Header
# ============================================================

st.title(
    "YouTube Comments RAG"
)

st.write(
    "Retrieve evidence from YouTube comments and summarize it "
    "with a locally served Llama 3 model."
)

st.caption(
    f"Runtime device: {DEVICE}"
)

st.divider()


# ============================================================
# Retrieval backend
# ============================================================

st.sidebar.header(
    "Retrieval configuration"
)

backend = st.sidebar.radio(
    "Retrieval backend",
    [
        "Modern: Qwen3 + BGE",
        "Legacy: MiniLM + FAISS",
    ],
    index=0,
)

use_modern = (
    backend
    == "Modern: Qwen3 + BGE"
)


# ============================================================
# Load selected pipeline
# ============================================================

try:

    if use_modern:

        with st.spinner(
            "Loading Qwen3 + BGE retrieval models..."
        ):

            pipeline = (
                get_modern_pipeline()
            )

    else:

        with st.spinner(
            "Loading legacy MiniLM + FAISS pipeline..."
        ):

            pipeline = (
                get_legacy_pipeline()
            )

except FileNotFoundError as exc:

    st.error(
        str(exc)
    )

    if use_modern:

        st.info(
            "Build the modern offline document embeddings first:\n\n"
            "`python processing\\build_modern_index_v1.py`"
        )

    st.stop()


# ============================================================
# Modern retrieval profile
# ============================================================

candidate_depth = 25

if use_modern:

    retrieval_profile = (
        st.sidebar.radio(
            "Reranking profile",
            [
                "Quality-first (Top-25)",
                "CPU-balanced (Top-15)",
            ],
            index=0,
        )
    )

    if (
        retrieval_profile
        == "Quality-first (Top-25)"
    ):

        candidate_depth = 25

        st.sidebar.caption(
            "Reference benchmark configuration: "
            "nDCG@10 = 0.7347."
        )

    else:

        candidate_depth = 15

        st.sidebar.caption(
            "Lower CPU cost. Benchmark nDCG@10 = 0.7108 "
            "(96.75% of Top-25 nDCG)."
        )


# ============================================================
# Scope
# ============================================================

scope = st.radio(
    "Analysis scope",
    [
        "Single video",
        "All videos",
    ],
    horizontal=True,
)

video_id_filter = None
legacy_video_title_filter = None


# ============================================================
# Single-video selector
# ============================================================

if scope == "Single video":

    if use_modern:

        video_table = (
            pipeline.comments_df[
                [
                    "video_id",
                    "video_title",
                ]
            ]
            .drop_duplicates(
                subset=[
                    "video_id"
                ]
            )
            .sort_values(
                [
                    "video_title",
                    "video_id",
                ]
            )
            .reset_index(
                drop=True
            )
        )

        video_options = []

        video_lookup = {}

        for row in video_table.itertuples(
            index=False
        ):

            label = (
                f"{row.video_title} "
                f"[{row.video_id}]"
            )

            video_options.append(
                label
            )

            video_lookup[
                label
            ] = str(
                row.video_id
            )

        selected_video = (
            st.selectbox(
                "Choose a video to analyze",
                video_options,
                index=0
                if video_options
                else None,
            )
        )

        if selected_video:

            video_id_filter = (
                video_lookup[
                    selected_video
                ]
            )

            st.caption(
                "Retrieval is restricted to comments from "
                "the selected video. This matches the scope "
                "used by the 60-query retrieval benchmark."
            )

    else:

        all_titles = (
            pipeline.comments_df[
                "video_title"
            ]
            .dropna()
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

        selected_title = (
            st.selectbox(
                "Choose a video to analyze",
                all_titles,
                index=0
                if all_titles
                else None,
            )
        )

        if selected_title:

            legacy_video_title_filter = (
                selected_title
            )

            st.caption(
                "Legacy retrieval is restricted to this "
                "video title."
            )

else:

    if use_modern:

        st.warning(
            "The modern pipeline supports global retrieval, "
            "but the reported 60-query benchmark evaluated "
            "single-video retrieval only. Global-search quality "
            "is therefore not represented by the published "
            "benchmark metrics."
        )

    else:

        st.caption(
            "Legacy FAISS search considers comments from "
            "all videos."
        )


# ============================================================
# Query
# ============================================================

query = st.text_input(
    "Question",
    placeholder=(
        "Example: What laptops do viewers recommend "
        "for programming?"
    ),
)


# ============================================================
# Final context size
# ============================================================

if use_modern:

    top_k = st.slider(
        "Number of reranked comments passed to the LLM",
        min_value=5,
        max_value=20,
        value=10,
        step=1,
    )

else:

    top_k = st.slider(
        "Number of comments passed to the LLM",
        min_value=5,
        max_value=20,
        value=10,
        step=1,
    )


# ============================================================
# Run
# ============================================================

run_button = st.button(
    "Retrieve & Generate Answer",
    type="primary",
)

if (
    run_button
    and query.strip()
):

    # ========================================================
    # Retrieval
    # ========================================================

    with st.spinner(
        "Retrieving comments..."
    ):

        if use_modern:

            results = (
                pipeline.retrieve(
                    query=query,
                    top_k=top_k,
                    video_id_filter=video_id_filter,
                    candidate_depth=candidate_depth,
                )
            )

        else:

            results = (
                pipeline.retrieve(
                    query=query,
                    top_k=top_k,
                    video_title_filter=(
                        legacy_video_title_filter
                    ),
                )
            )

    st.write(
        f"Retrieved **{len(results)}** comments."
    )

    if not results:

        st.info(
            "No comments were retrieved. "
            "Try another query or scope."
        )

        st.stop()

    # ========================================================
    # Generation
    # ========================================================

    st.subheader(
        "Grounded answer"
    )

    with st.spinner(
        "Generating answer with local Llama 3..."
    ):

        tokenizer, model = (
            get_llm()
        )

        prompt = (
            build_rag_prompt(
                question=query,
                comments=results,
                max_comments=min(
                    20,
                    len(results),
                ),
            )
        )

        raw_output = (
            generate_answer(
                tokenizer,
                model,
                prompt,
                max_new_tokens=180,
                temperature=0.4,
                top_p=0.9,
                do_sample=True,
            )
        )

    marker = (
        "### ANSWER START"
    )

    if marker in raw_output:

        answer_text = (
            raw_output
            .split(
                marker,
                1,
            )[1]
            .strip()
        )

    else:

        answer_text = (
            raw_output.strip()
        )

    if answer_text:

        st.markdown(
            answer_text
        )

    else:

        st.warning(
            "The local LLM returned an empty answer."
        )

    # ========================================================
    # Retrieved evidence
    # ========================================================

    st.divider()

    st.subheader(
        "Retrieved evidence"
    )

    for i, result in enumerate(
        results,
        start=1,
    ):

        st.markdown(
            f"### {i}. {result.video_title}"
        )

        st.write(
            result.text
        )

        metadata = []

        if result.extra:

            if use_modern:

                dense_score = (
                    result.extra.get(
                        "dense_score"
                    )
                )

                reranker_score = (
                    result.extra.get(
                        "reranker_score"
                    )
                )

                occurrence_count = (
                    result.extra.get(
                        "occurrence_count"
                    )
                )

                if dense_score is not None:

                    metadata.append(
                        "Dense score: "
                        f"{float(dense_score):.4f}"
                    )

                if reranker_score is not None:

                    metadata.append(
                        "Reranker score: "
                        f"{float(reranker_score):.4f}"
                    )

                if (
                    occurrence_count
                    is not None
                    and not pd.isna(
                        occurrence_count
                    )
                ):

                    metadata.append(
                        "Occurrences in video: "
                        f"{occurrence_count}"
                    )

            else:

                views = (
                    result.extra.get(
                        "views"
                    )
                )

                uploaded = (
                    result.extra.get(
                        "uploaded_date"
                    )
                )

                likes = (
                    result.extra.get(
                        "likes_on_video"
                    )
                )

                if (
                    views is not None
                    and not pd.isna(
                        views
                    )
                ):

                    metadata.append(
                        f"Views: {views}"
                    )

                if uploaded:

                    metadata.append(
                        f"Uploaded: {uploaded}"
                    )

                if (
                    likes is not None
                    and not pd.isna(
                        likes
                    )
                ):

                    metadata.append(
                        f"Likes on video: {likes}"
                    )

        caption_parts = []

        if result.video_link:

            caption_parts.append(
                f"[Open video]({result.video_link})"
            )

        caption_parts.extend(
            metadata
        )

        if caption_parts:

            st.caption(
                " | ".join(
                    caption_parts
                )
            )

        st.markdown(
            "---"
        )