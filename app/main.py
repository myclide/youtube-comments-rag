import streamlit as st
import pandas as pd

from rag_pipeline import RAGPipeline, RetrievedComment
from models import load_llm, generate_answer

st.set_page_config(page_title="YouTube Comments RAG", layout="wide")


@st.cache_resource
def get_pipeline() -> RAGPipeline:
    # Load the pipeline once and reuse it (GPU model + FAISS index)
    return RAGPipeline(device="cuda")


@st.cache_resource
def get_llm():
    # Load tokenizer + model once, reused across requests
    tokenizer, model = load_llm(device="cuda")
    return tokenizer, model


def build_rag_prompt(
    question: str,
    comments: list[RetrievedComment],
    max_comments: int = 20,
) -> str:
    """
    Build a very simple, strict prompt for the LLM.

    IMPORTANT:
    - We end the prompt with '### ANSWER START'.
    - After generation we show only what comes after that marker.
    """
    selected = comments[:max_comments]

    lines: list[str] = []

    # Short & strict instructions – no templates, no placeholders
    lines.append(
        "You summarize YouTube comments for a user.\n"
        "You will be given a question and a list of numbered comments.\n"
        "Answer the question using ONLY information from those comments.\n"
        "Do NOT use outside knowledge or invent new facts.\n"
        "If the comments do not contain information needed to answer part of the question,\n"
        "write: 'The comments do not mention this.'\n"
        "Write your answer in two parts:\n"
        "1) One short paragraph (2–4 sentences) giving the overall opinion of commenters.\n"
        "2) 3–5 bullet points with specific details from the comments.\n"
        "Do not copy comments word-for-word; paraphrase them briefly.\n"
        "Stay under 180 words in total.\n"
    )

    # User question
    lines.append("\nUser question:\n")
    lines.append(question.strip())

    # Comments block
    lines.append("\n\nComments:\n")
    for i, rc in enumerate(selected, start=1):
        text = rc.text.replace("\n", " ").strip()
        if len(text) > 220:
            text = text[:220] + "..."
        title = rc.video_title.replace("\n", " ").strip()
        lines.append(f"Comment {i} [Video: {title}]: {text}")

    # Marker
    lines.append(
        "\n\nNow write the answer.\n"
        "Start immediately with the paragraph (do not restate these instructions).\n"
        "### ANSWER START\n"
    )

    return "\n".join(lines)


# Initialise pipeline and LLM
pipeline = get_pipeline()
tokenizer, model = get_llm()

# -------------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------------

st.title("YouTube Comments RAG System")

st.write(
    "Type a question about video comments below. "
    "The system retrieves similar comments and uses a local LLM to summarize what viewers say.\n\n"
    "This version is STRICT: the answer must only use information actually present in comments."
)

st.divider()

# --- Scope selection: all videos vs single video ---

scope = st.radio(
    "Analysis scope",
    ["All videos", "Single video"],
    horizontal=True,
)

video_filter_value = None

if scope == "Single video":
    # Build dropdown of unique video titles from the comments DataFrame
    all_titles = (
        pipeline.comments_df["video_title"]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    selected_title = st.selectbox(
        "Choose a video to analyze",
        all_titles,
        index=0 if all_titles else None,
    )

    if selected_title:
        video_filter_value = selected_title
        st.caption("Only comments from this video will be used for retrieval and LLM answering.")
else:
    st.caption("Comments from all videos are eligible for retrieval.")

# --- Question + retrieval settings ---

query = st.text_input(
    "Enter your question (e.g. 'What do people like about this video?', "
    "'What complaints do viewers have?', 'What do people say about the audio quality?')"
)

top_k = st.slider(
    "Number of comments to retrieve for the LLM context",
    min_value=10,
    max_value=60,
    value=30,
    step=10,
)

run_button = st.button("Retrieve & Generate Answer")

if run_button and query.strip():
    # ---- RETRIEVAL ----
    with st.spinner("Retrieving comments..."):
        results = pipeline.retrieve(
            query=query,
            top_k=top_k,
            video_title_filter=video_filter_value,
        )

    st.write(f"Retrieved **{len(results)}** comments.")

    if not results:
        if scope == "Single video":
            st.info(
                "No comments found for this query in the selected video. "
                "Try a different question or switch to 'All videos'."
            )
        else:
            st.info("No comments found for this query. Try another phrase.")
    else:
        # ---- LLM ANSWER ----
        st.subheader("LLM Answer (based on retrieved comments)")

        with st.spinner("Generating summary with local LLM..."):
            prompt = build_rag_prompt(question=query, comments=results)
            raw_output = generate_answer(
                tokenizer,
                model,
                prompt,
                max_new_tokens=160,   # short-ish answer
                temperature=0.6,      # a bit more creative to escape copying
                top_p=0.9,
                do_sample=True,
            )

        # Keep only the part after our marker
        marker = "### ANSWER START"
        if marker in raw_output:
            answer_text = raw_output.split(marker, 1)[1].strip()
        else:
            answer_text = raw_output.strip()

        if not answer_text:
            st.warning("LLM returned an empty answer for this prompt.")
        else:
            st.markdown(answer_text)

        # ---- RAW COMMENTS LIST ----
        st.divider()
        st.subheader("Retrieved comments (context)")

        for i, rc in enumerate(results, start=1):
            # Title
            st.markdown(f"### {i}. {rc.video_title}")

            # Comment text
            st.write(rc.text)

            # Metadata
            meta_parts: list[str] = []
            if rc.extra is not None:
                views = rc.extra.get("views")
                uploaded = rc.extra.get("uploaded_date")
                likes = rc.extra.get("likes_on_video")
                dislikes = rc.extra.get("dislikes_on_video")

                # Treat as strings to avoid int() errors
                if views is not None and not pd.isna(views) and str(views).strip():
                    meta_parts.append(f"Views: {views}")
                if uploaded:
                    meta_parts.append(f"Uploaded: {uploaded}")
                if likes is not None and not pd.isna(likes) and str(likes).strip():
                    meta_parts.append(f"Likes on video: {likes}")
                if dislikes is not None and not pd.isna(dislikes) and str(dislikes).strip():
                    meta_parts.append(f"Dislikes on video: {dislikes}")

            meta_text = " | ".join(meta_parts) if meta_parts else ""

            # Link + metadata caption
            link = rc.video_link if rc.video_link else ""
            if link:
                st.caption(f"[Open video]({link})  {meta_text}")
            else:
                st.caption(meta_text)

            st.markdown("---")
