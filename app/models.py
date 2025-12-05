"""
models.py

LLM backend for the YouTube comments RAG system.

This version uses a local Llama 3 model served by Ollama.
We keep the same interface as before:
    - load_llm(device) -> (tokenizer, model)
    - generate_answer(tokenizer, model, prompt, max_new_tokens, temperature, top_p, ...)

So other parts of the app (rag_pipeline, main.py) do not need to change.
"""

import requests

# Ollama chat API endpoint
OLLAMA_URL = "http://localhost:11434/api/chat"

# System prompt to keep the model grounded in the given comments
SYSTEM_PROMPT = (
    "You are an assistant that answers questions about YouTube comments. "
    "You will receive a prompt that already contains the user's question "
    "and a list of retrieved comments (numbered). "
    "Use ONLY those comments to answer. "
    "Do not invent new comments, videos, people, or facts. "
    "If the comments do not contain enough information to answer a part of "
    "the question, explicitly say: 'The comments do not mention this.'"
)


def load_llm(device: str = "cuda"):
    """
    Compatibility stub so existing code can still call load_llm(...).

    For Ollama we don't need to load a model in Python; the model
    runs inside the Ollama server. So we just return (None, None).
    """
    tokenizer = None
    model = None
    return tokenizer, model


def _call_llama3(
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 256,
) -> str:
    """
    Call local Llama 3 via Ollama and return the assistant's reply text.
    """
    payload = {
        "model": "llama3",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            # sampling controls
            "temperature": float(temperature),
            "top_p": float(top_p),
            # limit length of completion (optional but nice)
            "num_predict": int(max_new_tokens),
        },
    }

    resp = requests.post(OLLAMA_URL, json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()

    # Ollama chat format: data["message"]["content"]
    return data["message"]["content"]


def generate_answer(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    **kwargs,
) -> str:
    """
    Main entry point used by the RAG pipeline.

    tokenizer and model are ignored (kept only for API compatibility).
    Any extra keyword arguments (e.g. top_k, repetition_penalty) are
    accepted via **kwargs and ignored, so main.py can pass them safely.
    """
    return _call_llama3(
        prompt=prompt,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )


# Optional: small self-test when running this file directly.
if __name__ == "__main__":
    test_prompt = (
        "You see these comments:\n"
        "1. I love this video, very helpful.\n"
        "2. Great explanation but a bit too fast.\n"
        "3. Thanks for the detailed review!\n\n"
        "Question: What do people like about this video?"
    )
    print("TEST PROMPT:\n", test_prompt)
    print("\nLLAMA 3 ANSWER:\n", generate_answer(None, None, test_prompt))
