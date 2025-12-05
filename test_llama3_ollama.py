import requests

OLLAMA_URL = "http://localhost:11434/api/chat"

def call_llama3(prompt: str) -> str:
    """
    Call local Llama 3 via Ollama and return the assistant's reply text.
    """
    payload = {
        "model": "llama3",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }

    # Send request to Ollama
    resp = requests.post(OLLAMA_URL, json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()

    # Ollama chat format: data["message"]["content"]
    return data["message"]["content"]

def main():
    prompt = "Say hello in one short sentence about YouTube comments."
    answer = call_llama3(prompt)
    print("PROMPT:", prompt)
    print("\nLLAMA 3 ANSWER:\n", answer)

if __name__ == "__main__":
    main()
