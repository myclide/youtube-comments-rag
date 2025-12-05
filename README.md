\# YouTube Comments RAG System



Local Retrieval-Augmented Generation (RAG) system that analyzes \*\*YouTube video comments\*\* and summarizes what viewers say.  



\- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`

\- Index: FAISS (on disk)

\- LLM: \*\*Llama 3\*\* via \[Ollama](https://ollama.com/) (GPU-accelerated on local machine)

\- UI: Streamlit web app



The system can answer questions such as:



\- “What do people like about this video?”

\- “What complaints do viewers have?”

\- “What do commenters say about the audio quality?”



It supports \*\*all videos\*\* in the dataset or \*\*per-video analysis\*\* via dropdown.



---



\## Features



\- 🔍 \*\*Semantic search\*\* over hundreds of thousands of comments using FAISS.

\- 🧠 \*\*Local LLM (Llama 3 via Ollama)\*\* – no OpenAI API required.

\- 🎛️ \*\*Analysis scope\*\*:

&nbsp; - All videos

&nbsp; - Single selected video

\- 📊 Summaries that are \*\*grounded in actual comments\*\* (strict prompt to avoid hallucinations).

\- ⚡ GPU acceleration for both embeddings (PyTorch) and Llama 3.



---



\## Project Structure



```text

youtube-rag/

├─ app/

│  ├─ main.py             # Streamlit app (UI)

│  ├─ preprocess.py       # Load CSV -> explode comments -> save Parquet

│  ├─ build\_index.py      # Encode comments -> build FAISS index

│  ├─ rag\_pipeline.py     # RAG pipeline (retrieve + LLM summarize)

│  ├─ models.py           # Llama 3 (Ollama) wrapper

│  └─ \_\_init\_\_.py

├─ data/

│  ├─ Final Result.csv    # Original YouTube export (not tracked in git by default)

│  ├─ comments\_raw.parquet

│  ├─ comments.parquet

│  └─ comments.index      # FAISS index (not tracked in git by default)

├─ README.md

├─ .gitignore

└─ requirements.txt       # Python dependencies (optional)



