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



## Dataset

This project uses a YouTube comments dataset created by **Ahmed Shahriar Sakib** from Kaggle:

- Kaggle notebook:  
  `https://www.kaggle.com/code/ahmedshahriarsakib/scrape-youtube-comments-for-free-no-google-api/input`
- Original file name: **`Final Result.csv`**

That notebook scrapes YouTube comments (without using the official Google API) and saves the results into `Final Result.csv`. The file contains multiple videos and their comments, with columns such as:

- `Video Id` / `Video Title`
- `Comment`
- `Likes`
- `Time`
- … (plus any extra fields from the notebook)

### How this repo uses the dataset

The raw Kaggle CSV is **not included in this repository** (for size and licensing reasons).  
To run the app yourself, you need to:

1. Download `Final Result.csv` from the Kaggle notebook above.
2. Place it in the `data/` folder. For example:

   - `data/Final Result.csv`  *(original name)*  
   - or rename to something like `data/comments_raw.csv` if you prefer.

3. Update `app/preprocess.py` if necessary so it reads the correct filename and column names (e.g., map `Comment` → `comment_text`, `Video Title` → `video_title`, etc.).
4. Run the preprocessing and index-building steps:

   ```bash
   cd path/to/your/project
   python app/preprocess.py        # cleans data and writes processed file (e.g. data/comments.parquet)
   python app/rag_pipeline.py      # builds FAISS index over processed comments



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



