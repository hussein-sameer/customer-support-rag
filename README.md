# Customer Support RAG Agent

A retrieval-augmented generation (RAG) assistant that answers customer support questions from a company FAQ document. It is built with **LangChain**, **Google Gemini** and **ChromaDB**.

The agent reads the *NovaCore FAQ Knowledge Base* (PDF), splits it into chunks, embeds them into a local vector store and, for each question, retrieves the most relevant passages before Gemini writes an answer based on them.

## How it works

```
 NovaCore_FAQ_Knowledge_Base.pdf
            │
            ▼
   PyPDFLoader  ──►  RecursiveCharacterTextSplitter (500 chars, 250 overlap)
                                   │
                                   ▼
                 GoogleGenerativeAIEmbeddings (gemini-embedding-2)
                                   │
                                   ▼
                      Chroma vector store (persisted in data/)
                                   │
   question ──► retriever (similarity, top k=3) ──► prompt template ──► Gemini (gemini-3.5-flash-lite) ──► answer
```

1. **Load**: `PyPDFLoader` reads the FAQ PDF from `data/`.
2. **Chunk**: `RecursiveCharacterTextSplitter` splits the text on newlines, then spaces, then characters (`chunk_size=500`, `chunk_overlap=250`).
3. **Embed and store**: the chunks are embedded with `gemini-embedding-2` and stored in a Chroma database saved to `data/`.
4. **Retrieve**: a similarity retriever returns the 3 chunks closest to the question.
5. **Generate**: an LCEL chain (`retriever | prompt | llm`) puts the retrieved context and the question into the prompt and sends it to `gemini-3.5-flash-lite`.

## Tech stack

| Component        | Tool                                        |
| ---------------- | ------------------------------------------- |
| Orchestration    | LangChain (LCEL)                            |
| LLM              | Google Gemini `gemini-3.5-flash-lite`       |
| Embeddings       | Google `gemini-embedding-2`                 |
| Vector store     | ChromaDB (`langchain-chroma`)               |
| Document loading | `PyPDFLoader` (`pypdf`)                     |
| Config           | `python-dotenv`                             |

## Project structure

```
customer_support_agent/
├── data/
│   ├── NovaCore_FAQ_Knowledge_Base.pdf   # source knowledge base
│   └── chroma.sqlite3                    # persisted Chroma vector store
├── src/
│   └── customer_support_agent/
│       └── app.py                        # RAG pipeline: load → chunk → embed → retrieve → answer
├── .env.example
├── requirements.txt
└── README.md
```

## Getting started

### Prerequisites

- Python 3.10+
- A Google AI Studio API key ([get one here](https://aistudio.google.com/apikey))

### Installation

```bash
git clone https://github.com/hussein-sameer/customer-support-rag.git
cd customer-support-rag

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

### Run

Run the script from the **project root**, because it reads `data/` with a relative path:

```bash
python src/customer_support_agent/app.py
```

The script builds the vector store and answers the sample question in `app.py`:

```python
response = rag_chain.invoke("whats the warranty for my personnal data protection?")
```

To ask something else, change the string passed to `rag_chain.invoke(...)`.

## Design decisions

**1. `data/` at the project root instead of `src/data/`**
Keeping data outside `src/` leaves the source tree for code only, and makes the data easy to manage separately (for example in `.gitignore`).

**2. `RecursiveCharacterTextSplitter` instead of `CharacterTextSplitter`**
The recursive splitter tries newlines first, then spaces, then single characters. This keeps FAQ entries and sentences together where possible, which gives more coherent chunks for retrieval.

**3. Chunk size tuned for the Gemini free tier**
The free tier allows about 100 requests per minute, and `GoogleGenerativeAIEmbeddings` sends a separate request for each chunk. The chunking parameters were set so the document produces **fewer than 50 chunks**, which leaves room for about 2 full runs per minute while testing. In production, the chunk size and overlap should be chosen by running an evaluation over several configurations.

## Limitations

- The vector store is rebuilt each time the script runs. Because Chroma persists to `data/`, this re-embeds the PDF and adds duplicate chunks on every run.
- Questions are hard-coded in `app.py`; there is no interactive interface yet.
- The prompt does not tell the model what to do when the context does not contain the answer.

## Roadmap

- [ ] Chat UI for asking questions interactively
- [ ] Reuse the persisted Chroma store instead of re-embedding on every run
- [ ] Evaluation set to compare chunk sizes, overlap and `k`
- [ ] Prompt rules for answers not found in the knowledge base
