# Local RAG

This repo implements a local, fully offline RAG system designed to help an LLM answer specific questions about documents.  

Queries are scoped to a single document at a time: the vector database is filtered to only the chunks belonging to the target document, and retrieval happens entirely within that document's vectors. It is a **document-level retrieval system**.

The documents are transcripts of public institutional legal debates. They are long and messy, containing irrelevant passages, transcription errors, punctuation artifacts, and OCR noise. The goal is to extract and summarize the arguments made about a specific topic.

**This system is fully local and offline.**

---

## Why RAG?

- We know what we are looking for, but not which documents discuss the topic
- When a document does address it, we want to surface the relevant arguments and summarize each in a sentence
- The documents are too long and noisy to feed directly to an LLM without risking hallucinations or poor output quality
- RAG lets us chunk and index each document, then retrieve only the passages most likely to contain relevant arguments, giving the LLM a smaller, cleaner context and sharper instructions, which significantly improves output quality
- The system is versatile: the same pipeline and database can be reused with different prompts to extract different types of information (arguments, sentiment, positions, ...)

---

## Stack

- [LanceDB](https://lancedb.com/) — vector database
- [sentence-transformers](https://sbert.net/) — embedding model
- [Polars](https://docs.pola.rs/) — data manipulation and document chunking
- [Pydantic-AI](https://ai.pydantic.dev/) — AI agent framework
- [LM Studio](https://lmstudio.ai/) — local LLM inference server

---

<details>
<summary>## Folder structure</summary>
    ```sh
    .
    ├── compose.yml
    ├── Dockerfile
    ├── LICENSE
    ├── pyproject.toml
    ├── README.md
    ├── uv.lock
    ├── notebooks
    │   ├── chunking_benchmark.py
    │   ├── explo_docs.py
    │   └── explo_output.py
    ├── scripts
    │   ├── run_ingestion.py   # <- entry point for the ingestion pipeline
    │   └── run_query.py       # <- entry point for the query pipeline
    └── src
        └── rag
            ├── config.py
            ├── data_models.py
            ├── ingestion
            │   ├── helpers.py
            │   └── pipeline.py
            └── query
                ├── agent.py
                ├── build_dataset.py
                ├── helpers.py
                ├── merge.py
                └── pipeline.py
    ```
</details>

---

## Input documents

Documents are OCR or scraped transcriptions of legal debates. They contain a share of messy content: OCR errors, web page artifacts, and passages that are not part of the debate itself.

A full document is approximately 100k characters, but the source files deliver them in pre-extracted chunks of ~20k characters. Each document also has a publication date (year, month, day).

---

## RAG system

The system has two main components:

- **Ingestion pipeline** — text chunking, embedding, and storage in the vector database
- **Query pipeline** — retrieval, prompt construction, and LLM response

---

### Ingestion pipeline

The ingestion pipeline is purely data work, no AI involved.

**Steps:**
1. Re-aggregate the source chunks back to full document level
2. Re-chunk the documents with overlap
3. Embed the chunks
4. Store everything in the vector database

#### Chunking (steps 1–3)

Documents are first reconstructed from their source chunks, then re-chunked with the following parameters:

| Parameter | Value |
|-----------|-------|
| Chunk size | 1,000 characters |
| Overlap | 200 characters |

Implementation details are in the [chunking helper docstrings](https://github.com/CalHenry/RAG/blob/main/src/rag/ingestion/helpers.py#L49).

Once chunked, the documents are embedded using [BAAI/bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5) via sentence-transformers. The resulting vectors are added to the chunked document dataset.

#### Vector database (step 4)

The vector database is built with LanceDB and stored on disk at `./data/database/`.

LanceDB stores vectors alongside their associated data (chunk text, document ID, date) in a single unified format, with no secondary lookups needed.

**Why LanceDB?**

LanceDB uses its [own columnar format](https://docs.lancedb.com/lance), built on top of Apache Arrow, the same memory model as Polars. Because they share a common data layer, [they interoperate natively](https://docs.lancedb.com/integrations/data/polars_arrow): no conversion, no copy. In practice, this means passing a Polars DataFrame directly into a LanceDB table, or querying one back out, with zero overhead.

---

### Query pipeline

**Components:**
- Embedding model (same as ingestion: `BAAI/bge-large-zh-v1.5`)
- LanceDB retriever
- AI agent (framework: [Pydantic-AI](https://ai.pydantic.dev/), model: [Ministral-3B-Instruct](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-GGUF))

**Execution:**
1. Embed the user query
2. Retrieve the top 5 most similar chunks from the target document
3. Append the retrieved chunks to the system prompt as context
4. The LLM processes the query and responds with a structured output

#### AI agent & prompts

[Pydantic-AI](https://ai.pydantic.dev/) is an agent framework built around data validation and structured outputs. Ministral-3B is a small, fast LLM that supports structured output.

LM Studio exposes an OpenAI-compatible API for the local model, which Pydantic-AI can consume directly.

**The prompts are critical here.** The documents are in French, so the prompts are written in French as well, this is important to ensure the query embeddings align correctly with the document vectors in the database.

The prompts instruct the LLM to:
- Only use the provided context (*"Réponds uniquement à partir du contexte fourni"*)
- State whether the document addresses the topic of interest with a yes/no answer (*"Ce document traite-t-il de {topic} ? Réponds d'abord par Oui ou Non."*)
- If yes, list each argument and summarize it in one sentence (*"Si oui, liste chaque argument en une phrase"*)

The prompts can be found in [/rag/query/agents.py](https://github.com/CalHenry/RAG/blob/main/src/rag/query/agent.py).

####
