# Local RAG system

We want to build a RAG system to help an LLM answer specific questions about documents.  
The documents are long and messy. They contain a lot of noise that comes from irrelevant text and sentences as well as transcription mistakes, punctuation artifacts, OCR errors.  
The documents are the transcrip of legal debate public institutions debates.
The goal is to extract the arguments regarding a specific topic.
This system is fully local and offline.

Why RAG works and help:
- we know what we want to find
- we don't know which documents talk about the subject
- if it does, we want to catch the arguments and have them summarized in a sentence
- The documents are long and messy and cannot be process directly by LLMs or at the risk of hallucinations or poor restitution from the model
- RAG allow us to cut the documents into chunks, index them in a database and use only the chunks that most likely contains the arguments
- The LLM then has a much smaller/ simpler context to process, clear instructions, which result in a much better performances
- Versatile: We can use the same RAG and database, change the prompts and extract different informations from the documents 


## Stack
- [Lancedb](https://lancedb.com/) for the vector database
- [sentence-transformer](https://sbert.net/) for the embedding model
- [Polars](https://docs.pola.rs/) for the chunking
- polars and pyarrow to deal with dataframes and input data
- [LM studio](https://lmstudio.ai/) for a local LLM api


## Input documents
- Documents are texts between 20k and 100k characters. Date metadata is attached (year, month, day)


## RAG system 
A RAG system has 2 main components:
- Ingestion pipeline (text embedding, chunking, storage in a vector database<>)
- Query pipeline (retreiver, promtp builder)


### Ingestion pipeline

chunks: 1000 characters  
overlap: 200 characters

Embedding model [BAAI/bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5)


### Steps:
1. Prepare the raw data for chunking with overlap
2. Chunk the documents
3. Embedd the chunks
4. Create and store the chunks in the vector database


### Query pipeline

Pipeline elements:
- AI agent (framework: Pydantic-ai, model: [Ministral-3-3B-Instruct](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-GGUF))
- Local vector database (LanceDB)
- retreiver function for the vector database
- Embedding model (same as the documents embedding model: BAAI/bge-large-zh-v1.5)
- 2 system prompts, 1 user prompt, additional context from the RAG

Pipeline execution:
1. Embedd the user uery
2. Select the top 5 most similar vectors from the database, append them to the prompt
3. LLM process the user query with the added context
4. LLM respond following a structured output


### AI agent

### Prompts
