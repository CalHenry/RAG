# Attempt at creating a RAG system locally


## Stack
- lancedb for the vector database
- sentence-transformers for the embedding model
- langchain-text-splitters for the chunking
- polars and pyarrow to deal with dataframes and input data (that is in a spreadsheet format)


## WHat are the documents
- Documents are texts between 20k and 100k characters. Date metadata is attached (year, month, day)


## RAG system 
A RAG system has 2 main components:
- Ingestion pipeline (text embedding, chunking, storage in a vector database<>)
- Query pipeline (retreiver, promtp builder)


### Ingestion pipeline

chunks: 1500 or 2000
overlap: 200

Embedding model "bge-small-en-v1.5" with sentence-transformers


#### Steps:
1. Prepare the raw data for chunking with overlap
2. Chunk the documents
3. Embedd the chunks
4. Create and store the chunks in the vector database
