# Local RAG

We want to build a RAG system to help an LLM answer specific questions about documents.  
The documents are long and messy. They contain a lot of noise that comes from irrelevant text and sentences as well as transcription mistakes, punctuation artifacts, OCR errors.  
The documents are the transcrip of legal debate public institutions debates.
The goal is to extract the arguments regarding a specific topic.
This system is fully local and offline.

Why RAG works and helps:
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
- [Polars](https://docs.pola.rs/) for document chunking
- Polars and Arrow to deal with dataframes and input data
- [LM studio](https://lmstudio.ai/) for a local LLM api


## Input documents
- Documents are OCR or scraped transcriptions of legal debates. They contains a share of messy content, OCR errors, artifacts of the web page, irrelevant content or informations that are not the debate's content themselves.  
- The size of a full documents is ~100k characters but the extraction chunked them into piece of ~20K characters. We also have the publish date (year, month, day)


## RAG system 
The system got 2 main components:
- Ingestion pipeline (text embedding, chunking, storage in a vector database)
- Query pipeline (retriever, prompt builder, AI agent)


### Ingestion pipeline

Ingestion pipeline is purely data work, no AI. 

Steps:
  1. Prepare the raw data for chunking with overlap  
  2. Chunk the documents
  3. Embedd the chunks
  4. Create and store the chunks in the vector database  

#### Data preparation (steps 1, 2 and 3):
- Data is first re-aggregated to the full document level (from the original chunks)
- Then the documents are re chunked with these parameters:   
chunks: 1000 characters    
overlap: 200 characters  
step: 100 characters  

Details of the chunking emplementation can be found in the [docstrings](https://github.com/CalHenry/RAG/blob/main/src/rag/ingestion/helpers.py#L49).

Once the documents are chunked, we embedd them using [BAAI/bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5) with sentence-transformer.  
The vectors are added to the dataset of the chunked documents (step 2).

#### Vector database (step 4)
Create a database using LanceDB and store the dataset with the vectors.
LanceDB allow to store the vectors along side the associated data (chunks text, id and date informations) in a single unified data format.  
We don't have a huge amount of data and the database is stored on disk in `./data/database/`.


**Why LanceDB ?**  
LanceDB uses its  [own data format](https://docs.lancedb.com/lance), built on top of Apache Arrow, the same memory model as Polars. Since they share a common data layer, [they interoperate natively](https://docs.lancedb.com/integrations/data/polars_arrow): no conversion, no copy.  
In practice, this means I can pass a Polars DataFrame directly into a LanceDB table, or query one back out, with zero overhead. This makes the life of the developper easier (me) and ensure really good performances.  
Less relevant to this project but still nice to notice: Unlike most vector databases, which store only vectors and metadata, leaving you to fetch the actual data from a separate store, LanceDB keeps vectors and raw data in the same table. In this project, that meant retrieving text chunks directly alongside their embeddings, with no secondary lookup.


### Query pipeline

Pipeline elements:
- AI agent (framework: Pydantic-ai, model: [Ministral-3-3B-Instruct](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-GGUF))
- Local vector database (LanceDB)
- Retriever function for the vector database
- Embedding model (same as the documents embedding model: BAAI/bge-large-zh-v1.5)
- 2 system prompts, 1 user prompt, additional context from the RAG

Pipeline execution:
1. Embedd the user uery
2. Select the top 5 most similar vectors from the database, append the corresponding chunks to the system prompt
3. LLM process the user query with the added context
4. LLM respond following a structured output


#### AI agent & Prompts

Pydantic-ai is an agent framework that is built on top of data validation.  
Ministral-3-3B is a small LLM that support structured output.  

The LLM runs locally thanks to LM studio, that emits an OpenAI like API for the model that we can use in Pydantic-ai.  

The prompts are really important in our RAG.  
The documents are in french and so the prompts should be as well to correctly match the embedding of the prompt with the vectors of the database.  

The prompts ask the LLm to:
- Only use the provided context ("Réponds uniquement à partir du contexte fourni")
- If the document talks about the topic of interest (yes or no answser is asked) ("Ce document traite-t-il de {ctx.deps.retrieval_query} ?", "Réponds d'abord par Oui ou Non.")
- If yes, list the arguments and summarize them in one sentence. ("Si oui, liste chaque argument en une phrase")
