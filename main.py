import lancedb
import polars as pl
from sentence_transformers import SentenceTransformer

from src.rag.data_model import DocumentModel
from src.rag.ingestion_pipeline_helpers import (
    chunk_documents,
    embedd,
    prepare_raw_data,
)

"""
Steps:
    1. Prepare the raw data for chunking with overlap
    2. Chunk the documents
    3. Embedd the chunks
    4. Create and store the chunks in the vector database
"""

# 0. Import data as lazyframe
lf_raw = (
    pl.read_excel("data/raw/senat_elec20_LLM_test.xlsx").lazy().rename({"ID": "id"})
)

# 1. Prepare the raw data for chunking with overlap
lf = prepare_raw_data(lf_raw)

# 2. Chunk the documents
CHUNK_SIZE = 1000
OVERLAP = 200
STEP = 100

chunk_lf = chunk_documents(lf, CHUNK_SIZE, OVERLAP, STEP)


# 3. Embedd the chunks
model = SentenceTransformer("BAAI/bge-large-zh-v1.5")
df_embeddings = embedd(model, chunk_lf, chunk_column="chunk_text")


# 4. Create and store the chunks in the vector database
# We use a LanceModel (pydantic) for input validation
# Lancedb accepts polars natively because they both use Arrow schema under the hood

db = lancedb.connect("./data/database/rag_vector_db")
table = db.create_table(name="documents", schema=DocumentModel, mode="overwrite")
table.add(df_embeddings.to_arrow())
