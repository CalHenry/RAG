import polars as pl
from sentence_transformers import SentenceTransformer

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


# 2. Chunk the documents
CHUNK_SIZE = 1000
OVERLAP = 200
STEP = 100


# 3. Embedd the chunks


# 4. Create and store the chunks in the vector database
