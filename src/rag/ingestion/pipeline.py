import os

import lancedb
import polars as pl
from sentence_transformers import SentenceTransformer

from config import DB_PATH, MODEL_NAME, MODEL_PATH, TABLE_NAME
from src.rag.data_models import DocumentModel
from src.rag.ingestion.helpers import (
    chunk_documents,
    embedd,
    prepare_raw_data,
)




def run_pipeline() -> None:
    """
    Steps:
        1. Prepare the raw data for chunking with overlap
        2. Chunk the documents
        3. Embedd the chunks
        4. Create and store the chunks in the vector database
    """
    # Env variables -------------------------------------------------------------
    use_local = True
    if not (MODEL_PATH and os.path.isdir(MODEL_PATH) and os.listdir(MODEL_PATH)):
        use_local = False

    # 0. Import data as lazyframe -----------------------------------------------
    lf_raw = pl.read_excel("data/raw/senat_elec20_LLM.xlsx").lazy().rename({"ID": "id"})

    # 1. Prepare the raw data for chunking with overlap
    lf = prepare_raw_data(lf_raw)

    # 2. Chunk the documents
    CHUNK_SIZE = 1000
    OVERLAP = 200
    STEP = 100

    chunk_lf = chunk_documents(lf, CHUNK_SIZE, OVERLAP, STEP)

    # 3. Embedd the chunks ------------------------------------------------------
    # Embedding model: "BAAI/bge-large-zh-v1.5" - https://huggingface.co/BAAI/bge-large-zh-v1.5

    print("Loading embedding model:")
    model = SentenceTransformer(
        MODEL_PATH if use_local else MODEL_NAME, local_files_only=use_local
    )
    df_embeddings = embedd(model, chunk_lf, chunk_column="chunk_text")

    # save to disk
    df_embeddings.write_parquet("./data/processed/df.embeddings.parquet")

    # 4. Create and store the chunks in the vector database ---------------------
    # We use a LanceModel (pydantic) for input validation
    # Lancedb accepts polars natively because they both use Arrow schema under the hood

    db = lancedb.connect(DB_PATH)
    table = db.create_table(TABLE_NAME, schema=DocumentModel, mode="overwrite")
    table.add(df_embeddings.to_arrow())


if __name__ == "__main__":
    run_pipeline()
