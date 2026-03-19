import asyncio
import os

import polars as pl
from sentence_transformers import SentenceTransformer

from rag.config import (
    FAILED_IDS_PATH,
    MODEL_NAME,
    MODEL_PATH,
    RETRIEVAL_QUERY,
    VECTOR_OUTPUT_PATH,
)
from rag.data_models import RAGDeps
from rag.query.agent import retrieve
from rag.query.helpers import log_failed, save_as_parquet

# Config --------------------------------------------------------------------

# Define the doc_ids to process
DOC_IDS: list[int] = list(range(1, 100))

# output df schema
df_schema = pl.Schema(
    {
        "doc_id": pl.Int64,
        "publish_date": pl.Date,
        "chunks": pl.List(
            pl.Struct(
                {
                    "chunk_id": pl.Int64,
                    "chunk_text": pl.String,
                    "_distance": pl.Float32,
                }
            )
        ),
    },
)

# Embedder setup ------------------------------------------------------------

use_local = True
if not (MODEL_PATH and os.path.isdir(MODEL_PATH) and os.listdir(MODEL_PATH)):
    use_local = False

embedder = SentenceTransformer(
    MODEL_PATH if use_local else MODEL_NAME, local_files_only=use_local
)


# Core async retrieval loop -------------------------------------------------


async def fetch_all(
    retrieval_query, doc_ids: list[int]
) -> tuple[list[dict], list[int]]:
    """
    Use the retrieve function to fetch for each document, the top k closest vectors.
    Return the associated chunks in a dataset.
    This is the non AI part of RAG.

    Return:
    ----
        rows: 1 row per document - used to make a dataset outside this function
        failed: doc_id of the failed iterations if any
    """
    failed: list[int] = []
    rows: list[dict] = []

    for doc_id in doc_ids:
        deps = RAGDeps.create(embedder, RETRIEVAL_QUERY, doc_id=doc_id)
        try:
            chunks: list[dict] = await retrieve(deps, RETRIEVAL_QUERY, doc_id=doc_id)

            publish_date = chunks[0].get("publish_date") if chunks else None

            rows.append(
                {
                    "doc_id": doc_id,
                    "publish_date": publish_date,
                    "chunks": [
                        {
                            "chunk_id": c["chunk_id"],
                            "chunk_text": c["chunk_text"],
                            "_distance": c["_distance"],
                        }
                        for c in chunks
                    ],
                }
            )
            print(f"[OK]  doc_id={doc_id} — {len(chunks)} chunk(s) retrieved")

        except Exception as e:
            print(f"[FAIL] doc_id={doc_id} — {e}")
            failed.append(doc_id)

    return rows, failed


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rows, failed = asyncio.run(fetch_all(RETRIEVAL_QUERY, doc_ids=DOC_IDS))
    save_as_parquet(df_schema, rows, VECTOR_OUTPUT_PATH)
    log_failed(failed, FAILED_IDS_PATH)

    if failed:
        print(f"\nFailed doc_ids: {failed}")
