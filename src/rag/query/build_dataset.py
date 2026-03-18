import asyncio
import json
import os
from datetime import datetime

import polars as pl
from sentence_transformers import SentenceTransformer

from rag.config import FAILED_IDS_PATH, MODEL_NAME, MODEL_PATH, RETRIEVAL_QUERY
from rag.data_models import RAGDeps
from rag.query.agent import retrieve
from rag.query.helpers import log_failed

# Config --------------------------------------------------------------------

OUTPUT_PATH = "data/dataset.parquet"

# Define the doc_ids to process
DOC_IDS: list[int] = list(range(1, 26))

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
    failed: list[int] = []
    rows: list[dict] = []

    for doc_id in doc_ids:
        deps = RAGDeps.create(embedder, RETRIEVAL_QUERY, doc_id=doc_id)
        try:
            chunks: list[dict] = await retrieve(deps, RETRIEVAL_QUERY, doc_id=doc_id)

            # sources = list of unique chunk ids returned by retrieve
            sources = [c.get("id") for c in chunks if c.get("id") is not None]

            rows.append(
                {
                    "doc_id": doc_id,
                    "chunks": json.dumps(chunks, ensure_ascii=False),
                    "sources": sources,
                    "retrieved_at": datetime.now(),
                }
            )
            print(f"[OK]  doc_id={doc_id} — {len(chunks)} chunk(s) retrieved")

        except Exception as e:
            print(f"[FAIL] doc_id={doc_id} — {e}")
            failed.append(doc_id)

    return rows, failed


# Parquet upsert (append new, update existing rows by doc_id) ---------------
def upsert_parquet(new_rows: list[dict], output_path: str) -> None:
    if not new_rows:
        print("No successful rows to write.")
        return

    new_df = pl.DataFrame(
        new_rows,
        schema={
            "doc_id": pl.Int64,
            "chunks": pl.Utf8,
            "sources": pl.List(pl.Int64),
            "retrieved_at": pl.Datetime,
        },
    )

    if os.path.exists(output_path):
        existing_df = pl.read_parquet(output_path)
        # Drop existing rows for doc_ids we just processed (upsert semantics)
        existing_df = existing_df.filter(~pl.col("doc_id").is_in(new_df["doc_id"]))
        combined_df = pl.concat([existing_df, new_df]).sort("doc_id")
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined_df = new_df.sort("doc_id")

    combined_df.write_parquet(output_path)
    print(f"\nWrote {len(combined_df)} row(s) to {output_path}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rows, failed = asyncio.run(fetch_all(DOC_IDS))
    upsert_parquet(rows, OUTPUT_PATH)
    log_failed(failed, FAILED_IDS_PATH)

    if failed:
        print(f"\nFailed doc_ids: {failed}")
        print("Re-run by editing DOC_IDS in this script with the failed list.")
