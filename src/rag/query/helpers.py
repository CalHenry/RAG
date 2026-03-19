import os
from pathlib import Path

import polars as pl

from rag.data_models import RAGDeps


async def retrieve(
    deps: RAGDeps, retrieval_query: str, doc_id: int | list[int]
) -> list[dict]:
    """
    Retrieve chunks of documents from the vector database.
    1. Embedd the user query to find the closests vectors in the database
    2. Get the top_k chunks from the db, return them in a list
    """
    # embed the query with the same model used at ingestion
    vector = deps.embedder.encode(retrieval_query).tolist()

    table = deps.db.open_table(deps.table_name)
    results = (
        table.search(vector)
        .where(f"id_doc = {doc_id}")
        .limit(deps.top_k)
        .select(["chunk_id", "chunk_text", "publish_date", "_distance"])
        .to_list()
    )
    return results


# Failed ids log ------------------------------------------------------------
def log_failed(failed: list[int], path: Path) -> None:
    if not failed:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        for doc_id in failed:
            f.write(f"{doc_id}\n")
    print(f"Logged {len(failed)} failed doc_id(s) to {path}")


# Save AI results to parquet ------------------------------------------------
def save_as_parquet(
    df_schema: dict, new_rows: list[dict], output_path: str | Path
) -> None:
    """ """
    if not new_rows:
        print("No successful rows to write.")
        return

    new_df = pl.DataFrame(
        new_rows,
        schema=df_schema,
    )

    if os.path.exists(output_path):
        existing_df = pl.read_parquet(output_path)
        # Drop existing rows for doc_ids we just processed
        existing_df = existing_df.filter(~pl.col("doc_id").is_in(new_df["doc_id"]))
        combined_df = pl.concat([existing_df, new_df]).sort("doc_id")
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined_df = new_df.sort("doc_id")

    combined_df.write_parquet(output_path)
    print(f"\nWrote {len(combined_df)} row(s) to {output_path}")
