import json
import os
from datetime import datetime

import polars as pl
from sentence_transformers import SentenceTransformer

from rag.config import (
    AI_OUTPUT_PATH,
    FAILED_IDS_PATH,
    MODEL_NAME,
    MODEL_PATH,
    RETRIEVAL_QUERY,
)
from rag.data_models import RAGDeps, RAGResponse
from rag.query.agent import rag_agent
from rag.query.helpers import log_failed, save_as_parquet

# Config --------------------------------------------------------------------
df_schema = pl.Schema(
    {
        "doc_id": pl.Int64,
        "retrieval_query": pl.Utf8,
        "is_nuclear": pl.Boolean,
        "arguments": pl.Utf8,  # JSON string: list[{name, summary}]
        "confidence": pl.Float64,
        "queried_at": pl.Datetime(),
    }
)


def run_pipeline(
    doc_ids: int | list[int],
    retrieval_query: str = RETRIEVAL_QUERY,
    df_schema: pl.Schema = df_schema,
) -> None:
    # Normalize to list
    if isinstance(doc_ids, int):
        doc_ids = [doc_ids]

    # Embedder setup ------------------------------------------------------------
    use_local = True
    if not (MODEL_PATH and os.path.isdir(MODEL_PATH) and os.listdir(MODEL_PATH)):
        use_local = False

    embedder = SentenceTransformer(
        MODEL_PATH if use_local else MODEL_NAME, local_files_only=use_local
    )

    rows: list[dict] = []
    failed: list[int] = []

    for doc_id in doc_ids:
        deps = RAGDeps.create(embedder, retrieval_query, doc_id=doc_id)
        try:
            result = rag_agent.run_sync(
                retrieval_query, deps=deps, output_type=RAGResponse
            )
            output = result.output

            rows.append(
                {
                    "doc_id": doc_id,
                    "retrieval_query": retrieval_query,
                    "is_nuclear": output.is_nuclear,
                    "arguments": json.dumps(
                        [arg.model_dump() for arg in output.arguments]
                        if output.arguments
                        else [],
                        ensure_ascii=False,
                    ),
                    "confidence": output.confidence,
                    "queried_at": datetime.now(),
                }
            )
            print(
                f"[OK]  doc_id={doc_id} — is_nuclear={output.is_nuclear}, confidence={output.confidence}"
            )

        except Exception as e:
            print(f"[FAIL] doc_id={doc_id} — {e}")
            failed.append(doc_id)

    if rows:
        save_as_parquet(
            df_schema,
            rows,
            AI_OUTPUT_PATH,
        )

    log_failed(failed, FAILED_IDS_PATH)
