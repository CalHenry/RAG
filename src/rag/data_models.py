from dataclasses import dataclass
from datetime import date
from typing import Any

import lancedb
from lancedb.pydantic import LanceModel, Vector
from pydantic import BaseModel


# ingestion pipeline
class DocumentModel(LanceModel):
    id_doc: int
    chunk_id: int
    publish_date: date
    chunk_text: str
    vector: Vector(1024)


# query pipeline
class RAGResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float


@dataclass
class RAGDeps:
    db: lancedb.DBConnection
    table_name: str
    embedder: Any
    query: str
    top_k: int = 5
