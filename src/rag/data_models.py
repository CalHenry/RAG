from dataclasses import dataclass
from datetime import date
from typing import Annotated, Any, Optional

import lancedb
from lancedb.pydantic import LanceModel, Vector
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from config import DB_PATH, TABLE_NAME


# ingestion pipeline
class DocumentModel(LanceModel):
    id_doc: int
    chunk_id: int
    publish_date: date
    chunk_text: str
    vector: Annotated[list[float], Vector(1024)]


# query pipeline
class NuclearArgument(BaseModel):
    nom: str
    resume: str


class RAGResponse(BaseModel):
    is_nuclear: bool
    arguments: list[NuclearArgument] | None = None
    sources: list[int]
    confidence: Optional[float]


@dataclass
class RAGDeps:
    db: lancedb.DBConnection
    table_name: str
    embedder: SentenceTransformer
    retrieval_query: str
    top_k: int = 5
    doc_id: int

    @classmethod
    def create(
        cls, embedder: SentenceTransformer, retrieval_query: str, doc_id: int
    ) -> "RAGDeps":
        """Factory method for RAGDeps. db and table_name are set in config.py"""
        return cls(
            db=lancedb.connect(DB_PATH),
            table_name=TABLE_NAME,
            embedder=embedder,
            retrieval_query=retrieval_query,
            doc_id=doc_id,
        )


RAGDeps.create
