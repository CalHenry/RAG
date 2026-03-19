from dataclasses import dataclass
from datetime import date
from typing import Annotated, TypedDict

import lancedb
from lancedb.pydantic import LanceModel, Vector
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from rag.config import DB_PATH, TABLE_NAME


# ingestion pipeline --------------------------------------------------------
class DocumentModel(LanceModel):
    id_doc: int
    chunk_id: int
    publish_date: date
    chunk_text: str
    vector: Annotated[list[float], Vector(1024)]


# query pipeline ------------------------------------------------------------
class NuclearArgument(BaseModel):
    nom: str
    resume: str


class RAGResponse(BaseModel):
    is_nuclear: bool
    arguments: list[NuclearArgument] | None = None
    sources: list[int]
    confidence: float | None


@dataclass
class RAGDeps:
    db: lancedb.DBConnection
    table_name: str
    embedder: SentenceTransformer
    retrieval_query: str
    doc_id: int | list[int]
    top_k: int = 5

    @classmethod
    def create(
        cls,
        embedder: SentenceTransformer,
        retrieval_query: str,
        doc_id: int | list[int],
    ) -> "RAGDeps":
        """Factory method for RAGDeps. db and table_name are set in config.py"""
        return cls(
            db=lancedb.connect(DB_PATH),
            table_name=TABLE_NAME,
            embedder=embedder,
            retrieval_query=retrieval_query,
            doc_id=doc_id,
        )


class ChunkResult(TypedDict):
    """output of retrieve() in helpers.py"""

    chunk_id: int
    chunk_text: str
    publish_date: date
    _distance: float
