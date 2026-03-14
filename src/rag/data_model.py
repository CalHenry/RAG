from datetime import date

from lancedb.pydantic import LanceModel, Vector


class DocumentModel(LanceModel):
    id_doc: int
    publish_date: date
    chunk_text: str
    vector: Vector(1024)
