import os

import lancedb
from sentence_transformers import SentenceTransformer

from config import MODEL_NAME, MODEL_PATH
from src.rag.agent import rag_agent
from src.rag.data_models import RAGDeps

use_local = True
if not (MODEL_PATH and os.path.isdir(MODEL_PATH) and os.listdir(MODEL_PATH)):
    use_local = False

embedder = SentenceTransformer(
    MODEL_PATH if use_local else MODEL_NAME, local_files_only=use_local
)

retrieval_query = "énergie nucléaire, centrale nucléaire, réacteur atomique"

deps = RAGDeps(
    db=lancedb.connect("./data/database/rag_vector_db"),
    table_name="documents",
    embedder=embedder,
    retrieval_query=retrieval_query,
    doc_id=9,
)

result = rag_agent.run_sync(retrieval_query, deps=deps)

print(result.output)
