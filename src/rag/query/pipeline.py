import os

from sentence_transformers import SentenceTransformer

from config import MODEL_NAME, MODEL_PATH, RETRIEVAL_QUERY
from src.rag.data_models import RAGDeps
from src.rag.query.agent import rag_agent


def run_pipeline() -> None:
    # Embedder setup ------------------------------------------------------------
    use_local = True
    if not (MODEL_PATH and os.path.isdir(MODEL_PATH) and os.listdir(MODEL_PATH)):
        use_local = False

    embedder = SentenceTransformer(
        MODEL_PATH if use_local else MODEL_NAME, local_files_only=use_local
    )

    #  Run the agent ------------------------------------------------------------
    deps = RAGDeps.create(embedder, RETRIEVAL_QUERY, doc_id=list(range(1, 26)))

    result = rag_agent.run_sync(RETRIEVAL_QUERY, deps=deps)

    print(result.output)


if __name__ == "__main__":
    run_pipeline()
