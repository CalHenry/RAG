from src.rag.data_models import RAGDeps


async def retrieve(deps: RAGDeps, retrieval_query: str, doc_id: int) -> list[dict]:
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
        .select(["chunk_text", "publish_date", "_distance"])
        .to_list()
    )
    return results
