from src.rag.data_models import RAGDeps


async def retrieve(deps: RAGDeps, query: str) -> list[dict]:
    # embed the query with the same model used at ingestion
    vector = deps.embedder.encode(query).tolist()

    table = deps.db.open_table(deps.table_name)
    results = (
        table.search(vector)
        .where(
            "id_doc = 9"
        )  # This has to change - ok for testing, not ok for production
        .limit(deps.top_k)
        .select(["chunk_text", "publish_date", "_distance"])
        .to_list()
    )
    return results
