from pathlib import Path

# Embedding model -----------------------------------------------------------
MODEL_PATH = "./models/bge-large-zh-v1.5"
MODEL_NAME = "BAAI/bge-large-zh-v1.5"

# Key words to match against the vectors in the vector database. Also added to the system prompt
RETRIEVAL_QUERY = "énergie nucléaire, centrale nucléaire, réacteur atomique"


# Vector database -----------------------------------------------------------
DB_PATH = Path(__file__).parent.parent.parent / "data" / "database" / "rag_vector_db"
TABLE_NAME = "documents"

# query pipeline variables --------------------------------------------------
FAILED_IDS_PATH = Path(__file__).parent.parent.parent / "data" / "failed_ids.txt"
