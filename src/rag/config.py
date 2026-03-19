import os
from pathlib import Path

from dotenv import load_dotenv

# Embedding model -----------------------------------------------------------
MODEL_PATH = "./models/bge-large-zh-v1.5"
MODEL_NAME = "BAAI/bge-large-zh-v1.5"

# query pipeline  --------------------------------------------------
load_dotenv()

USE_OPENROUTER = os.getenv("USE_OPENROUTER", "false").lower() == "true"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # None if not set


BASE = Path(__file__).parent.parent.parent  # := ./src/rag/

# Key words to match against the vectors in the vector database. Also added to the system prompt
RETRIEVAL_QUERY = "énergie nucléaire, centrale nucléaire, réacteur atomique"
FAILED_IDS_PATH = BASE / "data" / "failed_ids.txt"

# interim outputs (inputs to merge.py)
VECTOR_OUTPUT_PATH = BASE / "data" / "interim" / "vector_db_retrieval.parquet"
AI_OUTPUT_PATH = BASE / "data" / "interim" / "ai_processed.parquet"
# final merged output
FINAL_OUTPUT_PATH = BASE / "data" / "processed" / "merged.parquet"

# Vector database -----------------------------------------------------------
DB_PATH = BASE / "data" / "database" / "rag_vector_db"
TABLE_NAME = "documents"
