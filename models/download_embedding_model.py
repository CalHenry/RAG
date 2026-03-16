import os

from sentence_transformers import SentenceTransformer

from config import MODEL_NAME, MODEL_PATH

# SentenceTransformer model will be used to embedd the documents and the user query
# Model's HF page: https://huggingface.co/BAAI/bge-large-zh-v1.5


save_path = MODEL_PATH

if os.path.exists(save_path) and os.listdir(save_path):
    print(f"Model '{MODEL_NAME}' is already downloaded")
else:
    try:
        model = SentenceTransformer(MODEL_NAME)
        model.save(save_path)

        if os.path.exists(save_path) and os.listdir(save_path):
            print(f"'{MODEL_NAME}' is successfully downloaded")
        else:
            print(f"Failed to save model to '{save_path}'")

    except Exception as e:
        print(f"An error occurred: {e}")
