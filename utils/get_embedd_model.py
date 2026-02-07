from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
def embedding_model():
    load_dotenv()
    E5_MODEL_PATH = os.getenv(
            "EMBEDDING_MODEL_PATH",
            r"D:\models\e5-large"
        )

    model_kwargs = {
            "device": "cpu",
            "prompts": {
                "query": "query: ",
                "passage": "passage: "
            }
    }

    embeddings = HuggingFaceEmbeddings(
        model_name=E5_MODEL_PATH,
        model_kwargs=model_kwargs,
        encode_kwargs={
            "batch_size": 64,
            "normalize_embeddings": True
        }
    )
    print("Embedding model initialized")
    return embeddings
        