import os
import time
from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings
from utils.logger import get_logger

log = get_logger("utils.embedding")

class FreeTierCohereEmbeddings(CohereEmbeddings):
    """
    A custom wrapper around CohereEmbeddings designed specifically to survive
    the Cohere Free Tier Rate Limits by chunking massive document lists and 
    adding deliberate pacing to avoid 429 Too Many Requests errors.
    """
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Cohere API maximal batch size is 96. We use 90 for safety.
        batch_size = 90
        all_embeddings = []
        
        if len(texts) > batch_size:
            log.info(f"FreeTier Wrapper: Processing {len(texts)} chunks in batches of {batch_size}. This will take some time due to rate limits...")
            
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Use parent's method which internally handles the actual API request
            batch_emb = super().embed_documents(batch)
            all_embeddings.extend(batch_emb)
            
            # Sleep aggressively to respect ~10 requests/min trial limit
            if i + batch_size < len(texts):
                log.info(f"Processed {i + batch_size}/{len(texts)} chunks. Sleeping 12s to avoid Free Tier rate limits...")
                time.sleep(12) 
                
        return all_embeddings

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        # Same logic for async pipelines
        import asyncio
        batch_size = 90
        all_embeddings = []
        
        if len(texts) > batch_size:
            log.info(f"FreeTier Wrapper (Async): Processing {len(texts)} chunks in batches of {batch_size}...")
            
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            batch_emb = await super().aembed_documents(batch)
            all_embeddings.extend(batch_emb)
            
             # Sleep aggressively to respect ~10 requests/min trial limit
            if i + batch_size < len(texts):
                log.info(f"Processed {i + batch_size}/{len(texts)} chunks. Sleeping 12s to avoid Free Tier rate limits...")
                await asyncio.sleep(12) 
                
        return all_embeddings

def embedding_model():

    load_dotenv()

    api_key = os.getenv("COHERE_API_KEY")

    if not api_key:
        log.error("COHERE_API_KEY not found in environment")
        return None

    try:

        embeddings = FreeTierCohereEmbeddings(
            model="embed-multilingual-v3.0",
            cohere_api_key=api_key
        )

        log.info("Cohere v3.0 multilingual embedding model initialized (1024 dimensions) with Free Tier Rate Limiter")

        return embeddings

    except Exception as e:

        log.error(f"Error loading Cohere embedding model: {e}")

        return None