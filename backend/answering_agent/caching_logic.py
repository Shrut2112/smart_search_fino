import hashlib
from utils.schema import AnswerState
from database.db_ret import check_hash, check_semantic, update_cache
from utils.get_embedd_model import embedding_model
from datetime import datetime, timezone, timedelta
import asyncio
from utils.logger import get_logger

log = get_logger("answering_agent.caching")


embed = embedding_model()


def isFreah(create_at):
    now_utc = datetime.now(timezone.utc)
    
    if now_utc - create_at < timedelta(hours=24):
        return True
    
    return False

async def cache_check_node(state: AnswerState):
    user_query = state.get('query').lower().strip()
    language = state.get('language', 'en').lower().strip()
    
    cache_key = f"{user_query}_{language}"
    query_hash = hashlib.sha256(cache_key.encode()).hexdigest()
    
    loop = asyncio.get_event_loop()
    match = await loop.run_in_executor(None, check_hash, query_hash)
    
    if match and isFreah(match[1]):
        return {
            "answer": match[0],
            "cache_hit": True,
            "attempt_count": 0,          # Reset so next query starts clean
            "retrived_sem_doc": [],
            "retrived_key_doc": [],
            "reranked_docs": [],
            "final_doc": []
        }
    
    query_vector = await embed.aembed_query(user_query)
    
    if not match:
        match = await loop.run_in_executor(None, check_semantic, query_vector, language)
        
        if match and isFreah(match[1]):
            return {
                "answer": match[0],
                "cache_hit": True,
                "attempt_count": 0,      # Reset so next query starts clean
                "retrived_sem_doc": [],
                "retrived_key_doc": [],
                "reranked_docs": [],
                "final_doc": []
            }
    
    return {
        "cache_hit": False, 
        "query_hash": query_hash, 
        "query_embedding": query_vector,
        "attempt_count": 0,
        "retrived_sem_doc": [],   # ← Clear old accumulated docs
        "retrived_key_doc": [],   # ← Clear old accumulated docs
        "reranked_docs": [],
        "final_doc": []
    }


async def push_cache(state: AnswerState):
    """Internal helper to handle the actual DB write in the background."""
    user_query = state.get('query', "").lower().strip()
    language = state.get('language', 'en').lower().strip()
    answer = state.get('answer')
    embedding_vector = state.get('query_embedding')

    cache_key = f"{user_query}_{language}"
    query_hash = hashlib.sha256(cache_key.encode()).hexdigest()

    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, update_cache, query_hash, user_query, answer, embedding_vector, language)
        log.info("Cache Pushed in DB")
    except Exception as e:
        log.error(f"Error While pushing {e}")


    return {}        