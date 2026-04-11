import hashlib
from utils.schema import AnswerState
from database.db_ret import check_hash, check_semantic, update_cache
from utils.get_embedd_model import embedding_model
from datetime import datetime, timezone, timedelta
import asyncio

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
        return {"answer": match[0], "cache_hit": True}
    
    query_vector = await embed.aembed_query(user_query)
    
    if not match:
        match = await loop.run_in_executor(None, check_semantic, query_vector, language)
        
        if match and isFreah(match[1]):
            return {"answer": match[0], "cache_hit": True}
    
    return {
        "cache_hit": False, 
        "query_hash": query_hash, 
        "query_embedding": query_vector
    }
    user_query = state.get('query').lower().strip()
    query_hash = hashlib.sha256(user_query.encode()).hexdigest()
    
    loop = asyncio.get_event_loop()
    match = await loop.run_in_executor(None, check_hash, query_hash)
    
    if match and isFreah(match[1]):
        return {"answer": match[0], "cache_hit": True}
    
    # Use async embed
    query_vector = await embed.aembed_query(user_query)
    if not match:
        match = check_semantic(query_vector)
        if match and isFreah(match[1]):
            return {"answer": match[0], "cache_hit": True}
    
    return {"cache_hit": False, "query_hash": query_hash, "query_embedding": query_vector}

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
        print("Cache Pushed in DB")
    except Exception as e:
        print(f"Error While pushing {e}")

    return {}        