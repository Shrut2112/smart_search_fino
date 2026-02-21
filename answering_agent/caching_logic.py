import hashlib
from utils.schema import AnswerState
from answering_agent.db_ret import check_hash, check_semantic, update_cache
from utils.get_embedd_model import embedding_model
from datetime import datetime, timezone, timedelta

embed = embedding_model()

def isFreah(create_at):
    now_utc = datetime.now(timezone.utc)
    
    if now_utc - create_at < timedelta(hours=24):
        return True
    
    return False

def cache_check_node(state:AnswerState):
    user_query = state.get('query').lower().strip()
    query_hash = hashlib.sha256(user_query.encode()).hexdigest()
    
    match = check_hash(query_hash)
    
    if match and isFreah(match[1]):
        return {"answer": match[0], "cache_hit": True}
    
    query_vector = embed.embed_query(user_query)
    if not match:
        match = check_semantic(query_vector)
        if match and isFreah(match[1]):
            return {"answer": match[0], "cache_hit": True}
    
    return {"cache_hit": False, "query_hash": query_hash, "query_embedding": query_vector}

def push_cache(state:AnswerState):
    query_hash = state.get('query_hash')
    user_query = state.get('query')
    answer = state.get('answer')
    embedding_vector = state.get('query_embedding')
    
    try:
        update_cache(query_hash, user_query, answer, embedding_vector)
        print("Cache Pushed in DB")
    except Exception as e:
        print(f"Error While pushing {e}")
    
    return {}