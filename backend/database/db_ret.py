from database.db_hooks import get_db_connection
from utils.queries import sql_query, keyword_query, hash_check_query, semantic_cache_query, insert_cache
from datetime import datetime, timezone
def retrieve_similar_chunks(query_embedding, top_k=10):
    
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql_query, (query_embedding, query_embedding, top_k))
            results_semantic = cur.fetchall()
            return results_semantic
def retrieve_similar_chunks_key(query, top_k=5):
    
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(keyword_query, (query, query, top_k))
            results_semantic = cur.fetchall()
            return results_semantic

def check_hash(hash_query):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(hash_check_query,(hash_query,))
            result = cur.fetchone()
            if result:
                conn.commit()
            return result

def check_semantic(query,language):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(semantic_cache_query,(query, language,query))
            result = cur.fetchone()
            if result:
                conn.commit()
            return result
        
def update_cache(query_hash, user_query, answer, embedding_vector):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(insert_cache,(query_hash, user_query, answer, embedding_vector))
            conn.commit()