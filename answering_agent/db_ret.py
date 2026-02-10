from agents.db_hooks import get_db_connection
from utils.queries import sql_query, keyword_query

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