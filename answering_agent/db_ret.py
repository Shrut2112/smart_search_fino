from agents.db_hooks import get_db_connection
from utils.queries import sql_query, keyword_query
from utils.logger import get_logger

log = get_logger("answering.db_ret")

def retrieve_similar_chunks(query_embedding, top_k=10):
    
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql_query, (query_embedding, query_embedding, top_k))
            results_semantic = cur.fetchall()
            log.debug(f"Semantic search returned {len(results_semantic)} results")
            return results_semantic
def retrieve_similar_chunks_key(query, top_k=5):
    
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(keyword_query, (query, query, top_k))
            results_semantic = cur.fetchall()
            log.debug(f"Keyword search returned {len(results_semantic)} results")
            return results_semantic