from utils.schema import State
from agents.db_hooks import archive_chunk, upsert_chunks, upsert_doc
from agents.db_hooks import get_db_connection
from utils.logger import get_logger

log = get_logger("graph.chunks")

def remove_chunks(state:State):
    doc_id = state['base_doc_name']
    
    try:
        archive_chunk(doc_id)
    except Exception as e:
        log.error(f"Error executing archive chunk: {e}")
    
    return state

def push_chunks(state:State):
    
    chunks = state['chunks']
    doc_name = state['base_doc_name']
    
    try:
        with get_db_connection() as conn:
            log.info(f"Transaction started for {doc_name}")
            upsert_doc(doc_name, state['version'], state['extraction_stats'], state['content_hash'],conn)
            log.info("Step 1: Doc metadata staged in transaction")

            upsert_chunks(chunks,conn)
            log.info(f"Step 2: Staged {len(chunks)} chunks")

            conn.commit()
            log.info("Transaction committed successfully")

    except Exception as e:
        log.error(f"CRITICAL DB ERROR: {e}")
        log.error(f"Rollback triggered for '{doc_name}' — no data saved")
        raise RuntimeError(f"DB write failed: {e}")
        
    return {**state, "pipeline_status": "success"}