from utils.schema import State
from agents.db_hooks import archive_chunk, upsert_chunks, upsert_doc

def remove_chunks(state:State):
    doc_id = state['base_doc_name']
    
    try:
        archive_chunk(doc_id)
    except Exception as e:
        print(f"Error Executing archieve Chunk: {e}")
    
    return state

def push_chunks(state:State):
    
    chunks = state['chunks']
    
    try:
        upsert_doc(state['base_doc_name'],state['version'],state['extraction_stats'],state['content_hash'])
        upsert_chunks(chunks)
    except Exception as e:
        print(f"Error Executing doc or chunk upsert: {e}")
    
    return state