from utils.schema import State
from agents.db_hooks import archive_chunk, upsert_chunks, upsert_doc
from agents.db_hooks import get_db_connection

def remove_chunks(state:State):
    doc_id = state['base_doc_name']
    
    try:
        archive_chunk(doc_id)
    except Exception as e:
        print(f"Error Executing archieve Chunk: {e}")
    
    return state

def push_chunks(state:State):
    
    chunks = state['chunks']
    doc_name = state['base_doc_name']
    
    try:
        with get_db_connection() as conn:
            # In many drivers, 'with conn' starts the transaction context
            print(f"--- Transaction Started for {doc_name} ---")
            # 1. Attempt Doc Upsert
            upsert_doc(doc_name, state['version'], state['extraction_stats'], state['content_hash'],conn)
            print("Step 1: Doc metadata 'pending' in transaction.")

            # 2. Attempt Chunk Upsert (This is where the NUL/ti error usually happens)
            upsert_chunks(chunks,conn)
            print(f"Step 2: Successfully staged {len(chunks)} chunks.")

            # If we reach here, 'with conn' will automatically COMMIT
            conn.commit()
            print("--- Transaction Committed Successfully ---")

    except Exception as e:
        # If we are here, the 'with conn' block has already triggered a ROLLBACK
        print(f"!!! CRITICAL ERROR: {e}")
        print(f"--- ROLLBACK TRIGGERED AUTOMATICALLY ---")
        print(f"Result: No data for '{doc_name}' was saved to the database.")
        
        # Optional: You can explicitly check connection status if the driver supports it
        # e.g., if conn.closed: print("Connection closed after rollback.")
        
    return state