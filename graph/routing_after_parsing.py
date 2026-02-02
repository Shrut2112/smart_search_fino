from utils.schema import State
from agents.db_hooks import check_doc_present, query_doc_exists

def routing_after_parsing(state:State):
    revision_tag = state['revision_tag']
    filename = state['base_doc_name']
    version_no = state['version']
    content_hash = state.get('content_hash') # Ensure hash is in your state

    # 1. NEW: Check if this exact content is already in the system
    if content_hash and query_doc_exists(content_hash):
        print(f"Duplicate content detected for {filename}. Skipping to END.")
        return "end" # Or whatever your 'Finish' node is named
    if revision_tag is not None:
        return "compare"
    
    if check_doc_present(filename, version_no):
        return "archive_existing_chunks"
     
    return "push_all"
