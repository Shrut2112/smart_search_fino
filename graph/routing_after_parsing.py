from utils.schema import State
from agents.db_hooks import check_doc_present, query_doc_exists
from utils.logger import get_logger

log = get_logger("graph.routing_after_parsing")

def routing_after_parsing(state: State):
    
    #FIX: if parser failed, exit immediately
    if state.get("status") in ("failed", "unsupported") or not state.get("content_hash"):
        log.info(f"Routing after parsing -> end (status={state.get('status')})")
        return "end"

    revision_tag = state.get('revision_tag')
    filename = state.get('base_doc_name')
    version_no = state.get('version')
    content_hash = state.get('content_hash')

    if content_hash and query_doc_exists(content_hash):
        log.info(f"Duplicate content detected for {filename}. Routing -> end")
        return "end"

    if revision_tag is not None:
        log.info(f"Routing after parsing -> compare (revision={revision_tag})")
        return "compare"

    if check_doc_present(filename, version_no):
        log.info(f"Routing after parsing -> archive_existing_chunks (existing version found)")
        return "archive_existing_chunks"

    log.info(f"Routing after parsing -> push_all (new document)")
    return "push_all"