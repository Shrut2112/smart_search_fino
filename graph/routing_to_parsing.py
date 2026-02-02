from agents.db_hooks import check_doc_with_name_version
from utils.schema import State

def routing_to_parsing(state:State):
    revision_tag = state['revision_tag']
    if revision_tag is None:
        filename = state['base_doc_name']
        version_no = state['version']
        is_present = check_doc_with_name_version(filename,version_no)
        
        if is_present:
            return "end"
        else:
            return "parsing_agent"
    return "parsing_agent"