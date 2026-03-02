from agents.db_hooks import check_doc_with_name_version
from utils.schema import State
from utils.logger import get_logger

log = get_logger("graph.routing_to_parsing")

def routing_to_parsing(state: State):
    revision_tag = state.get("revision_tag")

    if revision_tag is None:
        filename = state.get("base_doc_name")
        version_no = state.get("version")

        if check_doc_with_name_version(filename, version_no):
            return "parsing_agent"

        return "parsing_agent"

    log.debug(f"Routing to parsing -> parsing_agent (revision_tag={revision_tag})")
    return "parsing_agent"