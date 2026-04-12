from langgraph.graph import StateGraph, START, END
from utils.schema import State
from data_ingestion.universal_naming_agent import universal_name_refiner_v4
from data_ingestion.universal_parser_agent import parser_graph
from data_ingestion.comparison_agent import get_comparison_agent
from database.db_hooks import query_doc_exists
from data_ingestion.graph.process_chunk_func import remove_chunks, push_chunks
from data_ingestion.graph.routing_after_parsing import routing_after_parsing
from data_ingestion.graph.routing_to_parsing import routing_to_parsing
from utils.logger import get_logger

log = get_logger("graph.main")

def db_dedup_check(state: State) -> dict:
    file_hash = state.get("file_hash")
    if not file_hash:
        return {**state, "is_raw_duplicate": False}
    existing = query_doc_exists(file_hash)
    if existing:
        log.info(f"Raw file duplicate detected -> hash={file_hash[:16]}..., existing_doc={existing}")
        return {
            **state,
            "is_raw_duplicate": True,
            "skip_reason": f"raw_file_already_indexed:{existing}",
        }
    return {**state, "is_raw_duplicate": False}

def mark_skipped(state: State) -> dict:
    log.info(f"Pipeline skipped -> reason={state.get('skip_reason', 'duplicate_or_collision')}")
    return {
        **state,
        "status_comp": "skipped",
        "skip_reason": state.get("skip_reason", "duplicate_or_collision"),
    }

def mark_completed(state: State) -> dict:
    log.info(f"Pipeline completed -> doc={state.get('base_doc_name', 'unknown')}")
    return {
        **state,
        "status_comp": "completed",
    }

def main_graph():
    builder = StateGraph(State)
    builder.add_node("naming_agent", universal_name_refiner_v4)
    builder.add_node("db_dedup_check", db_dedup_check)
    builder.add_node("parsing_agent", parser_graph())
    builder.add_node("archive_existing_chunks", remove_chunks)
    builder.add_node("push_all", push_chunks)
    builder.add_node("compare", get_comparison_agent())
    builder.add_node("mark_skipped", mark_skipped)
    builder.add_node("mark_completed", mark_completed)
    builder.add_edge(START, "naming_agent")
    builder.add_conditional_edges(
        "naming_agent",
        routing_to_parsing,
        {
            "parsing_agent": "db_dedup_check",
            "end": "mark_skipped",
        }
    )
    builder.add_conditional_edges(
        "db_dedup_check",
        lambda s: "end" if s.get("is_raw_duplicate") else "parsing_agent",
        {
            "parsing_agent": "parsing_agent",
            "end": "mark_skipped",
        }
    )
    builder.add_conditional_edges(
        "parsing_agent",
        routing_after_parsing,
        {
            "archive_existing_chunks": "archive_existing_chunks",
            "push_all": "push_all",
            "compare": "compare",
            "end": "mark_skipped",
        }
    )
    builder.add_edge("archive_existing_chunks", "push_all")
    builder.add_edge("push_all", "mark_completed")
    builder.add_edge("compare", "mark_completed")
    builder.add_edge("mark_skipped", END)
    builder.add_edge("mark_completed", END)
    return builder.compile()
