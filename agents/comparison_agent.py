# agents/comparison_agent.py - PRODUCTION READY v2.1

from typing import TypedDict, List, Dict, Any, Set
from langgraph.graph import StateGraph, END
from .db_hooks import (
    query_doc_exists, load_latest_chunks, archive_chunk, upsert_chunks
)
from sentence_transformers import SentenceTransformer, util
import os
from utils.schema import State

# class ComparisonState(TypedDict):
#     new_doc_id: str
#     new_doc_hash: str
#     new_chunks: List[Dict[str, Any]] 
#     old_chunks: List[Dict[str, Any]]
#     report: Dict[str, Any]
#     actions: List[str]
#     status: str
#     matched_old_ids: Set[str]
#     to_archive: List[str]  #all_old_ids - matched_old_ids

# Model cache
_MODEL_CACHE = None

def get_embedding_model():
    global _MODEL_CACHE
    if _MODEL_CACHE: 
        return _MODEL_CACHE

    model_path = os.getenv("EMBEDDING_MODEL_PATH", "D:/models/e5-large")
    try:
        _MODEL_CACHE = SentenceTransformer(model_path)
    except Exception as e:
        print(f"Fallback to all-MiniLM-L6-v2: {e}")
        _MODEL_CACHE = SentenceTransformer('all-MiniLM-L6-v2')
    return _MODEL_CACHE

def check_doc_exists(state: State) -> State:
    #SHA256
    if query_doc_exists(state["content_hash"]):
        return {
            **state,
            "status_comp": "complete",
            "report": {"status": "duplicate", "action": "skip"},
            "actions": [],
            "matched_old_ids": set(),
            "to_archive": []
        }
    
    doc_prefix = state["base_doc_name"]
    old_chunks = load_latest_chunks(doc_prefix)
    
    return {
        **state,
        "old_chunks": old_chunks or [],
        "matched_old_ids": set(),
        "to_archive": [],
        "report": {"status": "new_version"},
        "actions": [],
        ### if not duplicate 
        "status_comp": "chunk_compare" if old_chunks else "store_all"
    }

def exact_hash_matching(state: State) -> State:
    """MD5 exact matches"""
    old_hashes = {c["text_hash"]: c["chunk_id"] for c in state["old_chunks"]}
    matched_old = set()
    
    for chunk in state["chunks"]:
        if chunk["text_hash"] in old_hashes:
            chunk["comparison_status"] = "exact_unchanged"
            chunk["old_chunk_id"] = old_hashes[chunk["text_hash"]]
            matched_old.add(chunk["old_chunk_id"])
    
    return {
        **state,
        "matched_old_ids": matched_old,
        "report": {"exact_unchanged": len(matched_old)},
        "status_comp": "fuzzy_match"
    }

def cosine_fuzzy_matching(state: State) -> State:
    model = get_embedding_model()
    fuzzy_new = [c for c in state["chunks"] if c.get("comparison_status") != "exact_unchanged"]
    fuzzy_old = [c for c in state["old_chunks"] 
                if c["chunk_id"] not in state["matched_old_ids"]]
    
    if not fuzzy_new or not fuzzy_old or not model:
        return _calculate_archive(state)
    
    #BATCH COSINE
    new_embeds = model.encode([c["text"] for c in fuzzy_new], 
                            convert_to_tensor=True, normalize_embeddings=True)
    old_embeds = model.encode([c["text"] for c in fuzzy_old], 
                            convert_to_tensor=True, normalize_embeddings=True)
    
    matched_old = state["matched_old_ids"].copy()
    fuzzy_count = 0
    
    for i, chunk in enumerate(fuzzy_new):
        scores = util.cos_sim(new_embeds[i:i+1], old_embeds)  # [1,N] matrix
        best_idx = scores[0].argmax().item()  # Index in fuzzy_old
        best_score = scores[0][best_idx].item()
        
        if best_score >= 0.95:
            old_chunk_id = fuzzy_old[best_idx]["chunk_id"]

            if old_chunk_id not in matched_old:  # One-to-one
                chunk["comparison_status"] = "fuzzy_unchanged"
                chunk["old_chunk_id"] = old_chunk_id
                chunk["similarity"] = round(best_score, 3)
                matched_old.add(old_chunk_id)
                fuzzy_count += 1
        
        # Default if not matched: new chunk
        if chunk.get("comparison_status") != "fuzzy_unchanged":
            chunk["comparison_status"] = "new_chunk"
    
    return _calculate_archive({
        **state, 
        "matched_old_ids": matched_old,
        "report": {**state.get("report", {}), "fuzzy_unchanged": fuzzy_count}
    })

def _calculate_archive(state: State) -> State:
    """Archive unmatched old"""
    all_old_ids = {c["chunk_id"] for c in state["old_chunks"]}
    matched_ids = state["matched_old_ids"]
    to_archive = list(all_old_ids - matched_ids)
    
    return {
        **state,
        "to_archive": to_archive,
        "report": {
            **state["report"],
            "total_old": len(all_old_ids),
            "matched_old": len(matched_ids),
            "to_archive": len(to_archive)
        },
        "status_comp": "ready_archive"
    }

def generate_report(state: State) -> State:
    new_total = len(state["chunks"])
    unchanged = len([c for c in state["chunks"] if "unchanged" in c.get("comparison_status", "")])
    new_only = new_total - unchanged
    total_old = state["report"].get("total_old", 0)
    
    savings_pct = round(100 * state["report"]["matched_old"] / total_old, 1) if total_old > 0 else 0.0
    
    report = {
        "total_new": new_total,
        "unchanged": unchanged,
        "new_added": new_only,
        "old_archived": state["report"]["to_archive"],
        **state["report"],
        "storage_savings": f"{savings_pct}%",
        "action": "sync_complete"
    }
    
    actions = ["archive_old", "upsert_new"] if state.get("old_chunks") else ["upsert_new"]
    return {**state, "report": report, "actions": actions, "status_comp": "db_ready"}

def execute_db_actions(state: State) -> State:
    """Execute archive + upsert"""
   # 1. Archive old chunks first
    for old_chunk_id in state["to_archive"]:
        archive_chunk(old_chunk_id)
    
    # 2. Filter only chunks that need saving (exclude exact matches)
    chunks_to_save = []
    for chunk in state["chunks"]:
        status = chunk.get("comparison_status", "new_chunk")
        if status != "exact_unchanged":
            # Ensure the metadata has the base doc name
            chunk["metadata"]["doc_id"] = state["base_doc_name"]
            chunks_to_save.append(chunk)
    
    # 3. Call upsert_chunks ONCE with the full list
    if chunks_to_save:
        upsert_chunks(chunks_to_save)
        print(f"STORED {len(chunks_to_save)} chunks successfully.")
    
    return {**state, "status_comp": "completed"}


# WORKFLOW

def get_comparison_agent(state:State):
    workflow = StateGraph(State)
    workflow.add_node("check_doc", check_doc_exists)
    workflow.add_node("hash_match", exact_hash_matching)
    workflow.add_node("fuzzy_match", cosine_fuzzy_matching)
    workflow.add_node("report", generate_report)
    workflow.add_node("db_exec", execute_db_actions)

    workflow.set_entry_point("check_doc")
    workflow.add_edge("check_doc", "hash_match")
    workflow.add_edge("hash_match", "fuzzy_match")
    workflow.add_edge("fuzzy_match", "report")
    workflow.add_edge("report", "db_exec")
    workflow.add_edge("db_exec", END)

    comparison_agent = workflow.compile()
    return comparison_agent

# def run_comparison(new_doc_id: str, chunks: List[Dict], content_hash: str) -> State:
#     result = comparison_agent.invoke({
#         "base_doc_name": new_doc_id,
#         "content_hash": content_hash,
#         "chunks": chunks,
#         "old_chunks": [],
#         "report": {},
#         "actions": [],
#         "status_comp": "pending",
#         "matched_old_ids": set(),
#         "to_archive": []
#     })
#     print(f"RESULT: {result['report']}")
#     return result