# agents/comparison_agent.py

from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from .db_hooks import (
    query_doc_exists, load_latest_chunks, archive_chunk, upsert_chunk
)
from sentence_transformers import SentenceTransformer, util
import os

class ComparisonState(TypedDict):
    new_doc_id: str
    new_doc_hash: str
    new_chunks: List[Dict[str, Any]] 
    old_chunks: List[Dict[str, Any]]
    report: Dict[str, Any] 
    actions: List[str] #skip, store_all, store_changed, generate_report, archive_old, upsert_new
    status: str #complete, chunk_compare, fuzzy_next, db_ready, pending

# Model cache (E5-large)
_MODEL_CACHE = None

def get_embedding_model():
    global _MODEL_CACHE
    if _MODEL_CACHE: 
        return _MODEL_CACHE

    model_path = os.getenv("EMBEDDING_MODEL_PATH", "D:/models/e5-large")
    try:
        _MODEL_CACHE = SentenceTransformer(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load embedding model at {model_path}: {e}")
        
    return _MODEL_CACHE

# NODES 

def check_doc_exists(state: ComparisonState) -> ComparisonState:
    """REAL SHA256 check"""
    if query_doc_exists(state["new_doc_hash"]):
        return {
            **state,
            "old_chunks": [],
            "report": {"status": "duplicate", "action": "skip"},
            "actions": [],
            "status": "complete"
        }
    
    # Load latest version chunks
    # Assumes doc_id format: "name_vX" -> prefix "name"
    doc_prefix = state["new_doc_id"].rsplit("_v", 1)[0]
    old_chunks = load_latest_chunks(doc_prefix)
    
    # Even if old_chunks is empty, flow through to generate correct report
    return {
        **state, 
        "old_chunks": old_chunks or [], 
        "status": "chunk_compare"
    }

def exact_hash_matching(state: ComparisonState) -> ComparisonState:
    """MD5 exact matches (O(1))"""
    old_hashes = {c["text_hash"]: c["chunk_id"] for c in state["old_chunks"]}
    unchanged = 0
    
    for chunk in state["new_chunks"]:
        if chunk["text_hash"] in old_hashes:
            chunk["comparison_status"] = "exact_unchanged"
            chunk["old_chunk_id"] = old_hashes[chunk["text_hash"]]
            unchanged += 1
    
    return {
        **state,
        "report": {"exact_unchanged": unchanged},
        "status": "fuzzy_next"
    }

def cosine_fuzzy_matching(state: ComparisonState) -> ComparisonState:
    """threshold 95% cosine unchanged"""
    model = get_embedding_model()

    fuzzy_new = [c for c in state["new_chunks"] if c.get("comparison_status") != "exact_unchanged"]
    fuzzy_old = [c for c in state["old_chunks"] if c["text_hash"] not in 
                {nc["text_hash"] for nc in state["new_chunks"]}]
    
    if not fuzzy_new:
        return state

    if not fuzzy_old or not model:
        for c in fuzzy_new:
            c["comparison_status"] = "changed_new"
        return state
    
    # Batch embeddings with normalization
    new_embeds = model.encode(
        [c["text"] for c in fuzzy_new],
        convert_to_tensor=True,
        normalize_embeddings=True
    )
    old_embeds = model.encode(
        [c["text"] for c in fuzzy_old],
        convert_to_tensor=True,
        normalize_embeddings=True
    )
    
    fuzzy_unchanged = 0
    for i, chunk in enumerate(fuzzy_new):
        scores = util.cos_sim(new_embeds[i], old_embeds)
        best_score = scores.max().item()

        if best_score >= 0.95: #threshold
            chunk["comparison_status"] = "fuzzy_unchanged"
            fuzzy_unchanged += 1
        else:
            chunk["comparison_status"] = "changed_new"
    
    state["report"]["fuzzy_unchanged"] = fuzzy_unchanged
    return state

def generate_report(state: ComparisonState) -> ComparisonState:
    """Final report + actions"""
    total = len(state["new_chunks"])
    unchanged = len([c for c in state["new_chunks"] if "unchanged" in c.get("comparison_status", "")])
    
    report = {
        "total_new": total,
        "unchanged": unchanged,
        "ratio": round(unchanged/total, 2) if total > 0 else 0.0,
        "savings": round(100*(1-(total-unchanged)/total), 1) if total > 0 else 0.0,
        "action": "store_changed" if unchanged else "store_all"
    }
    
    actions = ["generate_report"]

    if report["action"] == "store_all":
        actions.append("store_all")
    elif state["old_chunks"]:
        actions.extend(["archive_old", "upsert_new"])
    
    return {**state, "report": report, "actions": actions, "status": "db_ready"}

def execute_db_actions(state: ComparisonState) -> ComparisonState:
    """REAL DB operations"""
    for action in state["actions"]:
        if action == "archive_old":
            for chunk in state["old_chunks"]:
                archive_chunk(chunk["chunk_id"])
        elif action == "upsert_new":
            for chunk in state["new_chunks"]:
                if chunk.get("comparison_status") != "exact_unchanged":
                    chunk["metadata"]["doc_id"] = state["new_doc_id"]
                    upsert_chunk(chunk)
        elif action == "store_all":
            for chunk in state["new_chunks"]:
                chunk["metadata"]["doc_id"] = state["new_doc_id"]
                upsert_chunk(chunk)
    
    return {**state, "status": "completed"}

# WORKFLOW

def route_check(state: ComparisonState):
    if state["status"] == "complete":
        return "db_exec"
    return "hash_match"

workflow = StateGraph(ComparisonState)
workflow.add_node("check_doc", check_doc_exists)
workflow.add_node("hash_match", exact_hash_matching)
workflow.add_node("fuzzy_match", cosine_fuzzy_matching)
workflow.add_node("report", generate_report)
workflow.add_node("db_exec", execute_db_actions)

workflow.set_entry_point("check_doc")
workflow.add_conditional_edges("check_doc", route_check, {
    "db_exec": "db_exec",
    "hash_match": "hash_match"
})
workflow.add_edge("hash_match", "fuzzy_match")
workflow.add_edge("fuzzy_match", "report")
workflow.add_edge("report", "db_exec")
workflow.add_edge("db_exec", END)

comparison_agent = workflow.compile()

# Ye niche ka part bas ese hi helper hai jo agent ko bs fixed starting boiler plate deta hai jisse aage sbko vesa hi mile
# Perplexity suggested.

def run_comparison(new_doc_id: str, new_chunks: List[Dict], new_doc_hash: str) -> ComparisonState:
    """Production entry point"""
    result = comparison_agent.invoke({
        "new_doc_id": new_doc_id,
        "new_doc_hash": new_doc_hash,
        "new_chunks": new_chunks,
        "old_chunks": [],
        "report": {},
        "actions": [],
        "status": "pending"
    })
    print(f" Comparison: {result['report']}")
    return result
