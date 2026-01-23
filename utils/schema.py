from pydantic import BaseModel
from typing import List, TypedDict, Dict, Optional, Any, Set

class State(TypedDict):
    original_filename: str
    normalized_filename: str
    version: str
    revision_tag: str | None
    is_collision: bool
    confidence: float
    version_detected: bool
    
    base_doc_name: str
    content_hash: str
    # from naming  
    raw_text: str
    chunks: List[Dict[str, Any]]    
    metadata: Dict[str, Any]        
    quality_score: float
    extraction_stats: Dict[str, Any]
    tables: List[str]
    parsing_errors: List[str]
    status: str   # failed / unsupported_format / chunked
    
    # new_doc_hash: str
    
    old_chunks: List[Dict[str, Any]]
    report: Dict[str, Any] 
    actions: List[str] #skip, store_all, store_changed, generate_report, archive_old, upsert_new
    status_comp: str #complete, chunk_compare, fuzzy_next, db_ready, pending
    matched_old_ids: Set[str]
    to_archive: List[str]  #all_old_ids - matched_old_ids
