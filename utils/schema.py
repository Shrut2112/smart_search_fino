from pydantic import BaseModel
from typing import List, TypedDict, Dict, Optional, Any

class State(TypedDict):
    original_filename: str
    normalized_filename: str
    version: str
    revision_tag: str | None
    is_collision: bool
    confidence: float
    version_detected: bool
    
    # from naming  
    raw_text: str
    chunks: List[Dict[str, Any]]    
    metadata: Dict[str, Any]        
    quality_score: float
    extraction_stats: Dict[str, Any]
    tables: List[str]
    parsing_errors: List[str]
    status: str   # failed / unsupported_format / chunked
    