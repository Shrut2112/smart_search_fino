# agents/universal_naming_agent.py
from typing import TypedDict, Callable
from pathlib import Path
import re

class NamingState(TypedDict):
    original_filename: str
    normalized_name: str
    version: str
    revision_tag: str | None
    is_collision: bool
    confidence: float
    version_detected: bool

# INFRA INTERFACE (TOP LEVEL - clearly marked)
def collision_exists(filename: str) -> bool:
    """
    INFRA HOOK - Implement in pipeline layer.
    
    Args:
        filename: Proposed normalized filename
        
    Returns:
        True if filename exists in storage (DB/S3/whatever)
        
    DEFAULT STUB: Always safe (no collision)
    """
    return False  # ← INTENTIONAL STUB

def universal_name_refiner_v4(state: NamingState) -> NamingState:
    """Pure agent logic - NO DB DEPENDENCY"""
    filename = Path(state['original_filename']).stem.lower()
    
    # STEP 1: Initialize (clean)
    version = "v1"
    revision_tag = None
    version_detected = False
    clean_name = filename
    
    # STEP 2: Junk stripping
    junk_patterns = [r'^\d+_*', r'\d{8,}', r'_?\d{10,}']
    for pattern in junk_patterns:
        clean_name = re.sub(pattern, '', clean_name)
    
    # STEP 3: REVISION (EXPLICIT FIRST)
    revision_keywords = r'\b(revised|rev|updated|update)\b'
    revision_match = re.search(revision_keywords, clean_name, re.IGNORECASE)
    if revision_match:
        revision_tag = "revised"
        clean_name = re.sub(revision_keywords, '', clean_name, flags=re.IGNORECASE)
    
    # STEP 4: VERSION (SEPARATE)
    version_patterns = [
        r'(?:version?|v)(\d+(?:\.\d+)?)(?=[_\s.\w]|$)',
        r'_?v(\d+(?:\.\d+)?)(?=[_\s.\w]|$)',
    ]
    for pattern in version_patterns:
        match = re.search(pattern, clean_name, re.IGNORECASE)
        if match:
            version = f"v{match.group(1)}"
            version_detected = True
            clean_name = re.sub(pattern, '', clean_name, count=1)
            break
    
    # STEP 5: Snake case
    clean_name = re.sub(r'[^\w\s-]', '', clean_name)
    clean_name = re.sub(r'[\s_-]+', '_', clean_name)
    clean_name = re.sub(r'^_+|_+$', '', clean_name)
    
    # STEP 6: Build base (deterministic)
    base_components = [clean_name, version]
    if revision_tag:
        base_components.append(revision_tag)
    
    base_name = "_".join(base_components)
    normalized_name = f"{base_name}.pdf"
    
    # STEP 7: LENGTH SAFETY
    if len(normalized_name) > 80:
        version_part = f"_{version}{'_revised' if revision_tag else ''}.pdf"
        max_main = 80 - len(version_part)
        clean_name = clean_name[:max_main]
        normalized_name = f"{clean_name}{version_part}"
    
    # STEP 8: COLLISION RESOLUTION (DB-AGNOSTIC)
    counter = 0
    final_name = normalized_name
    
    # 👈 INTERFACE CALL (Infra provides implementation)
    while collision_exists(final_name) and counter < 999:
        counter += 1
        suffix = f"_{counter:03d}"
        final_name = f"{base_name}{suffix}.pdf"
    
    is_collision = counter > 0
    
    # STEP 9: Confidence (explicit)
    confidence = 1.0
    if revision_tag:
        confidence *= 0.98
    if is_collision:
        confidence *= 0.95
    
    return {
        **state,
        "normalized_name": final_name,
        "version": version,
        "revision_tag": revision_tag,
        "version_detected": version_detected,
        "is_collision": is_collision,
        "confidence": confidence
    }
