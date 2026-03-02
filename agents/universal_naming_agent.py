# agents/universal_naming_agent.py
from typing import TypedDict, Callable
from pathlib import Path
import re
from utils.schema import State
from utils.logger import get_logger

log = get_logger("agent.naming")

def collision_exists(filename: str) -> bool:
    """
    INFRA HOOK - Implement in pipeline layer.
    
    Args:
        filename: Proposed normalized filename
        
    Returns:
        True if filename exists in storage (DB/S3/whatever)
        
    DEFAULT STUB: Always safe (no collision)
    """
    return False  

def universal_name_refiner_v4(state: State) -> State:
    """Pure agent logic - NO DB DEPENDENCY"""
    filename = Path(state['original_filename']).stem.lower()

    # STEP 1: Initialize
    version = "v1"
    revision_tag = None
    version_detected = False
    clean_name = filename

    # STEP 2: Junk stripping
    junk_patterns = [r'^\d+_*', r'\d{8,}', r'_?\d{10,}']
    for pattern in junk_patterns:
        clean_name = re.sub(pattern, '', clean_name)

    # STEP 3: REVISION (ROBUST)
    revision_patterns = [
        r'^revised[_\s-]?',
        r'^rev[_\s-]?',
        r'^updated[_\s-]?',
        r'[_\s-](revised|rev|updated|update)(?=[_\s-]|$)',
    ]

    for pattern in revision_patterns:
        if re.search(pattern, clean_name, re.IGNORECASE):
            revision_tag = "revised"
            clean_name = re.sub(pattern, '', clean_name, flags=re.IGNORECASE)
            break

    # STEP 4: VERSION (WITH ZERO NORMALIZATION)
    version_patterns = [
        r'(?:^|[_\s-])(?:version?|v)(\d+(?:\.\d+)?)(?=[_\s-]|$)',
    ]

    for pattern in version_patterns:
        match = re.search(pattern, clean_name, re.IGNORECASE)
        if match:
            raw_version = match.group(1)

            if "." in raw_version:
                major, minor = raw_version.split(".", 1)
                version = f"v{int(major)}.{minor}"
            else:
                version = f"v{int(raw_version)}"

            version_detected = True
            clean_name = re.sub(pattern, '', clean_name, count=1)
            break

    # STEP 5: Snake case normalization
    clean_name = re.sub(r'[^\w\s-]', '', clean_name)
    clean_name = re.sub(r'[\s_-]+', '_', clean_name)
    clean_name = re.sub(r'^_+|_+$', '', clean_name)

    # STEP 6: Build base name
    base_components = [clean_name, version]
    if revision_tag:
        base_components.append(revision_tag)

    base_name = "_".join(filter(None, base_components))
    normalized_name = f"{base_name}.pdf"

    # STEP 7: Length safety
    if len(normalized_name) > 80:
        suffix = f"_{version}{'_revised' if revision_tag else ''}.pdf"
        max_main = 80 - len(suffix)
        clean_name = clean_name[:max_main]
        normalized_name = f"{clean_name}{suffix}"

    # STEP 8: Collision handling
    counter = 0
    final_name = normalized_name
    while collision_exists(final_name) and counter < 999:
        counter += 1
        final_name = f"{base_name}_{counter:03d}.pdf"

    is_collision = counter > 0

    # STEP 9: Confidence
    confidence = 1.0
    if revision_tag:
        confidence *= 0.98
    if is_collision:
        confidence *= 0.95

    log.info(f"Naming: '{Path(state['original_filename']).name}' -> base='{clean_name}', version={version}, revision={revision_tag}, collision={is_collision}, final='{final_name}'")

    return {
        **state,
        "normalized_filename": final_name,
        "base_doc_name": clean_name,
        "version": version,
        "revision_tag": revision_tag,
        "version_detected": version_detected,
        "is_collision": is_collision,
        "confidence": confidence,
    }
