# agents/universal_parser_agent.py

from typing import TypedDict, List, Dict, Any, Tuple
from pathlib import Path
import fitz  # PyMuPDF
import pandas as pd
import hashlib
import re
import os
from datetime import datetime
from langgraph.graph import StateGraph, END
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
import io

# Optional OCR 
try:
    import pytesseract
    from PIL import Image
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

# =============================================================================
# STATE (Naming → Parser → Comparison)
# =============================================================================

class ParserState(TypedDict):
    file_path: str
    normalized_filename: str       # from naming  
    raw_text: str
    chunks: List[Dict[str, Any]]    
    metadata: Dict[str, Any]        
    quality_score: float
    extraction_stats: Dict[str, Any]
    tables: List[str]
    parsing_errors: List[str]
    status: str   # failed / unsupported_format / chunked

def log_extraction_metrics(metrics: Dict):
    """INFRA HOOK: LangSmith/Prometheus - pipeline provides"""
    pass

# =============================================================================
# LAYER 1-3: EXTRACTION ENGINE (Fixed)
# =============================================================================

def extract_structure_fixed(state: ParserState) -> ParserState:
    """Blocks → Tables → Fallback """
    file_path = Path(state['file_path'])
    
    if not file_path.exists():
        return {
            **state,
            "status": "failed",
            "parsing_errors": [f"File not found: {file_path}"],
            "chunks": [],
            "quality_score": 0.0
        }
    
    if file_path.suffix.lower() != '.pdf':
        return {
            **state,
            "status": "unsupported_format",
            "parsing_errors": [f"Unsupported format: {file_path.suffix}"],
            "chunks": []
        }
    
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        return {
            **state,
            "status": "failed",
            "parsing_errors": [f"PDF open failed: {str(e)}"],
            "chunks": []
        }

    text_blocks = []
    tables_content = []
    extraction_stats = {
        "total_pages": len(doc),
        "text_blocks": 0,
        "table_blocks": 0,
        "raw_chars": 0,
        "ocr_pages": 0,
        "extraction_timestamp": datetime.utcnow().isoformat()
    }
    
    full_text_parts = []
    low_quality_pages = 0
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # LAYER 1: BLOCK ANALYSIS
        blocks = page.get_text("blocks")
        page_text_content = []
        has_text = False
        
        for block in blocks:
            x0, y0, x1, y1, block_text, _, block_type = block
            if block_text.strip():
                if block_type == 0:  # Text
                    page_text_content.append(block_text)
                    text_blocks.append(block_text)
                    extraction_stats["text_blocks"] += 1
                    has_text = True
                else:  # Table/graphics
                    rect = fitz.Rect(x0, y0, x1, y1)
                    table_text = page.get_text("text", clip=rect)
                    if table_text.strip():
                        formatted_table = f"\n[TABLE]{table_text.strip()}[/TABLE]\n"
                        page_text_content.append(formatted_table)
                        tables_content.append(formatted_table)
                        extraction_stats["table_blocks"] += 1
        
        # LAYER 2: PyMuPDF TABLE DETECTION
        tables = page.find_tables()
        for table in tables:
            try:
                df = pd.DataFrame(table.extract())
                if not df.empty:
                    table_str = df.to_string(index=False, header=False)
                    formatted_table_data = f"\n[TABLE-DATA]\n{table_str}\n[/TABLE-DATA]"
                    page_text_content.append(formatted_table_data)
                    tables_content.append(formatted_table_data)
                    extraction_stats["table_blocks"] += 1
            except:
                pass
        
        # LAYER 3: RAW FALLBACK
        raw_text = page.get_text()
        if raw_text.strip() and len(raw_text) > 50:
            extraction_stats["raw_chars"] += len(raw_text)
        
        if not has_text or len(raw_text.strip()) < 50:
            low_quality_pages += 1
            
        full_text_parts.append("\n".join(page_text_content))
    
    doc.close()
    
    full_text = "\n\n=== PAGE BREAK ===\n\n".join(full_text_parts)
    needs_ocr = len(doc) > 0 and (low_quality_pages / len(doc)) > 0.2
    
    return {
        **state,
        "text_blocks": text_blocks,
        "tables": tables_content,
        "raw_text": full_text,
        "extraction_stats": extraction_stats,
        "needs_ocr": needs_ocr,
        "status": "extracted"
    }

# =============================================================================
# LAYER 4: CLEANING + METADATA (SINGLE SOURCE OF TRUTH)
# =============================================================================

def clean_text_fixed(state: ParserState) -> ParserState:
    """ normalized_filename = doc_id (NO HASHING)"""
    text = state["raw_text"]
    
    # Fino-specific cleaning
    text = re.sub(r'Classification: Internal.*?(?=\n{2,}|\Z)', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'FINO Payments Bank Limited\s*\n?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n{5,}', '\n\n\n', text)
    text = re.sub(r'\s{4,}', ' ', text)
    
    # Update stats
    state["extraction_stats"]["total_chars_post_clean"] = len(text)
    state["extraction_stats"]["total_words_post_clean"] = len(text.split())
    
    #  SINGLE SOURCE OF TRUTH
    doc_id = state["normalized_filename"]  
    
    #this is doc hash
    content_hash = hashlib.sha256(text.encode()).hexdigest()
    
    metadata = {
        "doc_id": doc_id,  #  Exact naming agent output
        "normalized_filename": state["normalized_filename"],
        "content_hash": content_hash,
        "extraction_stats": state["extraction_stats"].copy(),
        "language": "en",
        "has_tables": state["extraction_stats"]["table_blocks"] > 0,
        "cleaned_timestamp": datetime.utcnow().isoformat()
    }
    
    return {
        **state,
        "raw_text": text.strip(),
        "metadata": metadata,
        "status": "cleaned"
    }

# =============================================================================
#  CHUNKING (Canonical IDs)
# =============================================================================

def semantic_chunking_production(state: ParserState) -> ParserState:
    """ Canonical chunk IDs: docname_c001"""
    text = state["raw_text"]
    doc_id = state["metadata"]["doc_id"]  #  normalized_filename
    
    if not text.strip():
        return {**state, "chunks": [], "quality_score": 0.0}
    
    # Embedding model fallback chain
    # === SEMANTIC CHUNKING MODEL (PHASE-1 COMPATIBLE) ===
    E5_MODEL_PATH = os.getenv(
        "EMBEDDING_MODEL_PATH",
        r"D:\models\e5-large"   # EXACTLY what you used in Phase-1
    )

    if not Path(E5_MODEL_PATH).exists():
        raise RuntimeError(
            f"E5 model not found at {E5_MODEL_PATH}. "
            "Set EMBEDDING_MODEL_PATH correctly."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=E5_MODEL_PATH
    )

    print(f" Semantic chunking model (LOCAL E5): {E5_MODEL_PATH}")

 
    # Semantic chunking
    try:
        chunker = SemanticChunker(
            embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=85
        )
        chunk_texts = chunker.split_text(text)
    
    except Exception as e:
    
        raise RuntimeError(
            f"Semantic chunking failed using E5 model: {e}"
        )
    
    #  PRODUCTION CHUNKS
    chunks = []
    for i, chunk_text in enumerate(chunk_texts):
        chunk_text = chunk_text.strip()
        if len(chunk_text) < 100:
            continue
            

        #this is chunk hash
        is_table = bool(re.search(r'\[TABLE[^\]]*\]', chunk_text))
        chunk_hash = hashlib.md5(chunk_text.encode()).hexdigest()
        quality = calculate_chunk_quality(chunk_text, is_table)
        
        chunk = {
            "chunk_id": f"{doc_id}_c{i:03d}", 
            "chunk_index": i,
            "text": chunk_text,
            "text_hash": chunk_hash,
            "text_length": len(chunk_text),
            "word_count": len(chunk_text.split()),
            "is_table": is_table,
            "quality_score": quality,
            "metadata": {
                "doc_id": doc_id,  
                "normalized_filename": state["normalized_filename"],
                "chunk_index": i,
                "chunk_type": "table" if is_table else "text",
                "extraction_method": "semantic",
                "quality_score": quality
            }
        }
        chunks.append(chunk)
    
    #doc score
    quality_score = sum(c["quality_score"] for c in chunks) / max(len(chunks), 1)
    
    return {
        **state,
        "chunks": chunks,
        "quality_score": quality_score,
        "status": "chunked"
    }

def simple_chunk_fallback(text: str) -> list:
    """Emergency fallback"""
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip() and len(p.strip()) > 100]
    return paragraphs[:100]  # Cap chunks

def calculate_chunk_quality(text: str, is_table: bool = False) -> float:
    """Production quality score"""
    score = 0.0
    
    # Optimal length (300-800 chars)
    length_score = 1.0 - abs(len(text) - 500) / 500
    score += max(0, length_score) * 0.4
    
    # Table bonus
    if is_table:
        score += 0.3
    
    # Sentence diversity
    sentences = len(re.split(r'[.!?]+', text))
    sentence_score = min(1.0, sentences / 5.0)
    score += sentence_score * 0.3
    
    return min(1.0, score)

# =============================================================================
# OCR
# =============================================================================

def ocr_fallback_fixed(state: ParserState) -> ParserState:
    """OCR only first 5 pages if needed"""
    if not state.get("needs_ocr", False) or not HAS_OCR:
        return state
    
    pdf_path = Path(state['file_path'])
    doc = fitz.open(pdf_path)
    ocr_parts = []
    
    for page_num in range(min(5, len(doc))):
        page = doc[page_num]
        zoom = 2
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        try:
            image = Image.open(io.BytesIO(pix.tobytes("png")))
            ocr_text = pytesseract.image_to_string(image)
            if ocr_text.strip():
                ocr_parts.append(ocr_text.strip())
                state["extraction_stats"]["ocr_pages"] += 1
        except:
            pass
    
    doc.close()
    
    if ocr_parts:
        state["raw_text"] += f"\n\n=== OCR SUPPLEMENT ===\n\n" + "\n\n".join(ocr_parts)
    
    return {**state, "status": "ocr_completed"}

# =============================================================================
# WORKFLOW ROUTING
# =============================================================================

def quality_gate_fixed(state: ParserState) -> str:
    """Smart conditional routing"""
    if state.get("needs_ocr", False) and HAS_OCR:
        return "ocr"
    return "clean"

# =============================================================================
# PERFECTED LANGGRAPH
# =============================================================================

workflow = StateGraph(ParserState)
workflow.add_node("extract", extract_structure_fixed)
workflow.add_node("ocr", ocr_fallback_fixed)
workflow.add_node("clean", clean_text_fixed)
workflow.add_node("chunk", semantic_chunking_production)

workflow.set_entry_point("extract")
workflow.add_conditional_edges("extract", quality_gate_fixed, {
    "ocr": "ocr",
    "clean": "clean"
})
workflow.add_edge("ocr", "clean")
workflow.add_edge("clean", "chunk")
workflow.add_edge("chunk", END)

parser_agent = workflow.compile()

# =============================================================================
# PRODUCTION INTEGRATION API
# =============================================================================

def run_parser_pipeline(file_path: str, normalized_filename: str) -> ParserState:
    """
    Naming Agent → Parser Agent handoff
    
    Args:
        file_path: Raw PDF path
        normalized_filename: Naming agent output ("deposit_policy_v5.pdf")
    
    Returns:
        Complete ParserState with production chunks
    """
    result = parser_agent.invoke({
        "file_path": file_path,
        "normalized_filename": normalized_filename,  # ✅ Source of truth
        "chunks": [],
        "parsing_errors": [],
        "tables": [],
        "status": "pending"
    })
    
    # Production metrics
    log_extraction_metrics({
        "doc_id": result["metadata"]["doc_id"],
        "normalized_filename": normalized_filename,
        "chunks_created": len(result["chunks"]),
        "quality_score": result["quality_score"],
        "tables_detected": result["extraction_stats"]["table_blocks"],
        "status": result["status"]
    })
    
    return result

# =============================================================================
# TEST & VALIDATION
# =============================================================================

if __name__ == "__main__":
    # Simulate naming agent output
    BASE_DIR = Path(__file__).parent.parent  

    test_file = BASE_DIR / "data" / "pdfs" / "deposit_policy_v1.pdf"
    test_filename = "deposit_policy_v1.pdf"  
    
    result = run_parser_pipeline(test_file, test_filename)
    
    print("✅ PIPELINE COMPLETE")
    print(f"📄 Doc ID: {result['metadata']['doc_id']}")
    print(f"📊 Chunks: {len(result['chunks'])}")
    print(f"⭐ Quality: {result['quality_score']:.3f}")
    print(f"📋 Tables: {result['extraction_stats']['table_blocks']}")
    print(f"\n🔗 First chunk: {result['chunks'][0]['chunk_id']}")
    print(f"🔗 Chunk doc_id: {result['chunks'][0]['metadata']['doc_id']}")
    print(f"✅ CONSISTENT: {result['metadata']['doc_id'] == result['chunks'][0]['metadata']['doc_id']}")
