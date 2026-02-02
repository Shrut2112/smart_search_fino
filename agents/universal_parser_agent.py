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
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import io
from utils.schema import State
from dotenv import load_dotenv
load_dotenv()

# Optional OCR 
try:
    import pytesseract
    from PIL import Image
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

def log_extraction_metrics(metrics: Dict):
    """INFRA HOOK: LangSmith/Prometheus - pipeline provides"""
    pass


embeddings = None
def init_worker():
    global embeddings
    E5_MODEL_PATH = os.getenv(
        "EMBEDDING_MODEL_PATH",
        r"D:\models\e5-large"   # EXACTLY what you used in Phase-1
    )
    if embeddings is None:
        model_kwargs = {
            'device': 'cpu',
            'prompts': {
                "query": "query: ",
                "passage": "passage: "
            }
        }
        
        embeddings = HuggingFaceEmbeddings(
            model_name=E5_MODEL_PATH,
            model_kwargs=model_kwargs,
            encode_kwargs={
                'batch_size': 64,
                'normalize_embeddings': True
            }
        )
        print("Embedding model initialized. 64 batch size, CPU device.")
    
# =============================================================================
# LAYER 1-3: EXTRACTION ENGINE (Fixed)
# =============================================================================

def extract_structure_fixed(state: State) -> State:
    """Blocks → Tables → Fallback """
    original_filename = Path(state['original_filename'])
    
    if not original_filename.exists():
        return {
            **state,
            "status": "failed",
            "parsing_errors": [f"File not found: {original_filename}"],
            "chunks": [],
            "quality_score": 0.0
        }
    
    if original_filename.suffix.lower() != '.pdf':
        return {
            **state,
            "status": "unsupported_format",
            "parsing_errors": [f"Unsupported format: {original_filename.suffix}"],
            "chunks": []
        }
    
    try:
        doc = fitz.open(original_filename)
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
    
    full_text = "\n\n=== PAGE BREAK ===\n\n".join(full_text_parts)
    needs_ocr = len(doc) > 0 and (low_quality_pages / len(doc)) > 0.2
    
    doc.close()
    
    
    return {
        **state,
        "text_blocks": text_blocks,
        "tables": tables_content,
        "raw_text": full_text,
        "extraction_stats": extraction_stats,
        "needs_ocr": needs_ocr,
        "status": "extracted"
    }



def clean_text_fixed(state: State) -> State:
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
    doc_id = state["base_doc_name"]  
    
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
    with open("write.txt", "w", encoding="utf-8") as f:
        f.write(text)
    return {
        **state,
        "content_hash": content_hash,
        "raw_text": text.strip(),
        "metadata": metadata,
        "status": "cleaned"
    }

# =============================================================================
#  CHUNKING (Canonical IDs)
# =============================================================================


def semantic_chunking_production(state: State) -> State:
    """Chunks based on Page Breaks. Fast, reliable, and preserves context."""
    global embeddings
    raw_text = state["raw_text"]
    doc_id = state["metadata"]["doc_id"]
    version = state['version']
    
    # Split by the markers we created in the extraction step
    pages = raw_text.split("=== PAGE BREAK ===")
    
    # Fallback splitter ONLY for exceptionally long pages
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, #char size not tokens
        chunk_overlap=200,
        separators=["\n\n", "\n", " "]
    )
    
    final_chunks = []
    chunk_idx = 0
    
    for pg_num, page_content in enumerate(pages):
        page_content = page_content.strip()
            
        # If the page is a reasonable size, make it one chunk
        if len(page_content) <= 1200: 
            page_parts = [page_content]
        else:
            # Only use the splitter if the page is huge
            page_parts = fallback_splitter.split_text(page_content)
            
        for part in page_parts:
            # Clean up text
            clean_part = part.replace("=== PAGE BREAK ===", "").strip()
            
            is_table = "[TABLE" in clean_part
            chunk_hash = hashlib.md5(clean_part.encode()).hexdigest()
            quality = calculate_chunk_quality(clean_part, is_table)
            
            final_chunks.append({
                "chunk_id": f"{doc_id}_{version}_c{chunk_idx:03d}",
                "chunk_index": chunk_idx,
                "text": clean_part,
                "text_hash": chunk_hash,
                "quality_score": quality,
                "metadata": {
                    **state["metadata"],
                    "page_number": pg_num + 1,
                    "chunk_index": chunk_idx,
                    "is_table": is_table
                }
            })
            chunk_idx += 1
    chunk_text = [c["text"] for c in final_chunks]
    # EMBEDDING
    model = embeddings
    embed_vectors = model.embed_documents(chunk_text)
    embed_chunks = []
    for i, chunk in enumerate(final_chunks):
        chunk_copy = chunk.copy()
        chunk_copy["embedding"] = embed_vectors[i]
        embed_chunks.append(chunk_copy)
    return {
        **state,
        "chunks": embed_chunks,
        "quality_score": sum(c["quality_score"] for c in final_chunks) / max(len(final_chunks), 1),
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


def ocr_fallback_fixed(state: State) -> State:
    """OCR only first 5 pages if needed"""
    if not state.get("needs_ocr", False) or not HAS_OCR:
        return state
    
    pdf_path = Path(state['original_filename'])
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


def quality_gate_fixed(state: State) -> str:
    """Smart conditional routing"""
    if state.get("needs_ocr", False) and HAS_OCR:
        return "ocr"
    return "clean"

def parser_graph(state:State):
    workflow = StateGraph(State)
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
    
    return parser_agent

