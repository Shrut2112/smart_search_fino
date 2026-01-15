"""
Universal Parser Agent
Production-grade PDF parsing with safe PyMuPDF handling
"""

from pathlib import Path
from typing import Dict, Any, List, TypedDict
from datetime import datetime
import hashlib

import fitz  # PyMuPDF
import pandas as pd


# =============================================================================
# STATE DEFINITIONS
# =============================================================================

class ParserState(TypedDict, total=False):
    file_path: str
    normalized_filename: str

    raw_text: str
    chunks: List[Dict[str, Any]]

    metadata: Dict[str, Any]
    extraction_stats: Dict[str, Any]

    text_blocks: List[str]
    tables: List[str]

    quality_score: float
    parsing_errors: List[str]
    needs_ocr: bool
    status: str


# =============================================================================
# CORE EXTRACTION (CRITICAL FIX APPLIED)
# =============================================================================

def extract_structure_fixed(state: ParserState) -> ParserState:
    """
    Extract text + tables from PDF using PyMuPDF.

    CRITICAL GUARANTEE:
    - len(doc) is evaluated BEFORE processing
    - doc.close() happens ONLY in finally
    - no access after close
    """

    file_path = Path(state["file_path"])

    if not file_path.exists():
        return {
            **state,
            "status": "failed",
            "parsing_errors": [f"File not found: {file_path}"],
            "chunks": [],
            "quality_score": 0.0,
        }

    if file_path.suffix.lower() != ".pdf":
        return {
            **state,
            "status": "unsupported_format",
            "parsing_errors": [f"Unsupported format: {file_path.suffix}"],
            "chunks": [],
        }

    doc = None
    try:
        doc = fitz.open(file_path)

        # ✅ CRITICAL FIX: read once, before processing
        total_pages = len(doc)

        text_blocks: List[str] = []
        tables_content: List[str] = []
        full_text_parts: List[str] = []

        extraction_stats = {
            "total_pages": total_pages,
            "text_blocks": 0,
            "table_blocks": 0,
            "raw_chars": 0,
            "ocr_pages": 0,
            "extraction_timestamp": datetime.utcnow().isoformat(),
        }

        low_quality_pages = 0

        for page_num in range(total_pages):
            page = doc[page_num]

            page_text_content: List[str] = []
            has_text = False

            # ------------------------------------------------------------------
            # LAYER 1: BLOCK EXTRACTION
            # ------------------------------------------------------------------
            blocks = page.get_text("blocks")
            for block in blocks:
                x0, y0, x1, y1, block_text, _, block_type = block
                if not block_text.strip():
                    continue

                if block_type == 0:  # text
                    page_text_content.append(block_text)
                    text_blocks.append(block_text)
                    extraction_stats["text_blocks"] += 1
                    has_text = True
                else:  # table / graphic region
                    rect = fitz.Rect(x0, y0, x1, y1)
                    table_text = page.get_text("text", clip=rect)
                    if table_text.strip():
                        formatted = f"\n[TABLE]{table_text.strip()}[/TABLE]\n"
                        page_text_content.append(formatted)
                        tables_content.append(formatted)
                        extraction_stats["table_blocks"] += 1

            # ------------------------------------------------------------------
            # LAYER 2: PyMuPDF TABLE DETECTION
            # ------------------------------------------------------------------
            try:
                tables = page.find_tables()
                for table in tables:
                    df = pd.DataFrame(table.extract())
                    if not df.empty:
                        table_str = df.to_string(index=False, header=False)
                        formatted = (
                            "\n[TABLE-DATA]\n"
                            f"{table_str}\n"
                            "[/TABLE-DATA]"
                        )
                        page_text_content.append(formatted)
                        tables_content.append(formatted)
                        extraction_stats["table_blocks"] += 1
            except Exception:
                pass

            # ------------------------------------------------------------------
            # LAYER 3: RAW FALLBACK
            # ------------------------------------------------------------------
            raw_text = page.get_text()
            if raw_text.strip():
                extraction_stats["raw_chars"] += len(raw_text)

            if not has_text or len(raw_text.strip()) < 50:
                low_quality_pages += 1

            full_text_parts.append("\n".join(page_text_content))

        # ----------------------------------------------------------------------
        # FINAL ASSEMBLY
        # ----------------------------------------------------------------------
        full_text = "\n\n=== PAGE BREAK ===\n\n".join(full_text_parts)

        needs_ocr = (
            total_pages > 0
            and (low_quality_pages / total_pages) > 0.2
        )

        return {
            **state,
            "raw_text": full_text,
            "text_blocks": text_blocks,
            "tables": tables_content,
            "extraction_stats": extraction_stats,
            "needs_ocr": needs_ocr,
            "status": "extracted",
        }

    except Exception as e:
        return {
            **state,
            "status": "failed",
            "parsing_errors": [f"PDF processing failed: {str(e)}"],
            "chunks": [],
        }

    finally:
        if doc is not None:
            doc.close()


# =============================================================================
# CHUNKING
# =============================================================================

def chunk_text(
    text: str,
    doc_id: str,
    max_chars: int = 1200,
) -> List[Dict[str, Any]]:
    chunks = []
    current = ""
    chunk_index = 0

    for line in text.splitlines():
        if len(current) + len(line) > max_chars:
            chunks.append({
                "chunk_id": f"{doc_id}_c{chunk_index:03d}",
                "text": current.strip(),
                "metadata": {"doc_id": doc_id},
            })
            chunk_index += 1
            current = line + "\n"
        else:
            current += line + "\n"

    if current.strip():
        chunks.append({
            "chunk_id": f"{doc_id}_c{chunk_index:03d}",
            "text": current.strip(),
            "metadata": {"doc_id": doc_id},
        })

    return chunks


# =============================================================================
# PIPELINE ENTRYPOINT
# =============================================================================

def run_parser_pipeline(
    file_path: str,
    normalized_filename: str,
) -> ParserState:
    """
    End-to-end parser pipeline
    """

    state: ParserState = {
        "file_path": file_path,
        "normalized_filename": normalized_filename,
        "raw_text": "",
        "chunks": [],
        "metadata": {},
        "extraction_stats": {},
        "text_blocks": [],
        "tables": [],
        "quality_score": 0.0,
        "parsing_errors": [],
        "status": "pending",
    }

    # ------------------------------------------------------------------
    # EXTRACTION
    # ------------------------------------------------------------------
    state = extract_structure_fixed(state)

    if state["status"] != "extracted":
        return state

    # ------------------------------------------------------------------
    # METADATA
    # ------------------------------------------------------------------
    content_hash = hashlib.sha256(
        state["raw_text"].encode("utf-8", errors="ignore")
    ).hexdigest()

    state["metadata"] = {
        "doc_id": normalized_filename,
        "content_hash": content_hash,
        "has_tables": len(state["tables"]) > 0,
        "cleaned_timestamp": datetime.utcnow().isoformat(),
    }

    # ------------------------------------------------------------------
    # CHUNKING
    # ------------------------------------------------------------------
    chunks = chunk_text(state["raw_text"], normalized_filename)
    state["chunks"] = chunks

    # ------------------------------------------------------------------
    # QUALITY SCORE (SIMPLE, STABLE)
    # ------------------------------------------------------------------
    text_len = len(state["raw_text"])
    table_bonus = 0.1 if state["metadata"]["has_tables"] else 0.0
    state["quality_score"] = min(1.0, (text_len / 5000.0) + table_bonus)

    state["status"] = "chunked"
    return state
