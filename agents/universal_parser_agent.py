# agents/universal_parser_agent.py

from typing import TypedDict, List, Dict, Any, Tuple
from pathlib import Path
import fitz
import pandas as pd
import hashlib
import re
import os
import json
from datetime import datetime
from langgraph.graph import StateGraph, END
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import io
from utils.schema import State
from dotenv import load_dotenv
import pdfplumber

load_dotenv()

# Optional OCR
try:
    import pytesseract
    from PIL import Image
    HAS_OCR = True
except ImportError:
    HAS_OCR = False


embeddings = None


def init_worker():
    global embeddings

    if embeddings is None:
        E5_MODEL_PATH = os.getenv(
            "EMBEDDING_MODEL_PATH",
            r"D:\models\e5-large"
        )

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

        print("Embedding model initialized")


# EXTRACTION

def extract_structure_fixed(state: State) -> State:

    original_filename = Path(state['original_filename'])

    if not original_filename.exists():
        return {
            **state,
            "status": "failed",
            "parsing_errors": ["File not found"],
            "chunks": []
        }

    if original_filename.suffix.lower() != '.pdf':
        return {
            **state,
            "status": "unsupported",
            "parsing_errors": [f"Unsupported: {original_filename.suffix}"],
            "chunks": []
        }

    try:
        doc = fitz.open(original_filename)

    except Exception as e:
        return {
            **state,
            "status": "failed",
            "parsing_errors": [f"PDF failed: {e}"],
            "chunks": []
        }

    text_blocks = []
    structured_tables = []

    low_quality_pages = set()

    extraction_stats = {
        "total_pages": len(doc),
        "text_blocks": 0,
        "table_blocks": 0,
        "raw_chars": 0,
        "ocr_pages": 0,
        "extraction_timestamp": datetime.utcnow().isoformat()
    }

    # PDFPLUMBER
    with pdfplumber.open(original_filename) as pdf_plumber:

        for page_num, pdf_page in enumerate(pdf_plumber.pages[:len(doc)]):

            # Tables
            tables = pdf_page.extract_tables()

            for table_idx, table in enumerate(tables or []):

                if table and len(table) > 1:

                    df = pd.DataFrame(
                        table[1:],
                        columns=table[0] if table[0] else None
                    )

                    if not df.empty:

                        table_json = {
                            "type": "table",
                            "page": page_num + 1,
                            "table_id": f"t_{page_num}_{table_idx}",
                            "columns": df.columns.tolist(),
                            "rows": df.to_dict("records"),
                            "shape": df.shape,
                            "text_summary": table_to_sentences(df)
                        }

                        structured_tables.append(table_json)

                        extraction_stats["table_blocks"] += 1

            # Text
            page_text = pdf_page.extract_text()

            if page_text:
                text_blocks.append(page_text.strip())
                extraction_stats["text_blocks"] += 1

            else:
                low_quality_pages.add(page_num)

    # ---------- PYMUPDF FALLBACK (ONLY LOW QUALITY) ----------
    full_text_parts = text_blocks.copy()

    for page_num in low_quality_pages:

        if page_num >= len(doc):
            continue

        page = doc[page_num]

        page_text = page.get_text().strip()

        if page_text and len(page_text) >= 50:
            full_text_parts.append(page_text)

    full_text = "\n\n=== PAGE BREAK ===\n\n".join(full_text_parts)

    needs_ocr = len(low_quality_pages) / len(doc) > 0.2

    doc.close()

    low_quality_pages = list(low_quality_pages)

    return {
        **state,
        "text_blocks": text_blocks,
        "structured_tables": structured_tables,
        "raw_text": full_text,
        "extraction_stats": extraction_stats,
        "low_quality_pages": low_quality_pages,
        "needs_ocr": needs_ocr,
        "status": "extracted"
    }


# TABLE → TEXT

def table_to_sentences(df: pd.DataFrame) -> str:

    sentences = []

    for _, row in df.iterrows():

        parts = [f"{col}: {row[col]}" for col in df.columns]

        sentences.append(" | ".join(parts) + ".")

    return " ".join(sentences)


# CLEANING

def clean_text_fixed(state: State) -> State:

    text = state["raw_text"]

    text = re.sub(
        r'Classification: Internal.*?(?=\n{2,}|\Z)',
        '',
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    text = re.sub(
        r'FINO Payments Bank Limited\s*\n?',
        '',
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(r'\n{5,}', '\n\n\n', text)
    text = re.sub(r'\s{4,}', ' ', text)
    footer_start = "Registered Office: Mindspace Juinagar"
    footer_end = "website: www.finobank.com"

    # 2. Create a pattern that handles multi-line breaks and extra spaces
    # \s+ matches one or more whitespace characters (space, tab, or newline)
    flexible_pattern = re.escape(footer_start) + r".*?" + re.escape(footer_end)

    # 3. Use re.DOTALL so the '.' matches newlines as well
    text = re.sub(flexible_pattern, '', text, flags=re.DOTALL | re.IGNORECASE)

    state["extraction_stats"]["total_chars_post_clean"] = len(text)

    doc_id = state["base_doc_name"]

    content_hash = hashlib.sha256(text.encode()).hexdigest()

    metadata = {
        "doc_id": doc_id,
        "normalized_filename": state["normalized_filename"],
        "content_hash": content_hash,
        "extraction_stats": state["extraction_stats"].copy(),
        "language": "en",
        "has_tables": state["extraction_stats"]["table_blocks"] > 0,
        "cleaned_timestamp": datetime.utcnow().isoformat()
    }

    return {
        **state,
        "content_hash": content_hash,
        "raw_text": text.strip(),
        "metadata": metadata,
        "status": "cleaned"
    }


# CHUNKING

def semantic_chunking_production(state: State) -> State:

    global embeddings

    doc_id = state["metadata"]["doc_id"]
    version = state["version"]

    final_chunks = []
    chunk_idx = 0

    # TABLE CHUNKS
    if state.get("structured_tables"):

        for table in state["structured_tables"]:

            if not table.get("text_summary"):
                continue

            chunk = {
                "chunk_id": f"{doc_id}_{version}_t{chunk_idx:03d}",
                "chunk_index": chunk_idx,
                "chunk_type": "table",
                "text": table["text_summary"],
                "raw_content": json.dumps(table),
                "text_hash": hashlib.md5(
                    table["text_summary"].encode()
                ).hexdigest(),
                "quality_score": 0.95,
                "metadata": {
                    **state["metadata"],
                    "chunk_type": "table",
                    "page_number": table["page"],
                    "table_shape": table["shape"],
                    "chunk_index": chunk_idx
                }
            }

            final_chunks.append(chunk)

            chunk_idx += 1

    # TEXT CHUNKS
    raw_text = state["raw_text"]

    pages = raw_text.split("=== PAGE BREAK ===")

    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "."]
    )

    for pg_num, page_content in enumerate(pages):

        page_content = page_content.strip()

        if len(page_content) <= 1200:
            page_parts = [page_content]

        else:
            page_parts = fallback_splitter.split_text(page_content)

        for part in page_parts:

            clean_part = part.replace(
                "=== PAGE BREAK ===", ""
            ).strip()

            if len(clean_part) < 100:
                continue

            is_table = "[TABLE" in clean_part

            chunk_hash = hashlib.md5(
                clean_part.encode()
            ).hexdigest()

            quality = calculate_chunk_quality(
                clean_part,
                is_table
            )

            final_chunks.append({

                "chunk_id": f"{doc_id}_{version}_c{chunk_idx:03d}",

                "chunk_index": chunk_idx,

                "chunk_type": "text",

                "text": clean_part,

                "text_hash": chunk_hash,

                "quality_score": quality,

                "metadata": {
                    **state["metadata"],
                    "chunk_type": "text",
                    "page_number": pg_num + 1,
                    "chunk_index": chunk_idx,
                    "is_table": is_table
                }
            })

            chunk_idx += 1

    # EMBEDDINGS
    if embeddings is not None and final_chunks:

        chunk_text = [c["text"] for c in final_chunks]

        embed_vectors = embeddings.embed_documents(chunk_text)

        for i, chunk in enumerate(final_chunks):
            vector = embed_vectors[i]   
            chunk["embedding"] = vector.tolist() if hasattr(vector, 'tolist') else vector

    return {
        **state,
        "chunks": final_chunks,
        "quality_score": sum(
            c["quality_score"] for c in final_chunks
        ) / max(len(final_chunks), 1),
        "status": "chunked"
    }


def calculate_chunk_quality(text: str, is_table: bool = False) -> float:

    score = 0.0

    length_score = 1.0 - abs(len(text) - 500) / 500
    score += max(0, length_score) * 0.4

    if is_table:
        score += 0.3

    sentences = len(re.split(r'[.!?]+', text))
    score += min(1.0, sentences / 5.0) * 0.3

    return min(1.0, score)


# OCR

def ocr_fallback_fixed(state: State) -> State:

    if not state.get("needs_ocr", False) or not HAS_OCR:
        return state

    pdf_path = Path(state['original_filename'])

    doc = fitz.open(pdf_path)

    ocr_parts = []

    for page_num in state.get("low_quality_pages", []):

        if page_num >= len(doc):
            continue

        page = doc[page_num]

        pix = page.get_pixmap(
            matrix=fitz.Matrix(2, 2)
        )

        try:
            image = Image.open(
                io.BytesIO(pix.tobytes("png"))
            )

            ocr_text = pytesseract.image_to_string(image)

            if ocr_text.strip():

                ocr_parts.append(
                    f"\n\n=== PAGE BREAK ===\n"
                    f"=== OCR PAGE {page_num + 1} ===\n"
                    f"{ocr_text}"
                )

                state["extraction_stats"]["ocr_pages"] += 1

        except:
            pass

    doc.close()

    if ocr_parts:
        state["raw_text"] += (
            "\n\n=== OCR SUPPLEMENT ===\n\n" +
            "\n\n".join(ocr_parts)
        )

    return {**state, "status": "ocr_completed"}


def quality_gate_fixed(state: State) -> str:

    if state.get("needs_ocr", False) and HAS_OCR:
        return "ocr"

    return "clean"


# WORKFLOW

def parser_graph(state:State):

    workflow = StateGraph(State)

    workflow.add_node("extract", extract_structure_fixed)
    workflow.add_node("ocr", ocr_fallback_fixed)
    workflow.add_node("clean", clean_text_fixed)
    workflow.add_node("chunk", semantic_chunking_production)

    workflow.set_entry_point("extract")

    workflow.add_conditional_edges(
        "extract",
        quality_gate_fixed,
        {
            "ocr": "ocr",
            "clean": "clean"
        }
    )

    workflow.add_edge("ocr", "clean")
    workflow.add_edge("clean", "chunk")
    workflow.add_edge("chunk", END)

    return workflow.compile()


# parser_agent = parser_graph()
