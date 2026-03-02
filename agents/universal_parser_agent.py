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
import camelot

load_dotenv()

from utils.logger import get_logger
log = get_logger("agent.parser")


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
            "device": "cpu",
            "prompts": {
                "query": "query: ",
                "passage": "passage: "
            }
        }

        embeddings = HuggingFaceEmbeddings(
            model_name=E5_MODEL_PATH,
            model_kwargs=model_kwargs,
            encode_kwargs={
                "batch_size": 64,
                "normalize_embeddings": True
            }
        )

        log.info("Embedding model initialized")

def sanitize_text(text: Any) -> str:
    """Helper to remove NUL characters from any input."""
    if text is None:
        return ""
    return str(text).replace("\x00", "")

def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Deep cleans a dataframe of NUL characters in headers and cells."""
    if df is None or df.empty:
        return df
    # Clean column names
    df.columns = [sanitize_text(c) for c in df.columns]
    # Clean all cell values
    return df.map(lambda x: sanitize_text(x) if isinstance(x, str) else x)

def extract_tables_camelot(pdf_path: Path, page_num: int):

    dfs = []

    try:

        tables = camelot.read_pdf(
            str(pdf_path),
            pages=str(page_num + 1),
            flavor="lattice"
        )

        for t in tables:
            if not t.df.empty:
                dfs.append(sanitize_dataframe(t.df))

    except:
        pass


    if not dfs:

        try:

            tables = camelot.read_pdf(
                str(pdf_path),
                pages=str(page_num + 1),
                flavor="stream"
            )

            for t in tables:
                if not t.df.empty:
                    dfs.append(sanitize_dataframe(t.df))

        except:
            pass


    return dfs


def is_bad_table_strict(df: pd.DataFrame) -> bool:

    if df is None or df.empty:
        return True

    rows, cols = df.shape

    if rows < 2 or cols < 2:
        return True

    if cols <= 2:
        return True


    headers = [str(c).strip() for c in df.columns]

    long_headers = 0
    weird_headers = 0

    for h in headers:

        if len(h.split()) > 5:
            long_headers += 1

        if len(h) > 40:
            weird_headers += 1

        if re.search(r"[|<>={}]", h):
            weird_headers += 1


    if long_headers >= cols * 0.4:
        return True

    if weird_headers >= cols * 0.3:
        return True


    empty_count = 0
    total = rows * cols

    for r in range(rows):
        for c in range(cols):

            v = str(df.iat[r, c]).strip()

            if v == "" or v.lower() == "nan":
                empty_count += 1


    if empty_count / (total + 1) > 0.35:
        return True


    header_row = [str(x).strip() for x in headers]

    repeated = 0

    for r in range(rows):

        row = [str(df.iat[r, c]).strip() for c in range(cols)]

        if row == header_row:
            repeated += 1


    if repeated >= 2:
        return True


    bad_chars = 0
    total_chars = 0

    for r in range(rows):
        for c in range(cols):

            v = str(df.iat[r, c])

            total_chars += len(v)

            bad_chars += len(
                re.findall(r"[^\w\s\.,₹%-]", v)
            )


    if total_chars > 0 and bad_chars / total_chars > 0.15:
        return True


    first_col = [
        str(df.iat[r, 0]).strip()
        for r in range(rows)
    ]

    avg_len = sum(len(x) for x in first_col) / (len(first_col) + 1)

    if avg_len > 60:
        return True


    long_cells = 0

    for r in range(rows):
        for c in range(cols):

            if len(str(df.iat[r, c]).split()) > 12:
                long_cells += 1


    if long_cells > total * 0.2:
        return True


    return False


def looks_like_multi_header(df: pd.DataFrame) -> bool:

    if df.shape[0] < 2:
        return False

    row0 = [str(x).strip() for x in df.iloc[0]]
    row1 = [str(x).strip() for x in df.iloc[1]]

    score = 0

    for a, b in zip(row0, row1):

        if not re.search(r"\d", a) and not re.search(r"\d", b):
            score += 1


    return score >= len(row0) * 0.6


def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:

    if df.shape[0] < 2:
        return df


    first = [str(x).strip() for x in df.iloc[0]]
    second = [str(x).strip() for x in df.iloc[1]]

    merged = []


    for a, b in zip(first, second):

        if a and b:
            merged.append(a + " " + b)

        elif a:
            merged.append(a)

        else:
            merged.append(b)


    df.columns = merged

    return df.iloc[2:].reset_index(drop=True)



def extract_structure_fixed(state: State) -> State:

    original_filename = Path(state["original_filename"])


    if not original_filename.exists():
        log.info(f"Parsing started -> {original_filename.name}")

        return {
            **state,
            "status": "failed",
            "parsing_errors": ["File not found"],
            "chunks": []
        }


    if original_filename.suffix.lower() != ".pdf":

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


    with pdfplumber.open(original_filename) as pdf_plumber:

        for page_num, pdf_page in enumerate(
            pdf_plumber.pages[:len(doc)]
        ):


            tables = pdf_page.extract_tables()

            page_has_good_table = False


            for table_idx, table in enumerate(tables or []):

                if table and len(table) > 1:

                    df = pd.DataFrame(
                        table[1:],
                        columns=table[0] if table[0] else None
                    )
                    df = sanitize_dataframe(df)

                    if is_bad_table_strict(df):
                        continue


                    page_has_good_table = True


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


            if not page_has_good_table:

                low_quality_pages.add(page_num)

                camelot_dfs = extract_tables_camelot(
                    original_filename,
                    page_num
                )


                for c_idx, cdf in enumerate(camelot_dfs):

                    if looks_like_multi_header(cdf):
                        cdf = normalize_headers(cdf)

                    cdf = sanitize_dataframe(cdf)
                    
                    if is_bad_table_strict(cdf):
                        continue


                    table_json = {
                        "type": "table",
                        "page": page_num + 1,
                        "table_id": f"c_{page_num}_{c_idx}",
                        "columns": cdf.columns.tolist(),
                        "rows": cdf.to_dict("records"),
                        "shape": cdf.shape,
                        "text_summary": table_to_sentences(cdf)
                    }


                    structured_tables.append(table_json)

                    extraction_stats["table_blocks"] += 1


            page_text = pdf_page.extract_text()
            page_text = sanitize_text(page_text)
            # --------- REMOVE FLATTENED TABLE TEXT ---------
            if page_has_good_table and page_text:
            
                page_text = re.sub(
                    r"\n\s*\d+(\s+\d+)+.*",
                    "",
                    page_text
                )
            


            if page_text:

                text_blocks.append(page_text.strip())

                extraction_stats["text_blocks"] += 1

            else:
                low_quality_pages.add(page_num)


    # FIX: no .copy() (prevents duplication)
    full_text_parts = []


    for txt in text_blocks:
        full_text_parts.append(txt)


    for page_num in low_quality_pages:

        if page_num >= len(doc):
            continue


        page = doc[page_num]

        page_text = sanitize_text(doc[page_num].get_text().strip())


        if page_text and len(page_text) >= 50:
            full_text_parts.append(page_text)


    full_text = "\n\n=== PAGE BREAK ===\n\n".join(full_text_parts)

    # --------- DEDUPLICATION FIX ---------
    seen = set()
    clean_blocks = []

    for t in full_text.split("=== PAGE BREAK ==="):

        key = re.sub(r"\s+", " ", t.strip())[:200]
   # fingerprint (first 200 chars)

        if key not in seen:
            seen.add(key)
            clean_blocks.append(t.strip())

    full_text = "\n\n=== PAGE BREAK ===\n\n".join(clean_blocks)



    needs_ocr = len(low_quality_pages) / max(len(doc), 1) > 0.2


    doc.close()


    low_quality_pages = list(low_quality_pages)
    text = full_text.replace('\x00', 'ti')

    log.info(f"Extraction done -> pages={extraction_stats['total_pages']}, text_blocks={extraction_stats['text_blocks']}, tables={extraction_stats['table_blocks']}, needs_ocr={needs_ocr}")

    return {
        **state,
        "text_blocks": text_blocks,
        "structured_tables": structured_tables,
        "raw_text": text,
        "extraction_stats": extraction_stats,
        "low_quality_pages": low_quality_pages,
        "needs_ocr": needs_ocr,
        "status": "extracted"
    }

def table_to_sentences(df: pd.DataFrame) -> str:

    sentences = []


    for _, row in df.iterrows():

        parts = [
            f"{col}: {row[col]}"
            for col in df.columns
        ]

        sentences.append(" | ".join(parts) + ".")


    return " ".join(sentences)



def clean_text_fixed(state: State) -> State:
    raw = state.get("raw_text")
    if not raw:
        return {
            **state,
            "status": "failed",
            "raw_text": "",
            "parsing_errors": state.get("parsing_errors", []) + ["raw_text missing before clean_text_fixed"]
        }

    text = sanitize_text(raw)

    text = re.sub(
        r"Classification: Internal.*?(?=\n{2,}|\Z)",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    text = re.sub(
        r"FINO Payments Bank Limited\s*\n?",
        "",
        text,
        flags=re.IGNORECASE
    )


    text = re.sub(r"\n{5,}", "\n\n\n", text)
    text = re.sub(r"\s{4,}", " ", text)

    # --------- REMOVE FIRM HEADER / FOOTER ---------
    text = re.sub(
        r"DM\s*&\s*ASSOCIATES.*?LLP.*?Email:.*?\n",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    text = re.sub(
        r"REGD\.?\s*OFFICE.*?\n",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )



    footer_start = "Registered Office: Mindspace Juinagar"
    footer_end = "website: www.finobank.com"


    flexible_pattern = (
        re.escape(footer_start)
        + r".*?"
        + re.escape(footer_end)
    )


    text = re.sub(
        flexible_pattern,
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )
    # RE-SANITIZE STRUCTURED TABLES BEFORE CHUNKING
    if state.get("structured_tables"):
        for table in state["structured_tables"]:
            table["columns"] = [sanitize_text(c) for c in table["columns"]]
            table["text_summary"] = sanitize_text(table["text_summary"])
            for row in table.get("rows", []):
                for k, v in row.items():
                    if isinstance(v, str): row[k] = sanitize_text(v)

    extraction_stats = {**state["extraction_stats"], "total_chars_post_clean": len(text)}


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

    log.info(f"Text cleaned -> content_hash={content_hash[:16]}..., chars_post_clean={len(text.strip())}")

    return {
        **state,
        "content_hash": content_hash,
        "raw_text": text.strip(),
        "metadata": metadata,
        "status": "cleaned"
    }

def semantic_chunking_production(state: State) -> State:

    global embeddings


    metadata = state.get("metadata")
    if not metadata or not metadata.get("doc_id"):
        return {
            **state,
            "status": "failed",
            "chunks": [],
            "parsing_errors": state.get("parsing_errors", []) + ["metadata/doc_id missing before semantic_chunking"]
        }
    doc_id = metadata["doc_id"]
    
    
    version = state["version"]


    final_chunks = []
    chunk_idx = 0


    if state.get("structured_tables"):

        for table in state["structured_tables"]:

            if not table.get("text_summary"):
                continue


            chunk = {
                "chunk_id": f"{doc_id}_{version}_t{chunk_idx:03d}",
                "chunk_index": chunk_idx,
                "chunk_type": "table",
                "text": table["text_summary"],
                "raw_content": json.dumps(table).replace("\x00", ""),
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


    raw_text = state["raw_text"]


    pages = raw_text.split("=== PAGE BREAK ===")


    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,   # FIX (was 200)
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


    if embeddings is not None and final_chunks:

        chunk_text = [c["text"] for c in final_chunks]

        embed_vectors = embeddings.embed_documents(chunk_text)


        for i, chunk in enumerate(final_chunks):

            vector = embed_vectors[i]

            chunk["embedding"] = (
                vector.tolist()
                if hasattr(vector, "tolist")
                else vector
            )

    table_count = len([c for c in final_chunks if c.get("chunk_type") == "table"])
    text_count = len(final_chunks) - table_count
    log.info(f"Chunking done -> total={len(final_chunks)}, text={text_count}, table={table_count}, avg_quality={sum(c['quality_score'] for c in final_chunks) / max(len(final_chunks), 1):.2f}")

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


    sentences = len(re.split(r"[.!?]+", text))

    score += min(1.0, sentences / 5.0) * 0.3


    return min(1.0, score)



def ocr_fallback_fixed(state: State) -> State:

    if not state.get("needs_ocr", False) or not HAS_OCR:
        return state


    pdf_path = Path(state["original_filename"])


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


            ocr_text = sanitize_text(pytesseract.image_to_string(image))


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

        updated_stats = {**state["extraction_stats"], "ocr_pages": state["extraction_stats"]["ocr_pages"] + 1}
        updated_text = state["raw_text"] + "\n\n=== OCR SUPPLEMENT ===\n\n" + "\n\n".join(ocr_parts)
        return {**state, "raw_text": updated_text, "extraction_stats": updated_stats, "status": "ocr_completed"}


    return {**state, "status": "ocr_completed"}

def route_after_clean(state: State) -> str:
    if state.get("status") in ("failed", "unsupported"):
        return "failed"
    return "chunk"

def quality_gate_fixed(state: State) -> str:
    if state.get("status") in ("failed", "unsupported"):
        return "failed"
    if state.get("needs_ocr", False) and HAS_OCR:
        return "ocr"
    return "clean"


def parser_graph():
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
            "clean": "clean",
            "failed": END
        }
    )

    workflow.add_edge("ocr", "clean")

    workflow.add_conditional_edges(
        "clean",
        route_after_clean,
        {
            "chunk": "chunk",
            "failed": END
        }
    )

    workflow.add_edge("chunk", END)

    return workflow.compile()