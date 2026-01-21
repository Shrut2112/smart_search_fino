import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Any
import os
from contextlib import contextmanager
from dotenv import load_dotenv
from psycopg2.extras import execute_values

load_dotenv()

# ye config change krde

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "fino_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}


@contextmanager
def get_db_connection():
    """Safe DB connection with auto-close"""
    conn = None
    try:
        conn = psycopg2.connect("postgresql://postgres:[Codeis@04]@db.cxoartbafydqirizvyzh.supabase.co:5432/postgres")
        yield conn
    finally:
        if conn:
            conn.close()

def check_doc_with_name_version(filename:str,version_no:str)->bool:
    """Check Doc exists if no revised name"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS(
                    SELECT 1
                    FROM document_version
                    WHERE doc_id = %s and version = %s
                )                
                """,(filename,version_no))

            result = cur.fetchone()
            return result[0] if result else False
    

def query_doc_exists(doc_hash: str) -> bool:
    """SHA256 doc hash exists? check"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM document_versions 
                    WHERE content_hash = %s
                )
            """, (doc_hash,))
            return cur.fetchone()[0]

def load_latest_chunks(doc_prefix: str) -> List[Dict[str, Any]]:
    """
    Returns: [{"chunk_id": "...", "text_hash": "...", "text": "...", "metadata": {...}}]
    """
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            base_name = doc_prefix.rsplit('_v', 1)[0] if '_v' in doc_prefix else doc_prefix
            
            # Get latest doc_id first
            cur.execute("""
                SELECT doc_id FROM document_versions
                WHERE doc_id LIKE %s
                ORDER BY version DESC
                LIMIT 1
            """, (f"{base_name}%",))
            
            row = cur.fetchone()
            if not row:
                return []
            
            latest_doc_id = row['doc_id']
            
            # Fetch chunks for that specific doc_id
            cur.execute("""
                SELECT c.chunk_id, c.text_hash, c.text, c.metadata, c.status,
                       dv.version, dv.created_at
                FROM chunks c
                JOIN document_versions dv ON c.doc_id = dv.doc_id
                WHERE c.doc_id = %s 
                  AND c.status = 'active'
                ORDER BY c.chunk_index ASC
            """, (latest_doc_id,))
            
            return [dict(r) for r in cur.fetchall()]

def archive_chunk(chunk_id: str):
    """Archive old chunk: status='archived'"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE chunks 
                SET status = 'archived', 
                    archived_at = NOW()
                WHERE chunk_id = %s AND status = 'active'
            """, (chunk_id,))
            conn.commit()

def upsert_chunks(chunks: List[Dict[str, Any]]):
    """Batch upsert chunks logic"""
    if not chunks:
        return

    import json
    
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            values = []
            for chunk in chunks:
                chunk_id = chunk["chunk_id"]
                text_hash = chunk["text_hash"]
                text = chunk["text"][:8000]
                metadata = json.dumps(chunk["metadata"]) # CORRECT: JSON string
                doc_id = chunk["metadata"]["doc_id"]
                # DB status matches standard active/archived. Comparison status is in metadata.
                status = "active" 
                
                values.append((
                    chunk_id, text_hash, text, metadata, doc_id, status,
                    chunk["metadata"].get("chunk_index", 0),
                    chunk["metadata"].get("quality_score", 0.0)
                ))
            
            execute_values(cur, """
                INSERT INTO chunks (
                    chunk_id, text_hash, text, metadata, doc_id, status, 
                    chunk_index, quality_score
                )
                VALUES %s
                ON CONFLICT (chunk_id) 
                DO UPDATE SET
                    text_hash = EXCLUDED.text_hash,
                    text = EXCLUDED.text,
                    metadata = EXCLUDED.metadata,
                    doc_id = EXCLUDED.doc_id,
                    status = EXCLUDED.status,
                    updated_at = NOW()
            """, values)
            conn.commit()
