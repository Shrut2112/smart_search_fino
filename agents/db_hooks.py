import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Any
from contextlib import contextmanager
from dotenv import load_dotenv
from psycopg2.extras import execute_values
import json
from psycopg2 import pool
import os
from utils.logger import get_logger

log = get_logger("db.hooks")

# Create the pool ONCE at the top level of the module
# minconn=1, maxconn=10 (adjust maxconn based on your max_workers)
_db_pool = None

def init_db_pool():
    global _db_pool
    if _db_pool is None:
        load_dotenv()
        _db_pool = psycopg2.pool.SimpleConnectionPool(
            1, 10,
            dbname="postgres",
            user="postgres.cxoartbafydqirizvyzh",
            password="Codeis@04fino",
            host="aws-1-ap-southeast-2.pooler.supabase.com",
            port=6543,
            sslmode="require"
        )
        log.info("Connection Pool Initialized")

@contextmanager
def get_db_connection():
    """Borrow a connection from the pool"""
    if _db_pool is None:
        init_db_pool()
    
    conn = _db_pool.getconn()
    try:
        yield conn
    finally:
        # Give the connection back to the pool instead of closing it
        _db_pool.putconn(conn)
        
def check_doc_with_name_version(filename:str,version_no:str)->bool:
    """Check Doc exists if no revised name"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS(
                    SELECT 1
                    FROM document_versions
                    WHERE doc_id = %s and version = %s and active_status = 'active'
                )                
                """,(filename,version_no))

            result = cur.fetchone()
            return result[0] if result else False 

def check_doc_present(filename:str,version_no:str)->bool:
    """Check Doc exists if no revised name"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1
                    FROM document_versions
                    WHERE doc_id = %s
                      AND version <> %s
                      AND active_status = 'active'
                )                
                """,(filename,version_no))

            result = cur.fetchone()
            return result[0] if result else False 

def query_doc_exists(doc_hash: str) -> bool:
    """SHA256 doc hash exists? check"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT doc_id 
                FROM document_versions 
                WHERE content_hash = %s 
                  AND active_status = 'active'
                LIMIT 1
            """, (doc_hash,))
            
            result = cur.fetchone()
            # If result exists, it returns the filename (doc_id)
            # If not, it returns None
            log.debug(f"Query for doc_hash {doc_hash[:16]}... returned: {result[0] if result else 'None'}")
            return result[0] if result else None

def load_latest_chunks(doc_prefix: str) -> List[Dict[str, Any]]:
    """
    Returns: [{"chunk_id": "...", "text_hash": "...", "text": "...", "metadata": {...}}]
    """
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            base_name = doc_prefix
            
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

def archive_chunk(doc_id : str):
    """Archive old chunk: active_status='archived' to match your schema screenshot"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Update chunks table
            cur.execute("""
                UPDATE chunks 
                SET status = 'archived', 
                    archived_at = NOW()
                WHERE doc_id = %s AND status = 'active'
            """, (doc_id,))
            
            # Update document_versions table (Column is active_status per screenshot)
            cur.execute("""
                UPDATE document_versions
                SET active_status = 'archived', 
                    archived_at = NOW()
                WHERE doc_id = %s;
            """, (doc_id,))
            conn.commit()
def archive_chunk_by_chunk_id(chunk_id: str):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE chunks
                SET status = 'archived',
                    archived_at = NOW()
                WHERE chunk_id = %s
                  AND status = 'active'
            """, (chunk_id,))
            conn.commit()


# def upsert_doc(doc_id,version,extraction_stats,content_hash):
#     """Registers the parent document. MUST be called before upsert_chunks."""
#     with get_db_connection() as conn:
#         with conn.cursor() as curr:
#             # FIX: Convert dict to JSON string to avoid 'can't adapt type dict'
#             stats_json = json.dumps(extraction_stats) if isinstance(extraction_stats, dict) else extraction_stats
            
#             curr.execute("""
#                 INSERT INTO document_versions (
#                     doc_id, version, extraction_stats, active_status, content_hash
#                 )
#                 VALUES (%s, %s, %s, %s, %s)
                
#             """, (doc_id, version, stats_json, "active", content_hash))
#             conn.commit() # FIX: CRITICAL for Foreign Key Constraint
#             print(f"Document {doc_id} registered successfully.")
    
# def upsert_chunks(chunks: List[Dict[str, Any]]):
#     """
#     Batch upserts chunks by automatically resolving the parent UUID 
#     from the document_versions table.
#     """
#     if not chunks:
#         return

#     # 1. Extract the doc_id (filename) from the first chunk's metadata
#     # We assume all chunks in one batch belong to the same document
#     sample_doc_id = chunks[0].get("metadata", {}).get("doc_id")
    
#     if not sample_doc_id:
#         print("Error: No doc_id found in metadata. Cannot resolve Foreign Key.")
#         return

#     with get_db_connection() as conn:
#         with conn.cursor() as cur:
#             # 2. RESOLVE UUID: Find the PK from document_versions
#             cur.execute("""
#                 SELECT id FROM document_versions 
#                 WHERE doc_id = %s AND active_status = 'active'
#                 LIMIT 1
#             """, (sample_doc_id,))
            
#             result = cur.fetchone()
#             if not result:
#                 print(f"Error: Parent document '{sample_doc_id}' not found in DB.")
#                 return
            
#             parent_uuid = result[0]

#             # 3. PREPARE VALUES: Map chunks to the resolved UUID
#             values = []
#             for chunk in chunks:
#                 meta_dict = chunk.get("metadata", {})
#                 values.append((
#                     chunk["chunk_id"], 
#                     sample_doc_id,      # Original doc_id string (if column exists)
#                     chunk["text_hash"], 
#                     chunk["text"][:8000],
#                     chunk["embedding"],
#                     json.dumps(meta_dict), 
#                     "active",
#                     meta_dict.get("chunk_index", 0),
#                     meta_dict.get("quality_score", 0.0),
#                     parent_uuid        # THE RESOLVED FOREIGN KEY (chunk_doc_id_fkey)
#                 ))
            
#             # 4. EXECUTE: Insert with the resolved parent_uuid
#             # Note: Ensure the column names below match your exact DB schema
#             execute_values(cur, """
#                 INSERT INTO chunks (
#                     chunk_id, doc_id, text_hash, text, embedding, metadata, status, 
#                     chunk_index, quality_score, chunks_doc_id_fkey
#                 )
#                 VALUES %s
#             """, values)
            
#             conn.commit()
#             print(f"Successfully upserted {len(chunks)} chunks for {sample_doc_id}")

def upsert_doc(doc_id, version, extraction_stats, content_hash, conn=None):
    """Registers the parent document. If conn is provided, caller handles commit/rollback."""
    stats_json = json.dumps(extraction_stats) if isinstance(extraction_stats, dict) else extraction_stats
    
    # Logic to handle shared vs independent connection
    if conn is None:
        with get_db_connection() as shared_conn:
            return _execute_upsert_doc(shared_conn, doc_id, version, stats_json, content_hash, commit=True)
    else:
        return _execute_upsert_doc(conn, doc_id, version, stats_json, content_hash, commit=False)

def _execute_upsert_doc(conn, doc_id, version, stats_json, content_hash, commit=False):
    with conn.cursor() as curr:
        curr.execute("""
            INSERT INTO document_versions (
                doc_id, version, extraction_stats, active_status, content_hash
            )
            VALUES (%s, %s, %s, %s, %s)
        """, (doc_id, version, stats_json, "active", content_hash))
        if commit:
            conn.commit()
        log.info(f"Document {doc_id} registered successfully.")

def upsert_chunks(chunks: List[Dict[str, Any]], conn=None):
    """Batch upserts chunks. If conn is provided, caller handles commit/rollback."""
    if not chunks: return

    sample_doc_id = chunks[0].get("metadata", {}).get("doc_id")
    
    if conn is None:
        with get_db_connection() as shared_conn:
            _execute_upsert_chunks(shared_conn, chunks, sample_doc_id, commit=True)
    else:
        _execute_upsert_chunks(conn, chunks, sample_doc_id, commit=False)

def _execute_upsert_chunks(conn, chunks, sample_doc_id, commit=False):
    with conn.cursor() as cur:
        # 1. RESOLVE UUID
        cur.execute("""
            SELECT id FROM document_versions 
            WHERE doc_id = %s AND active_status = 'active'
            LIMIT 1
        """, (sample_doc_id,))
        
        result = cur.fetchone()
        if not result:
            raise Exception(f"Parent document '{sample_doc_id}' not found in DB.")
        
        parent_uuid = result[0]

        # 2. PREPARE VALUES
        values = []
        for chunk in chunks:
            meta_dict = chunk.get("metadata", {})
            # CRITICAL: Clean NUL characters for DB safety and LLM understanding
            cleaned_text = str(chunk["text"]).replace("\x00", "ti")[:8000]
            
            values.append((
                chunk["chunk_id"], sample_doc_id, chunk["text_hash"], 
                cleaned_text, chunk["embedding"], json.dumps(meta_dict), 
                "active", meta_dict.get("chunk_index", 0),
                meta_dict.get("quality_score", 0.0), parent_uuid
            ))
        
        # 3. EXECUTE
        execute_values(cur, """
            INSERT INTO chunks (
                chunk_id, doc_id, text_hash, text, embedding, metadata, status, 
                chunk_index, quality_score, chunks_doc_id_fkey
            )
            VALUES %s
        """, values)
        
        if commit:
            conn.commit()
        log.info(f"Successfully upserted {len(chunks)} chunks for {sample_doc_id}")