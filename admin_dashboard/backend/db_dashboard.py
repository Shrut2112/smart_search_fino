import sys
from pathlib import Path
_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.append(_root)

from psycopg2.extras import RealDictCursor
from database.db_hooks import get_db_connection

def get_document_summary():
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    COUNT(*) as total_docs,
                    SUM(CASE WHEN active_status = 'active' THEN 1 ELSE 0 END) as active_docs,
                    SUM(CASE WHEN active_status = 'archived' THEN 1 ELSE 0 END) as archived_docs
                FROM document_versions
            """)
            doc_stats = cur.fetchone()
            
            cur.execute("""
                SELECT COUNT(*) FROM chunks
            """)
            chunk_stats = cur.fetchone()
            
            return {
                "total_docs": doc_stats[0] if doc_stats else 0,
                "active_docs": doc_stats[1] if doc_stats else 0,
                "archived_docs": doc_stats[2] if doc_stats else 0,
                "total_chunks": chunk_stats[0] if chunk_stats else 0
            }

def get_all_documents(limit=50):
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT doc_id, version, active_status, created_at, content_hash, extraction_stats
                FROM document_versions
                ORDER BY created_at DESC
                LIMIT %s
            """, (limit,))
            return [dict(row) for row in cur.fetchall()]

def get_chunks_for_doc(doc_id):
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT chunk_id, text, quality_score, metadata, status, chunk_index
                FROM chunks
                WHERE doc_id = %s
                ORDER BY chunk_index ASC
            """, (doc_id,))
            return [dict(row) for row in cur.fetchall()]

def get_recent_activity(limit=10):
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT doc_id, version, active_status, created_at
                FROM document_versions
                ORDER BY created_at DESC
                LIMIT %s
            """, (limit,))
            return [dict(row) for row in cur.fetchall()]

def check_db_health():
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                return True
    except Exception:
        return False
