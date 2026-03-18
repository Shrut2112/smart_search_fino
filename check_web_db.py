"""Verify web content in DB."""
from agents.db_hooks import init_db_pool, get_db_connection

init_db_pool()

with get_db_connection() as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT doc_id, version, active_status FROM document_versions WHERE doc_id LIKE 'web_%' AND active_status = 'active'")
        docs = cur.fetchall()
        print(f"\n=== ACTIVE WEB DOCUMENTS ({len(docs)}) ===")
        for d in docs:
            print(f"  {d[0]} | {d[1]} | {d[2]}")

        cur.execute("SELECT count(*), doc_id FROM chunks WHERE doc_id LIKE 'web_%' AND status = 'active' GROUP BY doc_id")
        chunk_counts = cur.fetchall()
        print(f"\n=== ACTIVE WEB CHUNK COUNTS ===")
        for c in chunk_counts:
            print(f"  {c[1]}: {c[0]} chunks")

        cur.execute("SELECT chunk_id, substring(text, 1, 100) FROM chunks WHERE doc_id LIKE 'web_%' AND status = 'active' ORDER BY chunk_index LIMIT 5")
        samples = cur.fetchall()
        print(f"\n=== SAMPLE CHUNKS (first 5) ===")
        for s in samples:
            print(f"  {s[0]}: {s[1][:80]}...")
