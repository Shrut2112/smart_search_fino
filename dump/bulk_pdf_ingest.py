import os
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import fitz  # PyMuPDF — for fast page count check
from database.db_hooks import init_db_pool, query_doc_exists, get_db_connection
from data_ingestion.graph.graph import main_graph
from data_ingestion.universal_parser_agent import init_worker
from utils.logger import get_logger
from watchers.watcher import sha256

log = get_logger("bulk_ingest")

# ─────────── Configuration ─────────── #
MAX_PAGES = 500  # Skip any PDF with more than this many pages

# ─────────── Rich Terminal Helpers ─────────── #

def progress_bar(current, total, width=40):
    pct = current / max(total, 1)
    filled = int(width * pct)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {current}/{total} ({pct*100:.1f}%)"

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

# ─────────── Pre-flight Checks ─────────── #

def get_page_count(pdf_path):
    """Fast page count using PyMuPDF (no full parsing)."""
    try:
        doc = fitz.open(pdf_path)
        count = len(doc)
        doc.close()
        return count
    except Exception:
        return -1  # Can't determine — let pipeline handle it

def get_existing_doc_ids():
    """Fetch all already processed document IDs from database."""
    existing = set()
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT doc_id FROM document_versions WHERE active_status = 'active'")
                for row in cur.fetchall():
                    existing.add(row[0])
    except Exception as e:
        log.error(f"Failed to fetch existing docs: {e}")
    return existing

# ─────────── Process Single File ─────────── #

def process_file(pdf_path, pipeline, existing_docs):
    start = time.time()
    try:
        # ── Pre-flight 1: Page count gate ──
        page_count = get_page_count(pdf_path)
        if page_count > MAX_PAGES:
            elapsed = time.time() - start
            reason = f"too_large:{page_count}_pages(max={MAX_PAGES})"
            log.warning(f"SKIPPED {pdf_path.name}: {reason}")
            return {
                "status": "skipped",
                "chunks": 0,
                "elapsed": elapsed,
                "skip_reason": reason,
                "error": None,
                "pages": page_count,
            }
        
        # ── Pre-flight 2: Fast Filename Dedup ──
        base_name = pdf_path.stem
        if "_v" in base_name:
            base_name = base_name.rsplit("_v", 1)[0]
            
        if base_name in existing_docs:
            elapsed = time.time() - start
            reason = f"already_in_db:{base_name}"
            log.info(f"SKIPPED {pdf_path.name}: {reason}")
            return {
                "status": "skipped",
                "chunks": 0,
                "elapsed": elapsed,
                "skip_reason": reason,
                "error": None,
                "pages": page_count,
            }
        
        file_hash = sha256(pdf_path)
        # ── Full pipeline ──
        result = pipeline.invoke({
            "original_filename": str(pdf_path),
            "file_hash": file_hash
        })
        
        elapsed = time.time() - start
        status = result.get("status_comp", "unknown")
        chunks = len(result.get("chunks", []))
        
        return {
            "status": status,
            "chunks": chunks,
            "elapsed": elapsed,
            "skip_reason": result.get("skip_reason", ""),
            "error": None,
            "pages": page_count,
        }
        
    except Exception as e:
        elapsed = time.time() - start
        log.error(f"FAILED {pdf_path.name}: {e}")
        return {
            "status": "failed",
            "chunks": 0,
            "elapsed": elapsed,
            "skip_reason": "",
            "error": str(e),
            "pages": -1,
        }

# ─────────── Main Bulk Ingest ─────────── #

def run_bulk_ingest():
    pdf_dir = Path(r"G:\My Drive\Fino_dump\Website_Backup")
    
    if not pdf_dir.exists():
        log.error(f"Directory not found: {pdf_dir}")
        return False
        
    pdf_files = sorted(list(pdf_dir.glob("*.pdf")))
    total = len(pdf_files)
    
    if total == 0:
        log.error(f"No PDFs found in {pdf_dir}")
        return False
    
    # ── Banner ──
    print("\n" + "=" * 70)
    print("  [PDF] BULK PDF INGESTION -- Smart Search Fino")
    print("=" * 70)
    print(f"  Source:    {pdf_dir}")
    print(f"  PDFs:     {total} files")
    print(f"  Max Pages: {MAX_PAGES} (PDFs beyond this are skipped)")
    print(f"  Model:    Cohere embed-multilingual-v3.0 (1024 dim)")
    print(f"  Target:   PostgreSQL + pgvector (Supabase)")
    print(f"  Mode:     RESUMABLE (already-processed files auto-skipped)")
    print(f"  Started:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")
    sys.stdout.flush()
    
    # ── Initialize ──
    print("[INIT] Initializing embedding model...")
    sys.stdout.flush()
    init_worker()
    print("[INIT] Initializing database pool...")
    sys.stdout.flush()
    init_db_pool()
    print("[INIT] Compiling LangGraph pipeline...")
    sys.stdout.flush()
    pipeline = main_graph()
    
    print("[INIT] Fetching processed files from DB...")
    sys.stdout.flush()
    existing_docs = get_existing_doc_ids()
    print(f"  -> Found {len(existing_docs)} already-processed documents.\n")
    
    print("[OK] All systems ready. Starting ingestion...\n")
    sys.stdout.flush()
    
    # ── Counters ──
    processed = 0
    skipped = 0
    skipped_large = 0
    skipped_dedup = 0
    failed = 0
    total_chunks = 0
    start_time = time.time()
    
    for i, pdf in enumerate(pdf_files, 1):
        # Progress header
        elapsed_so_far = time.time() - start_time
        avg_per_file = elapsed_so_far / max(i - 1, 1) if i > 1 else 0
        eta = avg_per_file * (total - i + 1)
        
        print(f"\n{'~' * 70}")
        print(f"  {progress_bar(i, total)}  ETA: {format_time(eta)}")
        print(f"  [FILE] {pdf.name}")
        print(f"{'~' * 70}")
        sys.stdout.flush()
        
        result = process_file(pdf, pipeline, existing_docs)
        pages = result.get("pages", -1)
        pages_str = f" ({pages}pg)" if pages > 0 else ""
        
        if result["status"] == "completed":
            processed += 1
            total_chunks += result["chunks"]
            print(f"  [OK] Completed{pages_str} -- {result['chunks']} chunks in {format_time(result['elapsed'])}")
        elif result["status"] == "skipped":
            skipped += 1
            reason = result["skip_reason"] or "duplicate"
            if "too_large" in reason:
                skipped_large += 1
                print(f"  [SKIP-LARGE] {pages_str} Skipped -- {reason}")
            elif "already_in_db" in reason:
                skipped_dedup += 1
                print(f"  [SKIP-DUP] Skipped -- {reason} (already processed, no re-work)")
            else:
                print(f"  [SKIP] Skipped -- {reason}")
        else:
            failed += 1
            print(f"  [FAIL] Failed -- {result['error']}")
        
        # Running stats
        print(f"  [STATS] Running: {processed} new | {skipped_dedup} already-done | {skipped_large} too-large | {skipped - skipped_dedup - skipped_large} other-skip | {failed} failed | {total_chunks} chunks")
        sys.stdout.flush()
    
    # ── Final Report ──
    total_time = time.time() - start_time
    print("\n\n" + "=" * 70)
    print("  [REPORT] INGESTION COMPLETE -- FINAL REPORT")
    print("=" * 70)
    print(f"  Total PDFs:       {total}")
    print(f"  Newly Processed:  {processed}")
    print(f"  Already in DB:    {skipped_dedup} (no re-work)")
    print(f"  Too Large (>{MAX_PAGES}pg): {skipped_large}")
    print(f"  Other Skipped:    {skipped - skipped_dedup - skipped_large}")
    print(f"  Failed:           {failed}")
    print(f"  Total Chunks:     {total_chunks}")
    print(f"  Total Time:       {format_time(total_time)}")
    print(f"  Avg/File:         {format_time(total_time / max(total, 1))}")
    print(f"  Finished:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")
    sys.stdout.flush()
    
    return True

# ─────────── Web Crawl Chain ─────────── #

def run_web_crawl():
    """Kick off full website re-crawl using web_watcher in single mode."""
    print("\n\n")
    print("=" * 70)
    print("=" * 70)
    print("||                                                              ||")
    print("||   [WEB] PHASE 2: WEBSITE RE-CRAWL (Firecrawl)               ||")
    print("||   Now crawling the entire Fino Bank website...               ||")
    print("||   Pages will be scraped, chunked, embedded, and stored.      ||")
    print("||                                                              ||")
    print("=" * 70)
    print("=" * 70)
    print(f"\n  Started:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Target:   https://www.fino.bank.in")
    print(f"  Mode:     Full site crawl (map -> scrape -> process -> DB)\n")
    sys.stdout.flush()
    
    # Import and run web watcher in single-shot mode
    from watchers.web_watcher import run_crawl
    from scraping.crawler import FinoCrawler
    
    # init_worker already called during PDF phase, DB pool already initialized
    crawler = FinoCrawler()
    
    crawl_start = time.time()
    run_crawl(crawler)
    crawl_time = time.time() - crawl_start
    
    print("\n" + "=" * 70)
    print("  [WEB] WEBSITE CRAWL COMPLETE")
    print("=" * 70)
    print(f"  Crawl Time: {format_time(crawl_time)}")
    print(f"  Finished:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")
    sys.stdout.flush()

# ─────────── Entry Point ─────────── #

if __name__ == "__main__":
    overall_start = time.time()
    
    # Phase 1: PDF Ingestion
    pdf_success = run_bulk_ingest()
    
    if not pdf_success:
        print("[ABORT] PDF ingestion failed. Aborting web crawl.")
        sys.exit(1)
    
    # Phase 2: Website Crawl (auto-chained)
    run_web_crawl()
    
    # Grand total
    total = time.time() - overall_start
    print("\n" + "=" * 70)
    print(f"  [DONE] ALL COMPLETE! Total pipeline time: {format_time(total)}")
    print("=" * 70 + "\n")
    sys.stdout.flush()
