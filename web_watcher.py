#!/usr/bin/env python3
"""
Web Watcher — Firecrawl-based website monitor for Fino Bank RAG.

Crawls fino.bank.in, detects content changes via hashing, and processes
changed pages through the EXISTING pipeline nodes:
  - clean_text_fixed()        from universal_parser_agent
  - semantic_chunking_production()  from universal_parser_agent
  - push_chunks() / archive   from db_hooks + process_chunk_func

Usage:
  python web_watcher.py --single      # One-shot crawl
  python web_watcher.py               # Scheduled re-crawl loop
  python web_watcher.py --url <URL>   # Scrape a single URL
"""

import os
import sys
import time
import hashlib
import argparse
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from agents.universal_parser_agent import (
    init_worker,
    clean_text_fixed,
    semantic_chunking_production,
)
from agents.db_hooks import (
    init_db_pool,
    get_db_connection,
    upsert_doc,
    upsert_chunks,
    archive_chunk,
    query_doc_exists,
)
from web_scraper.crawler import FinoCrawler
from utils.logger import get_logger

log = get_logger("web_watcher")

CRAWL_INTERVAL_HOURS = int(os.getenv("WEB_CRAWL_INTERVAL_HOURS", "6"))


# ─────────────── Process one page ─────────────── #

def process_web_page(crawler: FinoCrawler, url: str, markdown: str, page_metadata: dict) -> bool:
    """
    Process a single scraped web page through the existing pipeline nodes.
    Returns True if page was processed (new/changed), False if skipped.
    """
    content_hash = crawler.content_hash(markdown)
    doc_id = crawler.url_to_doc_id(url)

    # ── Change detection ──
    if crawler.is_unchanged(url, content_hash):
        log.info(f"⏭️  Unchanged: {url}")
        return False

    # ── Check if this content already exists in DB ──
    existing = query_doc_exists(content_hash)
    if existing:
        log.info(f"⏭️  Content hash already in DB: {url} -> {existing}")
        crawler.update_snapshot(url, content_hash, doc_id)
        return False

    log.info(f"🔄 Processing: {url} -> doc_id={doc_id}")

    # ── Build state dict (same shape as pipeline State) ──
    page_title = ""
    if isinstance(page_metadata, dict):
        page_title = page_metadata.get("title", "") or page_metadata.get("ogTitle", "") or ""

    state = {
        # Naming fields (set directly, no naming agent needed for web)
        "original_filename": url,
        "normalized_filename": f"{doc_id}.md",
        "base_doc_name": doc_id,
        "version": "v1",
        "revision_tag": None,
        "is_collision": False,
        "confidence": 1.0,
        "version_detected": False,
        # Parser fields
        "raw_text": markdown,
        "structured_tables": [],  # web pages: tables already in markdown
        "extraction_stats": {
            "total_pages": 1,
            "text_blocks": 1,
            "table_blocks": 0,
            "raw_chars": len(markdown),
            "ocr_pages": 0,
            "extraction_timestamp": datetime.utcnow().isoformat(),
            "source_type": "website",
            "url": url,
            "page_title": page_title,
        },
        "parsing_errors": [],
        "status": "extracted",
        # Fields needed by clean/chunk but not set yet
        "content_hash": "",
        "metadata": {},
        "chunks": [],
        "quality_score": 0.0,
        "tables": [],
        "old_chunks": [],
        "report": {},
        "actions": [],
        "status_comp": "",
        "matched_old_ids": set(),
        "to_archive": [],
        "file_hash": content_hash,
    }

    # ── REUSE existing pipeline nodes ──

    # Step 1: Clean text (same function as PDF pipeline)
    state = clean_text_fixed(state)
    if state.get("status") == "failed":
        log.error(f"❌ Clean failed: {url} - {state.get('parsing_errors')}")
        return False

    # Step 2: Semantic chunking + embedding (same function as PDF pipeline)
    state = semantic_chunking_production(state)
    if not state.get("chunks"):
        log.warning(f"⚠️  No chunks produced: {url}")
        return False

    log.info(f"✅ Chunked: {url} -> {len(state['chunks'])} chunks")

    # ── DB write (reuse existing db_hooks) ──

    try:
        with get_db_connection() as conn:
            # Archive existing chunks for this doc_id (if re-crawling)
            old_snap = crawler.snapshots.get(url)
            if old_snap:
                try:
                    archive_chunk(doc_id)
                    log.info(f"   Archived old chunks for {doc_id}")
                except Exception as e:
                    log.warning(f"   Archive skipped (may not exist): {e}")

            # Insert new doc version + chunks (atomic transaction)
            upsert_doc(
                doc_id=doc_id,
                version=state["version"],
                extraction_stats=state["extraction_stats"],
                content_hash=state["content_hash"],
                conn=conn,
            )

            upsert_chunks(state["chunks"], conn=conn)

            conn.commit()
            log.info(f"   DB committed: {doc_id} ({len(state['chunks'])} chunks)")

    except Exception as e:
        log.error(f"❌ DB write failed for {url}: {e}", exc_info=True)
        return False

    # ── Update snapshot ──
    crawler.update_snapshot(url, content_hash, doc_id, chunk_count=len(state["chunks"]))

    return True


# ─────────────── URL Filtering ─────────────── #

# File extensions to skip (PDFs handled by separate pipeline)
SKIP_EXTENSIONS = ('.pdf', '.doc', '.docx', '.xlsx', '.xls', '.zip', '.xml')

# URL path patterns to skip
SKIP_PATTERNS = [
    '/blog', '/careers', '/login', '/media', '/press',
    '/investor', '/sitemap', '/news-media',
]


def should_scrape(url: str) -> bool:
    """Check if a URL should be scraped (not a doc/media/blog link)."""
    lower = url.lower()
    if any(lower.endswith(ext) for ext in SKIP_EXTENSIONS):
        return False
    if any(pat in lower for pat in SKIP_PATTERNS):
        return False
    return True


# ─────────────── Orchestration ─────────────── #

def run_crawl(crawler: FinoCrawler, single_url: str = None):
    """
    Execute one crawl cycle.

    Strategy: map_site() discovers URLs (0 credits) → scrape_page() each
    individually (1 credit/page). This gives full coverage since Angular SPA
    limits the built-in crawl discovery.
    """
    if single_url:
        log.info(f"Scraping single URL: {single_url}")
        page = crawler.scrape_page(single_url)
        if page:
            process_web_page(crawler, page["url"], page["markdown"], page["metadata"])
        else:
            log.error(f"Failed to scrape: {single_url}")
        return

    # Step 1: Discover all URLs via map (0 credits)
    all_urls = crawler.map_site(limit=200)

    if not all_urls:
        log.warning("No URLs discovered from map")
        return

    # Step 2: Filter to scrapeable content pages
    urls = [u for u in all_urls if should_scrape(u)]
    log.info(f"Map: {len(all_urls)} total, {len(urls)} after filtering")

    processed = 0
    skipped = 0
    failed = 0

    for i, url in enumerate(urls, 1):
        log.info(f"[{i}/{len(urls)}] Scraping: {url}")

        # Quick check: if snapshot exists and content hasn't changed,
        # we still need to scrape to check — but we can skip processing later
        page = crawler.scrape_page(url)

        if not page:
            log.warning(f"[{i}/{len(urls)}] ⚠️  Empty result: {url}")
            failed += 1
            continue

        if process_web_page(crawler, page["url"], page["markdown"], page["metadata"]):
            processed += 1
        else:
            skipped += 1

    log.info(
        f"Crawl cycle done: {processed} processed, {skipped} skipped, "
        f"{failed} failed, {len(urls)} total"
    )


def main():
    parser = argparse.ArgumentParser(description="Fino Bank Web Watcher")
    parser.add_argument("--single", action="store_true", help="One-shot crawl then exit")
    parser.add_argument("--url", type=str, help="Scrape a single URL")
    parser.add_argument("--map-only", action="store_true", help="Only run map discovery, no scraping")
    parser.add_argument(
        "--interval", type=int, default=CRAWL_INTERVAL_HOURS,
        help=f"Hours between re-crawls (default: {CRAWL_INTERVAL_HOURS})"
    )
    args = parser.parse_args()

    # Initialize shared resources
    init_worker()  # embedding model
    init_db_pool()  # DB connection pool

    crawler = FinoCrawler()

    if args.map_only:
        log.info("Map-only mode")
        links = crawler.map_site()
        filtered = [u for u in links if should_scrape(u)]
        log.info(f"Discovered {len(links)} URLs, {len(filtered)} scrapeable:")
        for url in filtered:
            snap = crawler.snapshots.get(url)
            status = f"(cached: {snap.chunk_count} chunks)" if snap else "(new)"
            log.info(f"  {url} {status}")
        return

    if args.url:
        log.info(f"Single URL mode: {args.url}")
        run_crawl(crawler, single_url=args.url)
        return

    if args.single:
        log.info("Single crawl mode (map → scrape → process)")
        run_crawl(crawler)
        return

    # Scheduled loop
    log.info(f"Starting web watcher (interval: {args.interval}h)")

    while True:
        try:
            run_crawl(crawler)
        except Exception as e:
            log.error(f"Crawl cycle error: {e}", exc_info=True)

        log.info(f"Next crawl in {args.interval} hours")
        time.sleep(args.interval * 3600)


if __name__ == "__main__":
    main()

