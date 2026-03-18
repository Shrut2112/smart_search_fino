"""
Firecrawl API wrapper for Fino Bank website scraping.

Handles: crawling, single-page scraping, content hashing, and snapshot management.
Snapshots prevent redundant API calls by tracking content hashes per URL.

Uses firecrawl-py v4.18.1 (v2 API):
  - client.scrape(url, formats=..., actions=..., wait_for=..., only_main_content=...)
  - client.crawl(url, include_paths=..., exclude_paths=..., scrape_options=ScrapeOptions(...), ...)
  - client.map(url)  — 0-credit URL discovery
  - Returns Document objects with .markdown / .metadata attributes.
  Angular SPA requires wait_for + WaitAction for JS rendering.
"""

import os
import json
import hashlib
import time
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from firecrawl import FirecrawlApp
# from firecrawl.v2.types import ScrapeOptions, WaitAction

from firecrawl import FirecrawlApp
try:
    from firecrawl import V1ScrapeOptions as ScrapeOptions
except ImportError:
    ScrapeOptions = None

from dotenv import load_dotenv

from utils.logger import get_logger

load_dotenv()
log = get_logger("web.crawler")

# ─────────────── Config ─────────────── #

FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "")
FINO_WEBSITE_URL  = os.getenv("FINO_WEBSITE_URL", "https://www.fino.bank.in")
SNAPSHOTS_DIR     = Path(os.getenv("WEB_SNAPSHOTS_DIR", "data/web_snapshots")).resolve()

# Angular SPA needs time to render
SPA_WAIT_MS = 10000
#SPA_WAIT_ACTION = WaitAction(type="wait", milliseconds=SPA_WAIT_MS)

# Pages to crawl (high-value for RAG)
INCLUDE_PATHS = [
    "/ways-to-bank/*",
    "/savings-account*",
    "/current-account*",
    "/service-charges*",
    "/interest-rates*",
    "/faqs*",
    "/faq*",
    "/notices*",
    "/alerts*",
    "/grievance*",
    "/about*",
    "/deposit*",
    "/loan*",
    "/insurance*",
    "/fund-transfer*",
    "/debit-card*",
    "/cms*",
    "/aadhaar*",
    "/remittance*",
]

# Pages to skip — Firecrawl treats these as regex patterns
EXCLUDE_PATHS = [
    "/blog.*",
    "/careers.*",
    "/login.*",
    "/media.*",
    "/press.*",
    "/investor.*",
    ".*\\.pdf$",
    ".*\\.doc$",
    ".*\\.docx$",
    ".*\\.xlsx$",
    ".*\\.xls$",
    ".*\\.zip$",
]


# ─────────────── Snapshot ─────────────── #

@dataclass
class PageSnapshot:
    url: str
    content_hash: str
    crawl_time: float
    doc_id: str
    chunk_count: int = 0

    def to_dict(self):
        return asdict(self)


# ─────────────── Crawler ─────────────── #

class FinoCrawler:
    """Wraps Firecrawl SDK (v2 / v4.18+) with content-hash based change detection."""

    def __init__(self):
        if not FIRECRAWL_API_KEY:
            raise ValueError("FIRECRAWL_API_KEY not set in .env")

        self.app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
        self.snapshots: Dict[str, PageSnapshot] = {}
        SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        self._load_snapshots()

    # ── Snapshot persistence ──

    def _load_snapshots(self):
        for snap_file in SNAPSHOTS_DIR.glob("*.json"):
            try:
                data = json.loads(snap_file.read_text(encoding="utf-8"))
                self.snapshots[data["url"]] = PageSnapshot(**data)
            except Exception as e:
                log.warning(f"Bad snapshot {snap_file.name}: {e}")
        log.info(f"Loaded {len(self.snapshots)} web snapshots")

    def _save_snapshot(self, snap: PageSnapshot):
        safe_name = re.sub(r'[^\w]', '_', snap.url.split("//")[-1])[:80]
        path = SNAPSHOTS_DIR / f"{safe_name}.json"
        path.write_text(json.dumps(snap.to_dict(), indent=2), encoding="utf-8")
        self.snapshots[snap.url] = snap

    # ── Content hashing ──

    @staticmethod
    def content_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def is_unchanged(self, url: str, new_hash: str) -> bool:
        old = self.snapshots.get(url)
        return old is not None and old.content_hash == new_hash

    # ── URL → doc_id ──

    @staticmethod
    def url_to_doc_id(url: str) -> str:
        """Convert URL to a clean doc_id like 'web_faqs' or 'web_savings_account'."""
        path = url.split("//")[-1]            # remove scheme
        path = path.split("?")[0].split("#")[0]  # remove query/fragment
        path = path.rstrip("/")
        # remove domain
        parts = path.split("/", 1)
        slug = parts[1] if len(parts) > 1 else "home"
        # sanitize
        slug = re.sub(r'[^\w]', '_', slug)
        slug = re.sub(r'_+', '_', slug).strip('_')
        return f"web_{slug}" if slug else "web_home"

    # ── Extract markdown + metadata from a Document ──

@staticmethod
def _extract_page_data(result, fallback_url: str = "") -> Optional[Dict]:
    """Normalize Firecrawl response (Document object or dict) to a page dict."""
    markdown = ""
    metadata = {}
    url = fallback_url

    if isinstance(result, dict):
        markdown = result.get("markdown", "") or ""
        metadata = result.get("metadata", {}) or {}
        url = metadata.get("sourceURL", url) or metadata.get("url", url) or url
    else:
        # Document object (v2 SDK)
        markdown = getattr(result, "markdown", "") or ""
        meta_obj = getattr(result, "metadata", None)

        if meta_obj is not None:
            # Use metadata_dict property if available (returns clean dict)
            if hasattr(result, "metadata_dict"):
                metadata = result.metadata_dict
            elif isinstance(meta_obj, dict):
                metadata = meta_obj
            elif hasattr(meta_obj, "__dict__"):
                metadata = {k: v for k, v in meta_obj.__dict__.items() if v is not None}
            else:
                metadata = {}

        url = (
            metadata.get("source_url", url)
            or metadata.get("url", url)
            or metadata.get("sourceURL", url)
            or url
        )

    # ---- URL FIX ----
    # Fix Firecrawl metadata typo (ffino → fino)
    if url:
        url = url.replace("www.ffino.bank.in", "www.fino.bank.in")

    # If URL still empty, use fallback
    if not url:
        url = fallback_url
    # -----------------

    if not markdown or len(markdown.strip()) < 50:
        return None

    return {
        "url": url,
        "markdown": markdown.strip(),
        "metadata": metadata,
    }
    # ── Map site (0 credits — URL discovery) ──

    def map_site(self, limit: int = 200) -> List[str]:
        """
        Discover URLs on fino.bank.in without using crawl credits.
        Uses the /v2/map endpoint (free).

        Returns list of discovered URL strings.
        """
        log.info(f"Mapping {FINO_WEBSITE_URL} (limit={limit})")
        try:
            result = self.app.map(FINO_WEBSITE_URL, limit=limit)
            # result is MapData with .links list of SearchResult objects
            links = []
            if hasattr(result, 'links'):
                for item in result.links:
                    link_url = getattr(item, 'url', str(item)) if not isinstance(item, str) else item
                    links.append(link_url)
            log.info(f"Map discovered {len(links)} URLs")
            return links
        except Exception as e:
            log.error(f"Map failed: {e}")
            return []

    # ── Single page scrape ──

    def scrape_page(self, url: str) -> Optional[Dict]:
        """
        Scrape one page. Returns dict with 'markdown', 'metadata', 'url'
        or None on failure. Uses wait_for + WaitAction for Angular SPA rendering.
        """
        try:
            result = self.app.scrape(
                url,
                formats=["markdown"],
                only_main_content=True,
                wait_for=SPA_WAIT_MS,
                #actions=[SPA_WAIT_ACTION],
            )
            page = self._extract_page_data(result, fallback_url=url)
            if not page:
                log.warning(f"Empty/tiny content from {url}")
            return page

        except Exception as e:
            log.error(f"Scrape failed for {url}: {e}")
            return None

    # ── Full site crawl ──

    def crawl_site(self, limit: int = 50) -> List[Dict]:
        """
        Crawl fino.bank.in. Returns list of page dicts.
        Uses v2 SDK app.crawl() which handles async polling internally.

        Args:
            limit: Max pages to crawl (default 50, keeps free-tier credits safe).
        """
        log.info(f"Starting full crawl of {FINO_WEBSITE_URL} (limit={limit})")

        try:
            result = self.app.crawl(
                FINO_WEBSITE_URL,
                max_discovery_depth=3,
                include_paths=INCLUDE_PATHS,
                exclude_paths=EXCLUDE_PATHS,
                limit=limit,
                scrape_options=ScrapeOptions(
                    formats=["markdown"],
                    only_main_content=True,
                    wait_for=SPA_WAIT_MS,
                    # actions=[SPA_WAIT_ACTION],
                    # store_in_cache=True,
                ),
                poll_interval=5,
            )
        except Exception as e:
            log.error(f"Crawl failed: {e}")
            return []

        pages = []

        # result is a CrawlJob with .data list of Document objects
        data_list = result.data if hasattr(result, "data") else []
        credits = getattr(result, "credits_used", "?")

        log.info(f"Crawl returned {len(data_list)} raw results, credits_used={credits}")

        for item in data_list:
            page = self._extract_page_data(item)
            if not page:
                continue

            # Skip PDF/doc links that slipped through
            lower_url = page["url"].lower()
            if any(lower_url.endswith(ext) for ext in [".pdf", ".doc", ".docx", ".xlsx"]):
                log.debug(f"Skipping document link: {page['url']}")
                continue

            pages.append(page)

        log.info(f"Crawl complete: {len(pages)} pages with content")
        return pages

    # ── Update snapshot after successful processing ──

    def update_snapshot(self, url: str, content_hash: str, doc_id: str, chunk_count: int = 0):
        snap = PageSnapshot(
            url=url,
            content_hash=content_hash,
            crawl_time=time.time(),
            doc_id=doc_id,
            chunk_count=chunk_count,
        )
        self._save_snapshot(snap)
