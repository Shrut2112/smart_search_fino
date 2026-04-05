import streamlit as st
import pandas as pd
import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Web Watcher", page_icon="🌐", layout="wide")
st.title("🌐 Web Crawl Dashboard")

st.markdown("Overview of Firecrawl snapshots and crawled website pages.")

snap_dir = Path(os.getenv("WEB_SNAPSHOTS_DIR", "data/web_snapshots")).resolve()

if not snap_dir.exists():
    st.warning(f"Snapshot directory not found: {snap_dir}")
    st.stop()

snapshots = []
for p in snap_dir.glob("*.json"):
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        snapshots.append(data)
    except Exception:
        continue

st.subheader("Crawled Pages Cache")
if snapshots:
    df = pd.DataFrame(snapshots)
    # Format the crawl_time from unix timestamp
    df['crawl_time'] = pd.to_datetime(df['crawl_time'], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')
    
    st.metric("Total Cached Pages", len(snapshots))
    st.dataframe(df[['url', 'doc_id', 'chunk_count', 'crawl_time']], use_container_width=True)
else:
    st.info("No snapshots found.")

st.divider()

st.subheader("Manual Controls")
st.info("To trigger a manual crawl, run `python watchers/web_watcher.py --single` or specify a `--url` parameter in your terminal.")

st.code(f"""
# Current Configuration (from .env)
FINO_WEBSITE_URL={os.getenv('FINO_WEBSITE_URL', 'https://www.fino.bank.in')}
WEB_CRAWL_INTERVAL_HOURS={os.getenv('WEB_CRAWL_INTERVAL_HOURS', '6')}
""", language="bash")
