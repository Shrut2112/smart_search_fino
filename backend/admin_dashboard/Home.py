import streamlit as st
import pandas as pd
from backend.db_dashboard import get_document_summary, get_recent_activity, check_db_health
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Fino Admin Dashboard", page_icon="⚙️", layout="wide")
st.title("⚙️ Fino Pipeline Admin Dashboard")

# 1. System Health
st.subheader("System Health")
col1, col2, col3 = st.columns(3)
db_healthy = check_db_health()
col1.metric("Supabase Database", "Online" if db_healthy else "Offline", delta="Healthy" if db_healthy else "-Error", delta_color="normal" if db_healthy else "inverse")

pdf_dir = Path(os.getenv("WATCH_DIR", "data/pdfs")).resolve()
queue_size = len(list(pdf_dir.glob("*.pdf"))) if pdf_dir.exists() else 0
col2.metric("PDF Watch Queue", f"{queue_size} files")

web_dir = Path(os.getenv("WEB_SNAPSHOTS_DIR", "data/web_snapshots")).resolve()
web_files = len(list(web_dir.glob("*.json"))) if web_dir.exists() else 0
col3.metric("Web Snapshots Managed", f"{web_files} URLs")

st.divider()

# 2. Database Metrics
st.subheader("Database Metrics")
try:
    stats = get_document_summary()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Documents", stats["total_docs"])
    c2.metric("Active Documents", stats["active_docs"])
    c3.metric("Archived Documents", stats["archived_docs"])
    c4.metric("Total Chunks Indexed", stats["total_chunks"])
except Exception as e:
    st.error(f"Could not load database metrics: {e}")

st.divider()

# 3. Recent Activity
st.subheader("Recent Ingestion Activity")
try:
    activity = get_recent_activity(10)
    if activity:
        df = pd.DataFrame(activity)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No recent activity found.")
except Exception as e:
    st.error(f"Could not load recent activity: {e}")
