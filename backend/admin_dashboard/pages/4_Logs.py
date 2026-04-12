import streamlit as st
from backend.log_reader import read_logs
import os
from pathlib import Path

st.set_page_config(page_title="Logs Viewer", page_icon="📋", layout="wide")
st.title("📋 Live System Logs")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_LOG_DIR = _PROJECT_ROOT / "logs"

col1, col2, col3, col4 = st.columns(4)

log_source = col1.selectbox("Select Log File", ["pipeline.log", "watcher.log", "api.log"])
lines_to_read = col2.selectbox("Lines to tail", [50, 100, 500, 1000], index=1)
level_filter = col3.selectbox("Log Level", ["ALL", "INFO", "WARNING", "ERROR"])
search_term = col4.text_input("Search (Optional)")

if st.button("Refresh Logs"):
    st.rerun()

log_path = _LOG_DIR / log_source

if log_path.exists():
    lines = read_logs(
        str(log_path), 
        lines=lines_to_read, 
        level_filter=level_filter, 
        search_text=search_term if search_term else None
    )

    st.markdown(f"### {log_source}")

    if not lines:
        st.info("No matching logs found.")
    else:
        logs_str = "".join(lines)
        st.code(logs_str, language="bash")
else:
    st.warning(f"Log file not found at {log_path}")
