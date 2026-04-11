import streamlit as st
import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Directory Watcher", page_icon="👁️", layout="wide")
st.title("👁️ File Watcher Status")

WATCH_DIR = Path(os.getenv("WATCH_DIR", "data/pdfs")).resolve()
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", "data/processed")).resolve()
FAILED_DIR = Path(os.getenv("FAILED_DIR", "data/failed")).resolve()

def get_file_info(directory):
    files = []
    if directory.exists():
        for p in directory.glob("*.*"):
            if p.is_file():
                stat = p.stat()
                files.append({
                    "Filename": p.name,
                    "Size (KB)": round(stat.st_size / 1024, 2),
                    "Modified": pd.to_datetime(stat.st_mtime, unit='s').strftime('%Y-%m-%d %H:%M:%S')
                })
    return files

col1, col2, col3 = st.columns(3)

queue_files = get_file_info(WATCH_DIR)
col1.metric("Pending Queue", f"{len(queue_files)} files")

proc_files = get_file_info(PROCESSED_DIR)
col2.metric("Successfully Processed", f"{len(proc_files)} files")

fail_files = get_file_info(FAILED_DIR)
col3.metric("Quarantined", f"{len(fail_files)} files")

st.divider()

col_a, col_b = st.columns(2)

with col_a:
    st.subheader(f"Pending Files (`{WATCH_DIR.name}/`)")
    if queue_files:
        st.dataframe(pd.DataFrame(queue_files))
    else:
        st.success("Queue is empty.")
        
with col_b:
    st.subheader(f"Failed Files (`{FAILED_DIR.name}/`)")
    if fail_files:
        st.dataframe(pd.DataFrame(fail_files))
        st.error("Some files failed to process and are quarantined.")
    else:
        st.success("No failed files.")
