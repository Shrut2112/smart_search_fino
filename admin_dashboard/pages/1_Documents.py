import streamlit as st
import pandas as pd
import json
from backend.db_dashboard import get_all_documents, get_chunks_for_doc

st.set_page_config(page_title="Documents Explorer", page_icon="📁", layout="wide")
st.title("📁 Documents Explorer")

st.markdown("Browse documents indexed in Supabase and viewing their parsed chunks.")

docs = get_all_documents(limit=100)
if not docs:
    st.info("No documents found in the database.")
    st.stop()

doc_df = pd.DataFrame(docs)

st.subheader("Recent Documents")
st.dataframe(
    doc_df[['doc_id', 'version', 'active_status', 'created_at']], 
    use_container_width=True
)

st.divider()

st.subheader("Inspect Document Chunks")
selected_doc_id = st.selectbox("Select a Document ID", doc_df['doc_id'].tolist())

if selected_doc_id:
    # Show metadata for the selected doc
    doc_info = next((d for d in docs if d["doc_id"] == selected_doc_id), None)
    if doc_info and doc_info.get("extraction_stats"):
        with st.expander("View Document Extraction Metadata"):
            st.json(doc_info["extraction_stats"])

    chunks = get_chunks_for_doc(selected_doc_id)
    if chunks:
        st.write(f"Found **{len(chunks)}** chunks for doc `{selected_doc_id}`.")
        
        chunk_data = []
        for c in chunks:
            meta = c.get("metadata", {})
            if isinstance(meta, str):
                try: meta = json.loads(meta)
                except: meta = {}
            
            chunk_data.append({
                "Index": c.get("chunk_index"),
                "Type": meta.get("chunk_type", "text"),
                "Page": meta.get("page_number", "N/A"),
                "Quality": c.get("quality_score"),
                "Text Preview": c.get("text", "")[:150] + "...",
                "Status": c.get("status")
            })
            
        st.dataframe(pd.DataFrame(chunk_data), use_container_width=True)
        
        with st.expander("View Raw Chunk Content"):
            idx = st.number_input("Select Chunk Index", min_value=0, max_value=len(chunks)-1, value=0)
            if idx < len(chunks):
                st.markdown(f"**Chunk ID**: `{chunks[idx]['chunk_id']}`")
                st.text_area("Content", chunks[idx]["text"], height=300)
    else:
        st.warning(f"No chunks found for {selected_doc_id}.")
