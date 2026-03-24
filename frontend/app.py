import streamlit as st
import requests
import os
import logging

try:
    from utils.logger import get_logger
    log = get_logger("frontend")
except ImportError:
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("frontend")

# Read the backend URL from an environment variable (default to localhost)
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Fino Smart Search", page_icon="🏦")
st.title("🏦 Fino Payments Bank Assistant")
st.markdown("Ask me anything about Fino's policies, directors, or operations.")

# 1. Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Handle User Input
if prompt := st.chat_input("How can I help you today?"):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 4. Generate Response via Backend API
    with st.chat_message("assistant"):
        with st.spinner("Searching records..."):
            try:
                log.info(f"Sending query to backend API: {prompt}")
                
                # Make POST request to the backend
                response = requests.post(
                    f"{BACKEND_URL}/chat", 
                    json={"query": prompt},
                    timeout=60 # Setting a reasonable timeout for LLM inference
                )
                response.raise_for_status() # Raise an exception for HTTP errors
                
                # Extract answer from the response JSON
                data = response.json()
                answer = data.get("answer", "I'm sorry, I couldn't process that request.")
                
                st.markdown(answer)
                log.info("Successfully received response from backend")
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except requests.exceptions.Timeout:
                error_msg = "The request timed out. The backend is taking too long to respond."
                log.error(error_msg)
                st.error(error_msg)
            except requests.exceptions.ConnectionError:
                error_msg = f"Failed to connect to backend at {BACKEND_URL}. Is it running?"
                log.error(error_msg)
                st.error(error_msg)
            except requests.exceptions.HTTPError as e:
                # Attempt to extract detail from FastAPI error response
                detail = str(e)
                try:
                    error_data = response.json()
                    if "detail" in error_data:
                        detail = error_data["detail"]
                except Exception:
                    pass
                error_msg = f"Backend returned an error: {detail}"
                log.error(error_msg)
                st.error(error_msg)
            except Exception as e:
                error_msg = f"An unexpected error occurred: {str(e)}"
                log.error(error_msg)
                st.error(error_msg)
