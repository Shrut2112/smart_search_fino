import streamlit as st
from answering_agent.answering_graph import create_graph

# 1. Initialize the Graph (Cache it so it doesn't reload every time)
@st.cache_resource
def load_graph():
    return create_graph()

st.set_page_config(page_title="Fino Smart Search", page_icon="🏦")
st.title("🏦 Fino Payments Bank Assistant")
st.markdown("Ask me anything about Fino's policies, directors, or operations.")

graph = load_graph()

# 2. Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Handle User Input
if prompt := st.chat_input("How can I help you today?"):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 5. Generate Response using LangGraph
    with st.chat_message("assistant"):
        with st.spinner("Searching records..."):
            try:
                # Invoke the graph
                # Using 'query' as the key because your AnswerState expects it
                result = graph.invoke({"query": prompt})
                response = result.get("answer", "I'm sorry, I couldn't process that request.")
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                st.error(error_msg)