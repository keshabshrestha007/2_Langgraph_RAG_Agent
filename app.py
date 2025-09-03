import streamlit as st
import uuid
from langchain_core.messages import HumanMessage, AIMessage
from multistep_rag_system1 import graph  

st.set_page_config(page_title="Streaming RAG Chat", page_icon="⚡")
st.header("💬 Ask about 'Attention is All You Need' Research paper")

# Session state for chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

config = {"configurable":{"thread_id":str(uuid.uuid4())}}

for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.text(message['content'])

# User input
user_input = st.chat_input("Ask a question:", key="user_input")
if user_input:
    with st.chat_message('user'):
        st.text(user_input)
    human_msg = HumanMessage(content=user_input.strip())
    st.session_state.chat_history.append({"role":"user","content":user_input.strip()})


    ai_message = st.write_stream(
        message_chunk.content for message_chunk, _ in graph.stream(
            {"messages":[],"question":human_msg}, config=config, stream_mode="messages")
        if isinstance(message_chunk, AIMessage))
        
    st.session_state.chat_history.append({"role":"assistant","content":ai_message})

  
    
