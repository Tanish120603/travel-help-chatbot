from llama_index.core import SimpleDirectoryReader, ServiceContext, GPTVectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.llms.groq import Groq
import os, json
from llama_index.legacy.embeddings import HuggingFaceEmbedding
import streamlit as st
import time
system_prompt = """Your responses should be short, precise, and related to travel help only.
Bold important responses such as titles, headings, labels, bullet points.
Provide and bold related links.
Do not address queries unrelated to travel.
Politely decline unrelated topics by saying, 'I do not have information on this topic.'
Limit your responses to the inquiries posed, without digressing."""


context_window = 2048

groq_api_key = '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'#os.getenv("GROQ_API_KEY")
llm = Groq(api_key=groq_api_key, model="Llama3-8b-8192", temperature=0.9, system_prompt= system_prompt, context_window=context_window)




def vector_embedding():
    st.session_state.emb_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    st.session_state.data = SimpleDirectoryReader(input_dir="data/").load_data(show_progress = True)
    st.session_state.service_context = ServiceContext.from_defaults(llm=llm, embed_model=st.session_state.emb_model, system_prompt = system_prompt, context_window=context_window)
    st.session_state.index = GPTVectorStoreIndex.from_documents(st.session_state.data, service_context=st.session_state.service_context, show_progress = True)
    st.session_state.engine = st.session_state.index.as_chat_engine('context', system_prompt= system_prompt)

# Streamlit app
st.title('Travel Help Chatbot')
st.markdown('Ask me anything related to travel help!')

user_query = st.text_input("Your question:")

if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")
if user_query:
    resp = st.session_state.engine.chat(user_query)
    st.markdown(resp.response)
