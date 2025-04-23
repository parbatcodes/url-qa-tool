import streamlit as st
import requests
from bs4 import BeautifulSoup
from llama_index.core import VectorStoreIndex, Document
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
import os
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding



# Set up LLaMA (Ollama must be running)
llm = Ollama(model="gemma:2b")
embed_model = HuggingFaceEmbedding(model_name="nomic-ai/nomic-embed-text-v1",trust_remote_code=True)
Settings.llm = llm
Settings.embed_model = embed_model

st.set_page_config(page_title="URL Q&A Tool", layout="centered")
st.title("🔍 Ask Questions About Any Webpage")

# Store session state
if "index" not in st.session_state:
    st.session_state.index = None

# Step 1: Input URLs
url_input = st.text_area("Enter one or more URLs (one per line):", height=150)

if st.button("Ingest Content"):
    urls = [url.strip() for url in url_input.split("\n") if url.strip()]
    documents = []

    for url in urls:
        try:
            page = requests.get(url, timeout=20)
            soup = BeautifulSoup(page.content, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            documents.append(Document(text=text))
        except Exception as e:
            st.warning(f"❌ Could not process URL: {url}\nError: {e}")

    if documents:
        index = VectorStoreIndex.from_documents(documents)
        st.session_state.index = index
        st.success("✅ Content ingested successfully!")
    else:
        st.error("⚠️ No valid documents found.")

# Step 2: Ask a Question
if st.session_state.index:
    question = st.text_input("Ask a question about the content:")
    if st.button("Get Answer") and question:
        query_engine = st.session_state.index.as_query_engine()
        response = query_engine.query(question)
        st.markdown(f"**Answer:** {response.response}")
