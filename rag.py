# =============================== 
# RAG PIPELINE 
# =============================== 

import requests
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import streamlit as st
import pickle
import json
import os
from datetime import datetime
from urllib.parse import urljoin, urlparse

# ===============================
# CONFIG
# ===============================
OPENROUTER_API_KEY = "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"   # your OpenRouter key
MODEL_NAME = "deepseek/deepseek-r1-0528:free"

# ===============================
# Initialize models
# ===============================
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# FAISS index
dimension = 384
index = faiss.IndexFlatL2(dimension)

# Metadata storage
metadata = []

# ===============================
# Smart Chunking
# ===============================
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# ===============================
# Index chunks into FAISS
# ===============================
def index_chunks(texts, base_name):
    global index, metadata

    for src, text in texts:
        chunks = chunk_text(text)
        embeddings = embedding_model.encode(chunks)
        index.add(np.array(embeddings))
        for i, chunk in enumerate(chunks):
            metadata.append({
                'content': chunk,
                'doc_name': base_name,
                'url': src,
                'chunk_id': i,
                'scraped_at': datetime.now().isoformat()
            })

def process_pdf(file_path):
    text = (file_path)
    if not text.strip():
        return False
    chunks = chunk_text(text)
    embeddings = embedding_model.encode(chunks)

    global index, metadata
    index.add(np.array(embeddings))
    for i, chunk in enumerate(chunks):
        metadata.append({
            'content': chunk,
            'doc_name': os.path.basename(file_path),
            'url': f"file://{file_path}",
            'chunk_id': i,
            'scraped_at': datetime.now().isoformat()
        })

    save_index()
    return True

# ===============================
# Save/Load index
# ===============================
def save_index():
    faiss.write_index(index, "faiss_index.index")
    with open("metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

def load_index():
    global index, metadata
    if os.path.exists("faiss_index.index") and os.path.exists("metadata.pkl"):
        index = faiss.read_index("faiss_index.index")
        with open("metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        return True
    return False

# ===============================
# Intent detection
# ===============================
def detect_intent(query):
    q = query.lower()
    if "compare" in q:
        return "comparison"
    elif "summarize" in q or "summary" in q:
        return "summary"
    return "general"

# ===============================
# Query FAISS
# ===============================
def query_content(query):
    intent = detect_intent(query)
    query_embedding = embedding_model.encode([query])
    D, I = index.search(np.array(query_embedding), k=5)
    results = [metadata[i] for i in I[0] if 0 <= i < len(metadata)]
    return results, intent

# ===============================
# Prompt construction
# ===============================
def generate_prompt(query, results, intent):
    context = "\n".join([
        f"Source ({r.get('url', 'N/A')}): {r.get('content', '')}"
        for r in results if isinstance(r, dict)
    ])

    if intent == "comparison":
        return f"Compare the following information:\n\nContext:\n{context}\n\nQuery:\n{query}"
    elif intent == "summary":
        return f"Summarize the following information:\n\nContext:\n{context}\n\nQuery:\n{query}"
    else:
        return f"Answer the following query based on the context:\n\nContext:\n{context}\n\nQuery:\n{query}"

# ===============================
# Generate response via OpenRouter
# ===============================
def generate_response(query, results, intent):
    prompt = generate_prompt(query, results, intent)

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"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",
            "Content-Type": "application/json",
        },
        data=json.dumps({
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 400
        })
    )

    result = response.json()
    try:
        return result["choices"][0]["message"]["content"]
    except Exception:
        return f"Error in response: {result}"

# ===============================
# Streamlit UI
# ===============================
st.title("🔎 Chat with Website & PDF (RAG + DeepSeek via OpenRouter)")

if load_index():
    st.info("📂 Loaded existing index!")
else:
    st.warning("⚠️ No saved index found. Please process a website or PDF first.")

# Query input
query = st.text_input("Ask a question about the indexed content:")
if query:
    results, intent = query_content(query)
    response = generate_response(query, results, intent)
    st.write("### 🧠 Response:")
    st.write(response)
