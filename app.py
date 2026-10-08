import numpy as np
import faiss
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq

st.set_page_config(page_title="Ask My Notes", page_icon="📘")


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def read_pdf(file):
    reader = PdfReader(file)
    pages = [page.extract_text() or "" for page in reader.pages]
    return " ".join(pages)


def split_text(text, size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + size].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap
    return chunks


def build_index(chunks, model):
    vectors = model.encode(chunks, normalize_embeddings=True)
    vectors = np.array(vectors, dtype="float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def retrieve(question, chunks, index, model, k=3):
    query = model.encode([question], normalize_embeddings=True)
    query = np.array(query, dtype="float32")
    scores, ids = index.search(query, k)
    return [chunks[i] for i in ids[0] if i != -1]


def generate_answer(question, context_chunks, api_key):
    client = Groq(api_key=api_key)
    context = "\n\n".join(context_chunks)
    prompt = (
        "Answer the question using only the context below. "
        "If the answer is not in the context, say \"I don't know based on this document.\"\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content


st.title("📘 Ask My Notes")
st.write("Upload a PDF and ask questions about it.")

api_key = st.sidebar.text_input("Groq API key", type="password")
uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded:
    model = load_model()

    if st.session_state.get("file_name") != uploaded.name:
        with st.spinner("Reading and indexing your PDF..."):
            text = read_pdf(uploaded)
            chunks = split_text(text)
            if not chunks:
                st.error("No readable text found. This may be a scanned PDF.")
                st.stop()
            st.session_state["chunks"] = chunks
            st.session_state["index"] = build_index(chunks, model)
            st.session_state["file_name"] = uploaded.name
        st.success(f"Indexed {len(st.session_state['chunks'])} chunks.")

    question = st.text_input("Ask a question about your PDF")

    if st.button("Get answer") and question:
        if not api_key:
            st.warning("Enter your Groq API key in the sidebar.")
        else:
            with st.spinner("Thinking..."):
                top_chunks = retrieve(
                    question,
                    st.session_state["chunks"],
                    st.session_state["index"],
                    model,
                )
                answer = generate_answer(question, top_chunks, api_key)
            st.subheader("Answer")
            st.write(answer)
            with st.expander("Source chunks used"):
                for i, chunk in enumerate(top_chunks, 1):
                    st.markdown(f"**Chunk {i}**")
                    st.write(chunk)
