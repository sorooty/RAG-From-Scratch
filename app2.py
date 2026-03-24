import os
import uuid
from pathlib import Path

import chromadb
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

# =========================
# CONFIG
# =========================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip()

DATA_DIR = Path("data")
PDF_DIR = DATA_DIR / "pdfs"
CHROMA_DIR = DATA_DIR / "chroma"

PDF_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# INIT
# =========================
st.set_page_config(page_title="Mini RAG Demo", page_icon=":books:", layout="wide")

@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=OPENAI_API_KEY)

@st.cache_resource
def get_chroma_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(name="rag_demo")
    return collection

openai_client = get_openai_client()
collection = get_chroma_collection()

# =========================
# UTILS
# =========================
def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages_text = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)

    return "\n".join(pages_text).strip()

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    """
    Découpe simple par caractères avec overlap.
    Suffisant pour un TP de démonstration.
    """
    text = text.replace("\x00", " ").strip()
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

def embed_texts(texts: list[str]) -> list[list[float]]:
    response = openai_client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=texts
    )
    return [item.embedding for item in response.data]

def embed_query(text: str) -> list[float]:
    response = openai_client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=[text]
    )
    return response.data[0].embedding

def add_pdf_to_vectorstore(pdf_path: Path) -> int:
    text = extract_text_from_pdf(pdf_path)

    if not text:
        return 0

    chunks = chunk_text(text)
    embeddings = embed_texts(chunks)

    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [
        {"source": pdf_path.name, "chunk_index": i}
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(chunks)

def search_relevant_chunks(query: str, top_k: int = 4):
    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    retrieved = []
    for doc, meta in zip(documents, metadatas):
        retrieved.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "chunk_index": meta.get("chunk_index", -1),
        })

    return retrieved

def build_messages(user_question: str, retrieved_chunks):
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[Source {i} | fichier={chunk['source']} | chunk={chunk['chunk_index']}]\n{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    system_message = (
        "Tu es un assistant pédagogique de RAG. "
        "Réponds uniquement à partir du contexte récupéré. "
        "Si l'information n'est pas présente dans le contexte, dis clairement que tu ne sais pas. "
        "À la fin, indique les sources utilisées."
    )

    user_message = f"""
Question :
{user_question}

Contexte récupéré :
{context}
"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

def answer_with_rag(user_question: str):
    retrieved_chunks = search_relevant_chunks(user_question, top_k=4)
    messages = build_messages(user_question, retrieved_chunks)

    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.2
    )

    answer = response.choices[0].message.content
    return answer, retrieved_chunks

# =========================
# SESSION STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================
# UI
# =========================
st.title(":books: Mini RAG Demo")
st.write("Importe des PDF puis pose des questions dessus.")

with st.sidebar:
    st.header("Import de PDF")

    uploaded_files = st.file_uploader(
        "Ajoute un ou plusieurs PDF",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Indexer les PDF"):
            total_chunks = 0

            for uploaded_file in uploaded_files:
                save_path = PDF_DIR / uploaded_file.name
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                nb_chunks = add_pdf_to_vectorstore(save_path)
                total_chunks += nb_chunks

            st.success(f"Indexation terminée. {total_chunks} chunks ajoutés.")

    st.divider()

    if st.button("Réinitialiser la base vectorielle"):
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            client.delete_collection("rag_demo")
        except Exception:
            pass
        st.cache_resource.clear()
        st.success("Base réinitialisée. Recharge la page.")

    st.caption("Embeddings via OpenAI API.")
    st.caption("Base vectorielle stockée localement avec Chroma.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Pose une question sur tes PDF...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Recherche + génération en cours..."):
            try:
                answer, retrieved_chunks = answer_with_rag(user_input)
                st.markdown(answer)

                with st.expander("Voir les passages récupérés"):
                    for i, chunk in enumerate(retrieved_chunks, start=1):
                        st.markdown(
                            f"**{i}. Source :** {chunk['source']} | **Chunk :** {chunk['chunk_index']}"
                        )
                        st.write(chunk["text"])
                        st.divider()

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as e:
                error_msg = f"Erreur : {e}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )