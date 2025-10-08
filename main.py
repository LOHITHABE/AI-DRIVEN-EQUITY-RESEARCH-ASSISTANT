import os
import streamlit as st
import pickle
import time
import requests
import re
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

# --- Load environment ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Streamlit UI Setup ---
st.set_page_config(page_title="News Research Tool", layout="wide")
st.markdown("## 🧠 News Research Tool 📈")
st.sidebar.title("Settings")

# --- Sidebar URL Inputs ---
st.sidebar.subheader("Enter up to 3 News Article URLs")
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    if url:
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store.pkl"
status_placeholder = st.empty()

# --- Helper: Clean text from HTML ---
def extract_clean_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg"]):
        tag.decompose()
    text = " ".join(soup.stripped_strings)
    return text

# --- Process URLs ---
if process_url_clicked and urls:
    status_placeholder.info("🔄 Loading and processing articles, please wait...")
    data = []

    for url in urls:
        try:
            res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            res.raise_for_status()

            # --- Smart extractor for Moneycontrol ---
            if "moneycontrol.com" in url:
                soup = BeautifulSoup(res.text, "html.parser")
                content = (
                    soup.find("div", {"class": "article_content"})
                    or soup.find("div", {"class": "content_wrapper"})
                    or soup.find("div", {"class": "article_page"})
                )
                if content:
                    text = " ".join(p.get_text() for p in content.find_all("p"))
                else:
                    text = extract_clean_text(res.text)
            else:
                text = extract_clean_text(res.text)

            if text.strip():
                data.append(Document(page_content=text.strip(), metadata={"source": url}))
            else:
                st.warning(f"⚠️ No readable content extracted from: {url}")

        except Exception as e:
            st.warning(f"⚠️ Failed to load {url}: {e}")

    if not data:
        st.error("❌ No valid content extracted. Please check the URLs and try again.")
    else:
        status_placeholder.success("✅ Articles processed successfully!")

        # --- Split into chunks ---
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", ","],
            chunk_size=1200,
            chunk_overlap=200,
        )
        docs = text_splitter.split_documents(data)
        st.info("🪄 Splitting text into chunks... Done!")

        # --- Build embeddings and FAISS index ---
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)

        with open(file_path, "wb") as f:
            pickle.dump(vectorstore, f)

        st.success("✅ Index built and saved successfully!")

# --- Question Input ---
query = st.text_input("Ask a question about these articles:")
answer = ""
docs = []

if query and os.path.exists(file_path):
    with open(file_path, "rb") as f:
        vectorstore = pickle.load(f)

    docs = vectorstore.similarity_search(query, k=3)

    # --- LLM with Groq ---
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    # --- RetrievalQA Chain ---
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
    )

    response = qa(query)
    answer = response["result"]
    docs = response["source_documents"]

# --- Display Answer ---
if answer:
    clean_answer = answer.strip()

    # Highlight numbers or prices (₹, lakh, crore, etc.)
    clean_answer = re.sub(
        r"(₹?\s?\d[\d,.\s]*\s?(?:lakh|crore|million|billion)?)",
        r"**\1**",
        clean_answer,
        flags=re.IGNORECASE,
    )

    # Collect sources
    faiss_sources = [doc.metadata.get("source", "") for doc in docs if "source" in doc.metadata]
    all_sources = sorted(set([src for src in faiss_sources if src]))

    # Display output nicely
    st.markdown(f"**Question:** {query}")
    st.markdown("### 🧩 Answer")
    st.markdown(clean_answer)

    if all_sources:
        st.markdown("### 🔗 Sources:")
        for src in all_sources:
            st.markdown(f"- [{src}]({src})")
