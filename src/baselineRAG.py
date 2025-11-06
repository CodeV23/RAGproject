"""
Baseline RAG (LangChain + Ollama + HuggingFace)
Vedant Kharwal - CS506 Final Project
-----------------------------------
Offline baseline RAG for efficiency benchmarking.
Uses Ollama LLM + Chroma vector store + HuggingFace embeddings.
"""

import time, psutil, tracemalloc
from pathlib import Path


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# ------------------ Helpers ------------------
def get_memory_mb():
    proc = psutil.Process()
    return proc.memory_info().rss / 1024 / 1024  # MB

def stopwatch(label):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            end = time.perf_counter()
            print(f"[TIMER] {label}: {end - start:.3f}s")
            return result
        return wrapper
    return decorator


# ------------------ 1. Load + Split ------------------
@stopwatch("Document loading & chunking")
def load_chunks(data_dir="data/raw"):
    docs = []
    for path in Path(data_dir).rglob("*.txt"):
        loader = TextLoader(str(path), encoding="utf-8")
        docs.extend(loader.load())
    if not docs:
        raise ValueError("No .txt files found in data/raw/")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    splits = splitter.split_documents(docs)
    print(f"[INFO] Loaded {len(docs)} docs → {len(splits)} chunks")
    return splits


# ------------------ 2. Build Embeddings + Store ------------------
@stopwatch("Embedding & Chroma index build")
def build_vectorstore(splits):
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(splits, embedding=embedder)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return retriever


# ------------------ 3. Build the RAG Chain ------------------
def build_rag_chain(retriever):
    prompt_template = """You are a helpful assistant.
Answer the QUESTION using only the CONTEXT below.
If the answer is not present, say 'I don't know.'

CONTEXT:
{context}

QUESTION: {question}
ANSWER:"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm = OllamaLLM(model="mistral:latest")
    rag_chain = (
        {"context": retriever | (lambda docs: "\n---\n".join([d.page_content for d in docs])),
         "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


# ------------------ 4. Benchmark Run ------------------
def main():
    print("=== Baseline RAG Benchmark (Ollama + LangChain) ===")
    tracemalloc.start()
    mem_before = get_memory_mb()

    splits = load_chunks()
    retriever = build_vectorstore(splits)
    rag_chain = build_rag_chain(retriever)

    mem_after = get_memory_mb()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"[MEMORY] Before: {mem_before:.2f} MB | After: {mem_after:.2f} MB | Peak: {peak/1024/1024:.2f} MB")

    query = input("\nEnter your question: ")
    t0 = time.perf_counter()
    answer = rag_chain.invoke(query)
    t1 = time.perf_counter()

    print(f"\n[ANSWER]\n{answer}")
    print(f"[TOTAL TIME] {t1 - t0:.3f}s\n")


if __name__ == "__main__":
    main()
