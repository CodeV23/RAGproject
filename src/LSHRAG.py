"""
LSH RAG (LangChain + Ollama + HuggingFace)
Vedant - CS599 Final Project
-----------------------------------
Offline RAG with LSH for efficient similarity search.
Uses Ollama LLM + Chroma vector store + HuggingFace embeddings.
"""
import re
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


def tokenize(text, n=2):
    """
    Tokenize text into word n-grams (default: bigrams) for better lexical overlap.
    Based on Broder (1997), "On the Resemblance and Containment of Documents".
    """
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    words = text.split()
    if len(words) < n:
        return set(words)
    return set(" ".join(words[i:i+n]) for i in range(len(words) - n + 1))



# ------------------ 1. Load + Split ------------------
@stopwatch("Document loading & chunking")
def load_chunks(data_dir="data/raw"):
    docs = []
    for path in Path(data_dir).rglob("*.txt"):
        loader = TextLoader(str(path), encoding="utf-8")
        docs.extend(loader.load())
    if not docs:
        raise ValueError("No .txt files found in data/raw/")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=120)
    splits = splitter.split_documents(docs)
    print(f"[INFO] Loaded {len(docs)} docs → {len(splits)} chunks")
    return splits



# ------------------ 2. Build LSH Retriever ------------------
from datasketch import MinHash, MinHashLSH

class LSHRetriever:
    """
    Simple Locality-Sensitive Hashing retriever.
    Uses MinHash similarity over tokenized document text chunks.
    """

    def __init__(self, docs, num_perm=64, threshold=0.3):
        self.docs = docs
        self.num_perm = num_perm
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.minhashes = []


        
        print("[INFO] Building LSH index...")
        for i, doc in enumerate(docs):
            m = MinHash(num_perm=num_perm) #min hash essentially creates a fingerprint/signature for each document
            tokens = tokenize(doc.page_content) 
            for t in tokens:
                m.update(t.encode('utf8')) #updating the minhash with each token
            self.lsh.insert(f"doc_{i}", m) #inserting into LSH index, which means adding to grouped "buckets" based on similarity
            self.minhashes.append(m) #storage
        print(f"[INFO] Indexed {len(docs)} documents with LSH.")

    def get_relevant_documents(self, query, k=5):
        m = MinHash(num_perm=self.num_perm) #creates the minhash for the query
        query_tokens = tokenize(query)
        for t in query_tokens:
            m.update(t.encode('utf8')) #updates minhash with each token from query
        result_keys = list(self.lsh.query(m)) #queries LSH index for closest buckets/matches
        if result_keys:
            print(f"[INFO] Found {len(result_keys)} matching buckets via LSH → {result_keys[:k]}")
        else:
            print("[WARN] No matches found via LSH; returning top-1 random chunk.")

            return [self.docs[0]]
        # Return up to k docs
        docs = [self.docs[int(key.split("_")[1])] for key in result_keys[:k]]
        return docs


@stopwatch("Embedding skipped (LSH indexing only)")
def build_vectorstore(splits):
    retriever = LSHRetriever(splits)
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
        {
            "context": (lambda q: "\n---\n".join(
                [d.page_content for d in retriever.get_relevant_documents(q)])),
            "question": RunnablePassthrough(),
        } #format for utilizing LSH retriever

        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


# ------------------ 4. Benchmark Run ------------------
def main():
    print("=== LSH RAG Benchmark ===")
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
