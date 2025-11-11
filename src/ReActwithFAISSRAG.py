"""
ReAct RAG (FAISS + Ollama) — GPU-Accelerated Semantic ANN Retriever
-------------------------------------------------------------------
- Offline text corpus (.txt) → chunked
- SentenceTransformer embeddings (default: all-MiniLM-L6-v2)
- FAISS GPU index (cosine similarity) for fast Approximate Nearest Neighbor (ANN) retrieval
- ReAct reasoning loop (Thought → Action → Observation → Finish)
- Benchmarks: total time + RSS before/after + tracemalloc peak
- Windows-safe decoding

Run:
python src/ReActwithFAISSRAG.py --build-index data/raw
python src/ReActwithFAISSRAG.py --ask "Who is Springtrap in FNAF 3?"
"""

import os, re, time, json, pathlib, random, subprocess, psutil, tracemalloc, threading
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np, faiss, torch
from sentence_transformers import SentenceTransformer
from rich import print as rprint
from rich.panel import Panel

# -------------------- CONFIG --------------------
DATA_DIR = pathlib.Path("./data")
INDEX_DIR = pathlib.Path("./index"); INDEX_DIR.mkdir(exist_ok=True)

DOCS_FILE = INDEX_DIR / "docs.json"
EMB_FILE  = INDEX_DIR / "embeddings.npy"
FAISS_FILE = INDEX_DIR / "faiss.index"

REASONING_MODEL = os.getenv("REASONING_MODEL", "mistral:instruct")
EMB_MODEL_NAME  = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MAX_STEPS = int(os.getenv("MAX_STEPS", "4"))
CHUNK_SIZE, CHUNK_OVERLAP = 1000, 120   # smaller, more precise chunks

# -------------------- Helpers --------------------
def get_memory_mb():
    return psutil.Process().memory_info().rss / 1024 / 1024

def stopwatch(label):
    def deco(fn):
        def wrap(*a, **kw):
            t0=time.perf_counter(); out=fn(*a,**kw); t1=time.perf_counter()
            print(f"[TIMER] {label}: {t1-t0:.3f}s"); return out
        return wrap
    return deco

def normalize(s:str)->str:
    s=s.lower(); s=re.sub(r"[^a-z0-9\s]"," ",s); return re.sub(r"\s+"," ",s).strip()

@stopwatch("Document loading & chunking")
def load_text_chunks(data_dir:pathlib.Path)->List[Dict[str,Any]]:
    docs=[]
    for p in data_dir.rglob("*.txt"):
        txt=p.read_text(encoding="utf-8",errors="ignore")
        step=CHUNK_SIZE-CHUNK_OVERLAP
        for i in range(0,len(txt),max(step,1)):
            chunk=txt[i:i+CHUNK_SIZE]
            if chunk:
                docs.append({"text":chunk,"meta":{"source":str(p),"start":i,"end":i+len(chunk)}})
    rprint(f"[INFO] Loaded {len(docs)} chunks from {data_dir}")
    return docs

# -------------------- FAISS Retriever --------------------
@dataclass
class FAISSRetriever:
    docs:List[Dict[str,Any]]
    dim:int=384
    index_path:str="index/faiss.index"

    def __post_init__(self):
        self.model=SentenceTransformer(EMB_MODEL_NAME)
        if os.path.exists(self.index_path):
            print("[INFO] Loading FAISS index from disk…")
            self.index=faiss.read_index(self.index_path)
        else:
            self._build_index()

    @stopwatch("Embedding & FAISS index build")
    def _build_index(self):
        texts=[d["text"] for d in self.docs]
        emb=np.array(self.model.encode(texts,show_progress_bar=True,normalize_embeddings=True),dtype="float32")
        np.save(EMB_FILE,emb); json.dump(self.docs,open(DOCS_FILE,"w",encoding="utf-8"))
        res=faiss.StandardGpuResources(); idx=faiss.IndexFlatIP(emb.shape[1])
        gpu_idx=faiss.index_cpu_to_gpu(res,0,idx); gpu_idx.add(emb)
        faiss.write_index(faiss.index_gpu_to_cpu(gpu_idx),self.index_path)
        self.index=gpu_idx; print(f"[OK] FAISS index built: {len(self.docs)} vectors (GPU).")

    def query(self,q:str,k:int=5):
        qv=np.array(self.model.encode([q],normalize_embeddings=True),dtype="float32")
        D,I=self.index.search(qv,k)
        return [self.docs[int(i)] for i in I[0]]

# -------------------- REASONING --------------------
REACT_GUIDE=(
"You are a STRICT ReAct agent. Use ONLY the latest Observation text.\n"
"If it lacks the answer, reply exactly: 'I don't know.'\n"
"Cite with (source:FILE#start-end).\n"
"Valid actions:\n- Retrieve[query]\n- Finish[final answer]\n"
"Format:\nThought: ...\nAction: Retrieve[...] or Action: Finish[...]"
)
FINISH_RE=re.compile(r"Action:\s*Finish\[(.*)\]",re.DOTALL)
RETRIEVE_RE=re.compile(r"Action:\s*Retrieve\[(.*)\]")

def call_ollama(messages:List[Dict[str,str]],timeout:int=300)->str:
    prompt="\n".join(m["content"] for m in messages)
    env={**os.environ,"NO_COLOR":"1","TERM":"dumb","OLLAMA_NO_SPINNER":"1","OLLAMA_SILENT":"1"}
    result=[None]
    def run_proc():
        result[0]=subprocess.run(
            ["ollama","run",REASONING_MODEL],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=False,env=env
        )
    t=threading.Thread(target=run_proc); t.start(); t.join(timeout)
    if t.is_alive(): return "[TIMEOUT] Ollama step exceeded limit."
    out=(result[0].stdout or b"").decode("utf-8","replace")
    return out.strip()

def run_react(retriever:FAISSRetriever,question:str)->Dict[str,Any]:
    msgs=[{"role":"system","content":REACT_GUIDE},
          {"role":"user","content":f"Question: {question}\nStart."}]
    log=pathlib.Path("./logs"); log.mkdir(exist_ok=True)
    steps=[]
    for s in range(1,MAX_STEPS+1):
        t0=time.perf_counter(); content=call_ollama(msgs); t1=time.perf_counter()
        print(f"[STEP {s}] Ollama call: {t1-t0:.2f}s")
        steps.append({"step":s,"llm":content})
        m=FINISH_RE.search(content or "")
        if m:
            ans=m.group(1).strip()
            json.dump({"step":s,"answer":ans},open(log/f"react_{int(time.time())}.jsonl","a",encoding="utf-8"))
            return {"answer":ans,"steps":steps}
        m=RETRIEVE_RE.search(content or ""); q=m.group(1).strip() if m else question
        res=retriever.query(q,5)
        def overlap(a,b): return len(set(normalize(a).split()) & set(normalize(b).split()))
        filt=[r for r in res if overlap(q,r["text"])>=1] or res[:1]
        obs="\n---\n".join(f"(source:{r['meta']['source']}#{r['meta']['start']}-{r['meta']['end']})\n{r['text'][:1000]}" for r in filt)
        msgs.append({"role":"assistant","content":content})
        msgs.append({"role":"user","content":"Observation:\n"+obs})
    msgs.append({"role":"system","content":"Step limit hit. Summarize best answer from Observation with citations."})
    final=call_ollama(msgs)
    return {"answer":final,"steps":steps}

# -------------------- CLI --------------------
def main():
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
    print("=== ReAct RAG (FAISS) Benchmark ===")
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--build-index",type=str)
    p.add_argument("--ask",type=str)
    args=p.parse_args()

    tracemalloc.start(); mem0=get_memory_mb(); t0=time.perf_counter()
    if args.build_index:
        docs=load_text_chunks(pathlib.Path(args.build_index))
        retriever=FAISSRetriever(docs); rprint("[green]FAISS index built & saved.[/]")
    elif args.ask:
        if not (DOCS_FILE.exists() and FAISS_FILE.exists()): rprint("[red]No FAISS index found.[/]"); return
        docs=json.load(open(DOCS_FILE,"r",encoding="utf-8")); retriever=FAISSRetriever(docs)
        tA=time.perf_counter(); out=run_react(retriever,args.ask); tB=time.perf_counter()
        rprint(Panel.fit(out["answer"],title="Final Answer"))
        print(f"[TIMER] ReAct reasoning: {tB-tA:.3f}s")
    else: p.print_help(); return
    t1=time.perf_counter(); mem1=get_memory_mb(); cur,peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
    print(f"[MEMORY] Before:{mem0:.1f}MB After:{mem1:.1f}MB Peak:{peak/1024/1024:.1f}MB")
    print(f"[TOTAL TIME] {t1-t0:.3f}s")

if __name__=="__main__":
    main()