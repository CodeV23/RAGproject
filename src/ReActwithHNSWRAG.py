#!/usr/bin/env python3
"""
ReAct RAG (HNSW + Ollama) — semantic ANN retriever
---------------------------------------------------
- Offline text corpus (.txt) → chunked
- SentenceTransformer embeddings (default: all-MiniLM-L6-v2)
- HNSW graph index (hnswlib) for fast ANN retrieval (cosine)
- ReAct reasoning loop (Thought → Action → Observation → Finish)
- Benchmarks: total time + RSS before/after + tracemalloc peak
- Windows-safe decoding

Run:
  python src/ReActwithHNSWRAG.py --build-index data/raw
  python src/ReActwithHNSWRAG.py --ask "Who is Springtrap in FNAF 3?"
"""

import os, re, time, json, pathlib, random, subprocess, psutil, tracemalloc, tempfile
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
import hnswlib
from sentence_transformers import SentenceTransformer

from rich import print as rprint
from rich.panel import Panel
import torch

# -------------------- CONFIG --------------------
DATA_DIR = pathlib.Path('./data')

INDEX_DIR = pathlib.Path('./index')
INDEX_DIR.mkdir(exist_ok=True)

DOCS_FILE = INDEX_DIR / 'docs.json'         # texts + metadata
EMB_FILE  = INDEX_DIR / 'embeddings.npy'    # float32 vectors
HNSW_FILE = INDEX_DIR / 'hnsw.bin'          # hnswlib serialized index

REASONING_MODEL = os.getenv('REASONING_MODEL', 'mistral:latest')
EMB_MODEL_NAME  = os.getenv('EMB_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')

MAX_STEPS = int(os.getenv('MAX_STEPS', '4'))
CHUNK_SIZE = 1600
CHUNK_OVERLAP = 120

# HNSW hyperparams (good defaults)
HNSW_SPACE = 'cosine'
HNSW_M = int(os.getenv('HNSW_M', '64'))
HNSW_EF_CONSTRUCTION = int(os.getenv('HNSW_EF_CONSTRUCTION', '200'))
HNSW_EF_SEARCH = int(os.getenv('HNSW_EF_SEARCH', '64'))

# -------------------- Bench helpers --------------------
def get_memory_mb():
    proc = psutil.Process()
    return proc.memory_info().rss / 1024 / 1024  # MB

def stopwatch(label):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            t1 = time.perf_counter()
            print(f"[TIMER] {label}: {t1 - t0:.3f}s")
            return out
        return wrapper
    return decorator

# -------------------- Normalization (for light filtering) --------------------
def normalize(text: str) -> str:
    s = text.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# -------------------- Load + Chunk --------------------
@stopwatch("Document loading & chunking")
def load_text_chunks(data_dir: pathlib.Path) -> List[Dict[str, Any]]:
    docs = []
    for path in data_dir.rglob('*.txt'):
        txt = path.read_text(encoding='utf-8', errors='ignore')
        step = CHUNK_SIZE - CHUNK_OVERLAP
        for i in range(0, len(txt), max(step, 1)):
            chunk = txt[i:i+CHUNK_SIZE]
            if not chunk:
                continue
            docs.append({
                'text': chunk,
                'meta': {'source': str(path), 'start': i, 'end': i+len(chunk)}
            })
    rprint(f"[INFO] Loaded {len(docs)} chunks from {data_dir}")
    return docs

# -------------------- HNSW Retriever --------------------
@dataclass
class HNSWRetriever:
    docs: List[Dict[str, Any]]
    model_name: str = EMB_MODEL_NAME
    space: str = HNSW_SPACE
    M: int = HNSW_M
    ef_construction: int = HNSW_EF_CONSTRUCTION
    ef_search: int = HNSW_EF_SEARCH

    def __post_init__(self):
        self.model = SentenceTransformer(self.model_name)
        # Build or load index based on presence of files
        if HNSW_FILE.exists() and EMB_FILE.exists() and DOCS_FILE.exists():
            # Fast path: load existing
            with open(DOCS_FILE, 'r', encoding='utf-8') as f:
                stored = json.load(f)
            self.docs = stored  # ensure in-memory docs match saved docs

            self.embeddings = np.load(EMB_FILE).astype('float32')
            dim = self.embeddings.shape[1]
            self.index = hnswlib.Index(space=self.space, dim=dim)
            self.index.load_index(str(HNSW_FILE))
            self.index.set_ef(self.ef_search)
            rprint(f"[green]Loaded HNSW index from disk ({len(self.docs)} vectors, dim={dim}).[/]")
        else:
            # Build from scratch (useful when called during --build-index)
            self._build_index(self.docs)

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        # SentenceTransformer returns float32 by default
        vecs = self.model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        return np.asarray(vecs, dtype='float32')

    @stopwatch("Embedding & HNSW index build")
    def _build_index(self, docs: List[Dict[str, Any]]):
        texts = [d['text'] for d in docs]
        self.embeddings = self._embed_texts(texts)
        dim = self.embeddings.shape[1]

        self.index = hnswlib.Index(space=self.space, dim=dim)
        self.index.init_index(max_elements=len(docs), ef_construction=self.ef_construction, M=self.M)
        ids = np.arange(len(docs))
        self.index.add_items(self.embeddings, ids)
        self.index.set_ef(self.ef_search)

        # Persist to disk for future runs
        with open(DOCS_FILE, 'w', encoding='utf-8') as f:
            json.dump(docs, f)
        np.save(EMB_FILE, self.embeddings)
        self.index.save_index(str(HNSW_FILE))

        rprint(f"[green]Built HNSW index: {len(docs)} vectors, dim={dim}.[/]")

    def query(self, q: str, k: int = 5) -> List[Dict[str, Any]]:
        qv = self._embed_texts([q])[0]
        labels, distances = self.index.knn_query(qv, k=k)
        labels = labels[0]
        # For cosine space, smaller distance is closer; no need to convert
        results = []
        for idx in labels:
            idx = int(idx)
            results.append(self.docs[idx])
        if not results:
            rprint('[yellow]No ANN results; returning random chunk (unexpected).[/]')
            return [random.choice(self.docs)]
        return results

# -------------------- REASONING --------------------
REACT_GUIDE = (
    "You are a STRICT ReAct agent. You MUST use ONLY the text in the latest Observation to answer.\n"
    "If the Observation does not contain the answer, respond exactly: 'I don't know.'\n"
    "Cite each substantive sentence with (source:FILE#start-end).\n\n"
    "Valid actions:\n- Retrieve[query]\n- Finish[final answer with citations]\n\n"
    "Format:\nThought: <short>\nAction: Retrieve[...] or Action: Finish[...]\n"
)

FINISH_RE = re.compile(r'Action:\s*Finish\[(.*)\]', re.DOTALL)
RETRIEVE_RE = re.compile(r'Action:\s*Retrieve\[(.*)\]')

def call_ollama(messages: List[Dict[str,str]]) -> str:
    # Use stdin and --no-spinner to avoid Windows arg length + spinner artifacts
    prompt = '\n'.join([m['content'] for m in messages])
    result = subprocess.run(
        ['ollama', 'run', REASONING_MODEL],
        input=prompt.encode('utf-8'),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        env={**os.environ, "NO_COLOR": "1", "TERM": "dumb"}
    )
    out = (result.stdout or b'').decode('utf-8', errors='replace')
    return out.strip()

def run_react(retriever: HNSWRetriever, question: str) -> Dict[str, Any]:
    messages = [
        {'role': 'system', 'content': REACT_GUIDE},
        {'role': 'user',  'content': f'Question: {question}\nStart.'}
    ]
    LOG_DIR = pathlib.Path('./logs'); LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f'react_{int(time.time())}.jsonl'
    steps = []

    for step in range(1, MAX_STEPS+1):
        content = call_ollama(messages)
        #trace
        thought_match = re.search(r'Thought:\s*(.*)', content)
        action_match = re.search(r'Action:\s*(.*)', content)
        if thought_match or action_match:
            print(f"\n[STEP {step}]")
            if thought_match:
                print(f"Thought → {thought_match.group(1).strip()}")
            if action_match:
                print(f"Action  → {action_match.group(1).strip()}")
            print('-' * 60)
        steps.append({'step': step, 'llm': content})
        m = FINISH_RE.search(content or '')
        if m:
            ans = m.group(1).strip()
            with open(log_path, 'a', encoding='utf-8') as f:
                json.dump({'step': step, 'answer': ans}, f); f.write('\n')
            return {'answer': ans, 'steps': steps}

        m = RETRIEVE_RE.search(content or '')
        query = m.group(1).strip() if m else question
        results = retriever.query(query, 5)

        # Light sanity filter (lexical overlap) to drop egregiously off-topic hits
        def _overlap(a: str, b: str) -> int:
            ta = set(normalize(a).split()); tb = set(normalize(b).split())
            return len(ta & tb)

        filtered = [r for r in results if _overlap(query, r['text']) >= 1] or results[:1]

        obs_text = '\n---\n'.join(
            f"(source:{r['meta']['source']}#{r['meta']['start']}-{r['meta']['end']})\n{r['text']}"
            for r in filtered
        )
        messages.append({'role': 'assistant', 'content': content})
        messages.append({'role': 'user', 'content': 'Observation:\n' + obs_text})

    messages.append({'role':'system','content':'You hit the step limit. Summarize best answer ONLY from Observation with citations or say I do not know.'})
    final = call_ollama(messages)
    return {'answer': final, 'steps': steps}

# -------------------- CLI (with timing + memory) --------------------
import torch
def main():
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
    print("=== ReAct RAG (HNSW) Benchmark ===")
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--build-index', type=str)
    p.add_argument('--ask', type=str)
    args = p.parse_args()

    tracemalloc.start()
    mem_before = get_memory_mb()
    t0 = time.perf_counter()

    if args.build_index:
        docs = load_text_chunks(pathlib.Path(args.build_index))
        # Build and persist HNSW index
        retriever = HNSWRetriever(docs)  # builds & saves if not present
        rprint('[green]Index built & saved.[/]')
    elif args.ask:
        # Load persisted docs + index
        if not (DOCS_FILE.exists() and HNSW_FILE.exists() and EMB_FILE.exists()):
            rprint("[red]No HNSW index found. Build it first with --build-index data/raw[/]")
            return
        with open(DOCS_FILE, 'r', encoding='utf-8') as f:
            docs = json.load(f)
        retriever = HNSWRetriever(docs)  # loads from disk fast
        t_react_start = time.perf_counter()
        out = run_react(retriever, args.ask)
        t_react_end = time.perf_counter()
        rprint(Panel.fit(out['answer'], title='Final Answer'))
        print(f"[TIMER] ReAct reasoning: {t_react_end - t_react_start:.3f}s")
    else:
        p.print_help(); return

    t1 = time.perf_counter()
    mem_after = get_memory_mb()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"[MEMORY] Before: {mem_before:.2f} MB | After: {mem_after:.2f} MB | Peak: {peak/1024/1024:.2f} MB")
    print(f"[TOTAL TIME] {t1 - t0:.3f}s")

if __name__ == '__main__':
    main()
