"""
ReAct RAG (LangChain + Ollama + HuggingFace)
Vedant - CS599 Final Project
-----------------------------------
Offline RAG with LSH for efficient similarity search
and ReAct reasoning loop (Thought → Action → Observation)
"""

"""
To run, 
1. build LSH index by python src/ReActRAG.py --build-index data/raw
2. ask your query by python ReActRAG.py --ask "Query?"
"""
import os, re, time, json, pathlib, random, subprocess, psutil, tracemalloc
from dataclasses import dataclass
from typing import List, Dict, Any
from rich import print as rprint
from rich.panel import Panel
from datasketch import MinHash, MinHashLSH

# -------------------- CONFIG --------------------
DATA_DIR = pathlib.Path('./data')
INDEX_FILE = pathlib.Path('./index/lsh_index.json')
LOG_DIR = pathlib.Path('./logs'); LOG_DIR.mkdir(exist_ok=True)
REASONING_MODEL = os.getenv('REASONING_MODEL', 'mistral:latest')
MAX_STEPS = int(os.getenv('MAX_STEPS', '4'))

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

# -------------------- Normalization & tokenization --------------------
STOPWORDS = {
    'the','a','an','is','are','was','were','be','to','of','in','on','for','and','or','with','at','by','from',
    'who','what','where','when','why','how','does','do','did','about','please','explain'
}
ALIASES = {
    'fnaf': "five nights at freddy s",
    "five nights at freddy's": "five nights at freddy s",
    'fnf': "five nights at freddy s"
}

def normalize(text: str) -> str:
    s = text.lower()
    for k, v in ALIASES.items():
        s = s.replace(k, v)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(text: str) -> set:
    """Return union of unigrams + bigrams for robust LSH signatures."""
    s = normalize(text)
    words = s.split()
    grams = set(words) #unigrams
    grams.update(' '.join(words[i:i+2]) for i in range(max(0, len(words)-1))) #bigrams get added
    return grams

def keywordize(q: str) -> str:
    s = normalize(q)
    tokens = [w for w in s.split() if w not in STOPWORDS]
    return ' '.join(tokens)

# -------------------- Load + Chunk --------------------
@stopwatch("Document loading & chunking")
def load_text_chunks(data_dir: pathlib.Path) -> List[Dict[str, Any]]:
    docs = []
    for path in data_dir.rglob('*.txt'):
        txt = path.read_text(encoding='utf-8', errors='ignore')
        for i in range(0, len(txt), 1600 - 120):
            chunk = txt[i:i+1600]
            docs.append({
                'text': chunk,
                'meta': {'source': str(path), 'start': i, 'end': i+len(chunk)}
            })
    rprint(f"[INFO] Loaded {len(docs)} chunks from {data_dir}")
    return docs

# -------------------- LSH RETRIEVER --------------------
@dataclass
class LSHRetriever:
    docs: List[Dict[str, Any]]
    num_perm: int = 64
    threshold: float = 0.2  # slightly lower to reduce misses

    def __post_init__(self):
        self.lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        self.minhashes = []
        for i, d in enumerate(self.docs):
            m = MinHash(num_perm=self.num_perm)
            for t in tokenize(d['text']):
                m.update(t.encode('utf8'))
            self.lsh.insert(f'doc_{i}', m)
            self.minhashes.append(m)
        rprint(f'[green]Indexed {len(self.docs)} chunks with LSH.[/]')

    def _search_keys(self, q: str):
        m = MinHash(num_perm=self.num_perm)
        for t in tokenize(q):
            m.update(t.encode('utf8'))
        return list(self.lsh.query(m))

    def query(self, q: str, k: int = 5) -> List[Dict[str, Any]]:
        keys = self._search_keys(q)
        if not keys:
            q2 = keywordize(q)
            if q2 != q:
                keys = self._search_keys(q2)
                if keys:
                    rprint(f"[cyan]Retried with keywordized query:[/] '{q2}'")
        if not keys:
            rprint('[yellow]No match via LSH, fallback to random chunk.[/]')
            return [random.choice(self.docs)]
        return [self.docs[int(key.split('_')[1])] for key in keys[:k]]

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
    prompt = '\n'.join([m['content'] for m in messages])
    result = subprocess.run(
        ['ollama', 'run', REASONING_MODEL],
        input=prompt.encode('utf-8'), #keep the prompt as stdin to avoid prompt too long error on windows subprocess.run()
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        env={**os.environ, "NO_COLOR": "1", "TERM": "dumb"} #we added dumb so the spinner doesnt show up since with stdin the output will be spammed with spinner chars
    )
    out = (result.stdout or b'').decode('utf-8', errors='replace')
    return out.strip()


def run_react(retriever: LSHRetriever, question: str) -> Dict[str, Any]:
    messages = [
        {'role': 'system', 'content': REACT_GUIDE},
        {'role': 'user', 'content': f'Question: {question}\nStart.'}
    ]
    log_path = LOG_DIR / f'react_{int(time.time())}.jsonl'
    steps = []

    for step in range(1, MAX_STEPS+1):
        content = call_ollama(messages)
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

        # Filter obviously irrelevant chunks by token-overlap with the query
        def _overlap(a: str, b: str) -> int:
            ta = set(normalize(a).split())
            tb = set(normalize(b).split())
            return len(ta & tb)
        filtered = [r for r in results if _overlap(query, r['text']) >= 2]
        if not filtered:
            filtered = results[:1]  # keep at least one

        obs_text = '\n---\n'.join(
            f"(source:{r['meta']['source']}#{r['meta']['start']}-{r['meta']['end']})\n{r['text']}"
            for r in filtered
        )
        messages.append({'role': 'assistant', 'content': content})
        messages.append({'role': 'user', 'content': 'Observation:\n' + obs_text})

    # Step cap fallback
    messages.append({'role':'system','content':'You hit the step limit. Summarize best answer ONLY from Observation with citations or say I do not know.'})
    final = call_ollama(messages)
    return {'answer': final, 'steps': steps}

# -------------------- CLI (with timing + memory) --------------------
def main():
    print("=== ReAct RAG Benchmark ===")
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
        pathlib.Path('./index').mkdir(exist_ok=True)
        with open(INDEX_FILE, 'w', encoding='utf-8') as f:
            json.dump(docs, f)
        rprint('[green]Index saved.[/]')
    elif args.ask:
        with open(INDEX_FILE, 'r', encoding='utf-8') as f:
            docs = json.load(f)
        retriever = LSHRetriever(docs)
        out = run_react(retriever, args.ask)
        rprint(Panel.fit(out['answer'], title='Final Answer'))
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
