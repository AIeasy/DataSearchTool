"""

```bash
# 1. No arg  → uses DEFAULT_CONFIG at top of file
python merge_datasets.py

# 2. One arg → path to a config file (json / yaml / py)
python merge_datasets.py my_job.yaml

# 3. Many args → ignored (handy when orchestrators always pass argv)
```

Config file **formats supported**
--------------------------------
| Extension | Loader                                     |
|-----------|--------------------------------------------|
| `.json`   | `json.load`                                |
| `.yaml`   | `yaml.safe_load` (requires `pyyaml`)       |
| `.py`     | Executed; must expose a variable `CONFIG`  |

If no path is given, simply edit the `DEFAULT_CONFIG` dict below.

"""
from __future__ import annotations
from collections import defaultdict
import os, re, sys, json, csv, asyncio, textwrap, hashlib, sqlite3, importlib.util
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional,Dict, List, Tuple, Optional

import httpx                # async requests
import polars as pl          # fast, lazy DataFrames
from tqdm.asyncio import tqdm_asyncio  # async‑aware progress bar

try:
    import yaml  # optional, for .yaml configs
except ImportError:
    yaml = None
import itertools
# ---------------------------------------------------------------------------
# Configuration dataclass + default inline dict
# ---------------------------------------------------------------------------
@dataclass
class Config:
    question: str
    datasets: List[str]
    out: str = "merged_training.csv"
    target_feature: str | None = None
    left_model: str = "gemini-1.5-pro-latest"
    map_model: str = "gemini-1.5-pro-latest"
    join_model: str = "gemini-1.5-pro-latest"
    max_concurrency: int = 5
    cache_path: Path = Path(".llm_cache.sqlite")
    graph_out: str = "join_graph.dot"
    graph_fmt: str = "dot"
    api_base: str = ""   #API url
    api_key : str = ""  #API key
def _expand_globs(patterns: List[str]) -> List[Path]:
    return [p for pat in patterns for p in Path().glob(pat)]

DEFAULT_CONFIG = Config(
    question="please help me find a dataset for training linear regression model to predict red wine quality ",
    datasets=["datas/*.csv"],
    out="merged_training.csv",
    target_feature=None,
)
# ---------------------------------------------------------------------------
# SQLite cache (prompt → completion)
# ---------------------------------------------------------------------------
class LLMCache:
    def __init__(self, db: Path):
        self.db = db; self._init()
    def _init(self):
        with sqlite3.connect(self.db) as con:
            con.execute("CREATE TABLE IF NOT EXISTS cache (hash TEXT PRIMARY KEY, completion TEXT)")
    def _h(self, prompt:str)->str: return hashlib.sha256(prompt.encode()).hexdigest()
    async def get(self, prompt:str)->Optional[str]:
        h=self._h(prompt)
        with sqlite3.connect(self.db) as con:
            row=con.execute("SELECT completion FROM cache WHERE hash=?", (h,)).fetchone()
            return row[0] if row else None
    async def set(self, prompt:str, completion:str):
        h=self._h(prompt)
        with sqlite3.connect(self.db) as con:
            con.execute("INSERT OR REPLACE INTO cache VALUES(?,?)", (h, completion)); con.commit()

# ---------------------------------------------------------------------------
# Async OpenAI wrapper with back‑off + caching
# ---------------------------------------------------------------------------
class LLMClient:
    """Async wrapper around goapi.gptnb.ai, with sqlite prompt cache."""
    def __init__(self, cache: LLMCache, cfg: Config):
        self.cache = cache
        self.sem   = asyncio.Semaphore(cfg.max_concurrency)

        self.url  = cfg.api_base
        self.key  = os.getenv("GPTNB_API_KEY", cfg.api_key)
        if not self.key:
            raise RuntimeError("Set GPTNB_API_KEY or cfg.api_key")

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.key}",
        }
        self.client = httpx.AsyncClient(timeout=120)

    async def chat(self, model: str, messages: List[Dict]) -> str:
        payload = {"model": model, "stream": False, "messages": messages}
        prompt  = json.dumps(payload, sort_keys=True)

        if (cached := await self.cache.get(prompt)):
            return cached

        async with self.sem:
            for attempt in range(5):
                try:
                    r = await self.client.post(self.url,
                                               headers=self.headers,
                                               json=payload)
                    r.raise_for_status()
                    content = r.json()["choices"][0]["message"]["content"]
                    await self.cache.set(prompt, content)
                    return content
                except httpx.HTTPStatusError as e:
                    if e.response.status_code >= 500:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        raise
            raise RuntimeError("LLM call failed after retries")

    async def aclose(self):
        await self.client.aclose()

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
SYS_LEFT = ''''
    "You are an advanced data scientist. List the minimal set of features "
    "(underscore_case) needed to build a model, at least 10 features. The first feature should be a target feature that furfill user's need. Respond ONLY as:\n"
    "1.feature_name\n2.feature_name ..."
    "Here is the user's need: {question}" '''
MAP_RE  = re.compile(r"^[\s\-•\*]*([\w_ ]+)\s*→\s*([^\(]+)\(index.*")
KEY_RE  = re.compile(r"^[•\-*\s]*([\w_]+)")
RIGHT_MAP_TMPL = ''' You are an expert DBA, who expert at matching dataset columns to a list of
semantic feature names.

### INPUT
• Header (comma‑separated): {header}
• First row (comma‑separated): {row}
• Target features (in order): {targets}
• Special feature : {special}
• Dataset name: {dname}
### TASK
1. Split the header by commas into a list.  
   Use **0‑based indexing** (the first column is index 0).
2. For each target feature, find the column(s) whose meaning is an exact
   match or the closest reasonable synonym.  
   – If several columns apply, list them all.  
   – If nothing fits, return `none`.
3. Output a bullet list in the format  
   feature_name → column_name (index = n)
   keeping the original header spelling.
4. If a special feature is not None, find the column(s) whose meaning is an exact match or the closest reasonable synonym. If not found, make it None.
### OUTPUT
(Only the bullet list, nothing else,the special feature should be the first item in the list)
'''
RIGHT_KEY_TMPL = """You are a data engineer.
Dataset = {dname}
Header  = {header}
Row     = {row}
Which column(s) look like a unique entity ID suitable for joining? Bullet list or `none`.You are a data engineer and you are here to help me with the following tasks, given info:
Dataset = {dname}
Header  = {header}
Row     = {row}
Question: Which column(s) represent a *unique entity key* suitable for joining
this dataset to others?  Return a bullet list of candidate column names, or
`none` if no such column exists."""

JOIN_TMPL = """You are a data integration expert to help me identify the SAME real-world entities. Here are the dataset:
Dataset‑A key column = {keyA}
Samples‑A = {valsA}
Dataset‑B key column = {keyB}
Samples‑B = {valsB}
If these two columns identify the SAME real‑world entities (even with different codes like CHN vs PRC), reply `joinable`.
Otherwise reply `not joinable`.  Respond with exactly one word."""

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def max_r2_against_target(path: Path,
                          target_col: str,
                          numeric_sample: int | None = None) -> float:
    """
    Return the largest R² (Pearson-r²) between *any* numeric column
    and `target_col` in the CSV at `path`.

    Parameters
    ----------
    numeric_sample : int | None
        If set, randomly sample N rows before computing correlations
        (faster on huge files).  None → use full file.
    """
    df = pl.read_csv(path, infer_schema_length=1000)

    if target_col not in df.columns:
        return float("-inf")

    if numeric_sample and len(df) > numeric_sample:
        df = df.sample(numeric_sample, seed=0)

    # ── FIX: use the numeric-dtype list Polars provides
    numeric_cols = [
        c for c, t in df.schema.items()
        if t in pl.NUMERIC_DTYPES and c != target_col
    ]
    if not numeric_cols:
        return float("-inf")

    best_r2 = float("-inf")
    for col in numeric_cols:
        r = df.select(pl.corr(col, target_col)).item()
        if r is not None:
            best_r2 = max(best_r2, r * r)
    return best_r2
def preview_csv(path:Path)->tuple[str,str]:
    """Return header & first row comma‑joined without loading full file."""
    with path.open(newline="") as f:
        r=csv.reader(f); header=next(r); row=next(r,["" for _ in header])
    return ",".join(header), ",".join(row)
def dump_dot(graph: dict[Path, dict[Path, tuple[str, str]]],
             base: Path, out_file: str):
    """Write an undirected DOT file with edge labels = join keys."""
    with open(out_file, "w") as f:
        f.write("graph joins {\n")
        f.write('  node [shape=box];\n')
        for node in graph:
            style = ('style="filled", fillcolor="#ffd966"'
                     if node == base else "")
            f.write(f'  "{node.name}" [{style}];\n')
        for a, nbrs in graph.items():
            for b, (kA, kB) in nbrs.items():
                if a < b:   # avoid duplicate undirected edges
                    label = f'label="{kA} <--> {kB}"'
                    f.write(f'  "{a.name}" -- "{b.name}" [{label}];\n')
        f.write("}\n")
def dump_json(graph: dict[Path, dict[Path, tuple[str, str]]],
              base: Path, out_file: str):
    data: dict[str, Any] = {
        "base": base.name,
        "edges": [
            {"left": a.name, "right": b.name, "keyA": kA, "keyB": kB}
            for a, nbrs in graph.items()
            for b, (kA, kB) in nbrs.items() if a < b
        ],
    }
    with open(out_file, "w") as f:
        json.dump(data, f, indent=2)
async def build_join_graph(
    client: OpenAIClient,
    candidates: List[Tuple[Path, Dict[str, str], List[str]]],
    cfg: Config,
) -> Dict[Path, Dict[Path, Tuple[str, str]]]:
    """
    Build an *undirected* join graph.

    Parameters
    ----------
    candidates : List[(csv_path, mapping_dict, key_list)]
        The same triple you already have after Agent-2.
    Returns
    -------
    graph : {Path -> {Path -> (key_on_left, key_on_right)}}
        Adjacency map with the first working key-pair for each edge.
    """
    print("candidates:    ",candidates,'\n')
    # Load each CSV lazily only once
    frames = {p: pl.scan_csv(p) for p, *_ in candidates}

    graph: Dict[Path, Dict[Path, Tuple[str, str]]] = \
        defaultdict(dict)

    async def _first_joinable(
        pA: Path, keysA: List[str],
        pB: Path, keysB: List[str],
    ) -> Optional[Tuple[str, str]]:
        """
        Try every kA × kB; return (kA, kB) on first success.
        If either list is empty → immediately give up (can’t join).
        """
        if not keysA or not keysB:
            return None
        for kA in keysA:
            for kB in keysB:
                if await judge_joinable(
                    client,
                    frames[pA], frames[pB],
                    kA, kB, cfg
                ):
                    return kA, kB
        return None

    # Build tasks for every unordered pair (no duplicates)
    pair_meta = []          # remember which task → which pair
    tasks = []
    for (pA, _mA, keysA), (pB, _mB, keysB) in itertools.combinations(candidates, 2):
        pair_meta.append((pA, keysA, pB, keysB))
        print(pA.name, keysA, pB.name, keysB)
        tasks.append(asyncio.create_task(_first_joinable(pA, keysA, pB, keysB)))

    # Run them concurrently under tqdm
    for (pA, keysA, pB, keysB), result in zip(
            pair_meta,
            await tqdm_asyncio.gather(*tasks, desc="Agent-3 join scan")
    ):
        if result is not None:
            kA, kB = result
            graph[pA][pB] = (kA, kB)   # undirected → store both directions
            graph[pB][pA] = (kB, kA)

    # Ensure every candidate has at least an empty dict entry
    for p, *_ in candidates:
        graph.setdefault(p, {})

    return graph

# ---------------------------------------------------------------------------
# Core asynchronous pipeline
# ---------------------------------------------------------------------------
async def get_features(client:OpenAIClient, cfg:Config)->List[str]:
    txt=await client.chat(cfg.left_model,[{"role":"user","content":SYS_LEFT.format(question=cfg.question)}])
    feats=[ln.split(".",1)[-1].strip() for ln in txt.splitlines() if ln.strip()]
    if cfg.target_feature and cfg.target_feature not in feats:
        feats.append(cfg.target_feature)
    return feats,feats[0] if feats else None

async def map_one_dataset(client:OpenAIClient, path:Path, feats:List[str], cfg:Config):
    header,row=preview_csv(path)
    mapping_raw=await client.chat(cfg.map_model,[{"role":"user","content":RIGHT_MAP_TMPL.format(header=header,row=row,targets=", ".join(feats), special=feats[0] if feats else "",dname=path.name)}])
    key_raw=await client.chat(cfg.map_model,[{"role":"user","content":RIGHT_KEY_TMPL.format(dname=path.name,header=header,row=row)}])
    mapping={m.group(1).strip():m.group(2).strip() for m in (MAP_RE.match(l.strip()) for l in mapping_raw.splitlines()) if m}
    keys=[KEY_RE.match(l.strip()).group(1) for l in key_raw.splitlines() if KEY_RE.match(l.strip())]
    if keys and keys[0].lower()=="None": keys=[]
    return mapping, keys

async def judge_joinable(client:OpenAIClient, dfa:pl.LazyFrame, dfb:pl.LazyFrame, keyA:str, keyB:str, cfg:Config)->bool:
    if keyA not in dfa.columns or keyB not in dfb.columns:
        return False
    valsA=", ".join(map(str,(dfa.select(keyA).unique().limit(50).collect())[keyA].to_list()))
    valsB = ", ".join(map(str, (dfb.select(keyB).unique().limit(50).collect())[keyB].to_list()))
    resp=await client.chat(cfg.join_model,[{"role":"user","content":JOIN_TMPL.format(keyA=keyA,valsA=valsA,keyB=keyB,valsB=valsB)}])
    return resp.strip().lower().startswith("join")

async def process(cfg: Config):
    # ── 0. init client & ask Agent-1 for feature list ────────────────
    cache  = LLMCache(cfg.cache_path)
    client = LLMClient(cache, cfg)

    feats,special_feat = await get_features(client, cfg)
    print("\n> Target features:", ", ".join(feats))

    paths = _expand_globs(cfg.datasets)
    if not paths:
        print("No datasets found."); await client.aclose(); return

    # ── 1. Agent-2: map features/keys on every CSV ───────────────────
    print(f"\nMapping {len(paths)} datasets …")
    tasks   = [map_one_dataset(client, p, feats, cfg) for p in paths]
    results = await tqdm_asyncio.gather(*tasks, total=len(tasks), desc="Mapping")
    mapping_results = {p: res for p, res in zip(paths, results)}

    # ── 2. Build candidates list (flag: “has the target feature?”) ───
    print("feats: ", feats[0])
    print("special_feat: ", special_feat)
    target_feat = feats[0] if special_feat else cfg.target_feature
    candidates: list[tuple[Path, dict, list, bool]] = []

    for p, (mapping, keys) in mapping_results.items():
        matched_cnt = sum(v.lower() != "none" for v in mapping.values())
        if matched_cnt < 3:          # skip datasets with too few matches
            continue

        has_target = (
            target_feat
            and mapping.get(target_feat, "none").lower() != "none"
        )
        candidates.append((p, mapping, keys, has_target))

    if not candidates:
        print("No dataset matched any requested feature.")
        await client.aclose(); return

    # ── 3. Build the join-graph from *all* candidates ────────────────
    cand_triplets = [(p, m, k) for p, m, k, _ in candidates]
    join_graph = await build_join_graph(client, cand_triplets, cfg)

    print("\njoin-graph:")
    for node, nbrs in join_graph.items():
        print(f"  {node.name} — {', '.join(n.name for n in nbrs) or 'None'}")

    if cfg.graph_out:
        dump_dot(join_graph, cand_triplets[0][0], cfg.graph_out)
        print("Graph written to", cfg.graph_out)

    # ── 4. Pick base table ───────────────────────────────────────────
    #     • Prefer datasets with the target feature, ranked by max R²
    #     • Fallback: dataset with most matched features
    with_target  = [(p, m, k) for p, m, k, flag in candidates if flag]
    without_tgt  = [(p, m, k) for p, m, k, flag in candidates if not flag]

    if with_target:
        def _max_r2(t):
            p, mapping, *_ = t
            tgt_col = mapping.get(target_feat)
            return max_r2_against_target(p, tgt_col or "", numeric_sample=5000)
        base_path, base_map, base_keys = max(with_target, key=_max_r2)
        print("\nBase dataset ->", base_path.name,
              "(max R^2 = {:.3f})".format(_max_r2((base_path, base_map))))
    else:
        # no table had the target column → fallback to most matches
        base_path, base_map, base_keys, _ = max(
            candidates,
            key=lambda t: sum(v.lower() != "none" for v in t[1].values())
        )
        print("\nBase dataset ->", base_path.name,
              "(chosen by most matched features — target column absent)")

    # ── 5. Breadth-first merge only through connected edges ──────────
    merged  = pl.read_csv(base_path)
    visited = {base_path}
    queue   = [base_path]

    while queue:
        current = queue.pop(0)
        for neighbour, (k_cur, k_nb) in join_graph[current].items():
            if neighbour in visited:
                continue
            if k_cur not in merged.columns:
                print(f"   ! Skipping {neighbour.name}: key '{k_cur}' "
                    f"not in current merged table")
                continue
            print(f"\n -> Joining {neighbour.name} via {k_cur} <-> {k_nb}")
            df_nb = pl.read_csv(neighbour)

            if k_nb != k_cur:
                df_nb = df_nb.rename({k_nb: k_cur})

            # keep the key column, drop only the *other* duplicates
            dup_cols = [c for c in df_nb.columns
                        if c in merged.columns and c != k_cur]
            df_nb = df_nb.drop(dup_cols)

            merged = merged.join(df_nb, on=k_cur, how="inner")
            visited.add(neighbour)
            queue.append(neighbour)


    # ── 6. Done ──────────────────────────────────────────────────────
    print(f"\nJoined {len(visited)} tables; final shape = {merged.shape}")

    #print a parsable list so the batch runner can grab it
    print("MERGED::", ", ".join(p.name for p in visited))

    merged.write_csv(cfg.out, include_header=True)
    meta_path = Path(cfg.out).with_suffix(".meta.json")
    meta_path.write_text(json.dumps({
            "query"        : cfg.question,
            "merged_tables": [p.name for p in visited],
            "rows"         : merged.height,
            "columns"      : merged.width,
        }, indent=2))
    print("Meta written ->", meta_path.name)
    await client.aclose()


# ---------------------------------------------------------------------------
# Config loader (json / yaml / py / default)
# ---------------------------------------------------------------------------

def load_config(path: Optional[str] = None) -> Config:
    if path is None:
        return DEFAULT_CONFIG
    p=Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    suffix=p.suffix.lower()
    if suffix in {".yaml",".yml"}:
        if yaml is None:
            raise RuntimeError("pyyaml not installed – cannot load yaml config")
        with p.open() as f:
            data=yaml.safe_load(f)
    elif suffix==".json":
        with p.open() as f:
            data=json.load(f)
    elif suffix==".py":
        spec=importlib.util.spec_from_file_location("cfg_module", p)
        mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        if not hasattr(mod,"CONFIG"):
            raise AttributeError("Python config must define a CONFIG variable")
        data=mod.CONFIG
    else:
        raise ValueError(f"Unsupported config file type: {suffix}")
    return Config(**data)

# ---------------------------------------------------------------------------
# Main entry‑point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg_path=sys.argv[1] if len(sys.argv) > 1 else None
    cfg=load_config(cfg_path)
    asyncio.run(process(cfg))
