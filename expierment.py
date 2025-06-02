"""
Run merge_datasets.py for every natural-language query in queries.csv
and save per-query outputs in ./results/.

queries.csv must have a column “NL Questions”.

Outputs per query (N = row index) inside ./results/
  ├─ qN_merged.csv         – merged training data
  ├─ qN_graph.dot          – join graph
  └─ qN_meta.json          – metadata:
        {
          "query"      : "...",
          "elapsed_s"  : 12.34,
          "r2_scores"  : { "dataset.csv": 0.81, ... }
        }
"""

import subprocess, csv, json, time, shutil, os, pathlib, sys, uuid
from pathlib import Path

ROOT        = Path(__file__).parent
PIPELINE    = ROOT / "the_pipeline_v2.py"       
QUERY_CSV   = ROOT / "queries_sig.csv"
RESULTS_DIR = ROOT / "results_geimi_sig"
RESULTS_DIR.mkdir(exist_ok=True)

def safe_slug(s: str, maxlen=40) -> str:
    slug = "".join(ch if ch.isalnum() else "_" for ch in s.lower())
    return slug[:maxlen] or str(uuid.uuid4())[:8]

# ------------------------------------------------------------------
# 1. read queries.csv
# ------------------------------------------------------------------
with QUERY_CSV.open(newline="") as f:
    reader = csv.DictReader(f)
    queries = [row["NL Questions"] for row in reader]

if not queries:
    sys.exit("queries.csv is empty or missing column 'NL Questions'.")

# ------------------------------------------------------------------
# 2. run pipeline for each query
# ------------------------------------------------------------------
summary = []
merged_tables = []   
for idx, question in enumerate(queries, 1):
    slug = f"q{idx}_{safe_slug(question)}"
    print(f"\n=== [{idx}/{len(queries)}] {question} ===")

    # build a _temporary_ config file overriding just the question & output
    tmp_cfg = RESULTS_DIR / f"{slug}.json"
    cfg = {
        "question": question,
        "datasets": ["datas/*.csv"],
        "out"     : str(RESULTS_DIR / f"{slug}_merged.csv"),
        "graph_out": str(RESULTS_DIR / f"{slug}_graph.dot"),
        # other keys fall back to DEFAULT_CONFIG inside merge_datasets.py
    }
    tmp_cfg.write_text(json.dumps(cfg, indent=2))

    t0 = time.time()

    proc = subprocess.run(
        [sys.executable, PIPELINE, str(tmp_cfg)],
        capture_output=True, text=True
    )
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"x Pipeline failed for query {idx}:\n", proc.stderr)
        continue

    # The pipeline prints R² lines; parse them if present
    r2_scores = {}
    for line in proc.stdout.splitlines():
        if line.startswith("MERGED::"):
            merged_tables = [t.strip() for t in line.split("::", 1)[1].split(",")]
        if line.strip().startswith("R2::"):
            # we’ll log lines like  R2:: dataset.csv 0.823
            _, dset, score = line.split()
            r2_scores[dset] = float(score)

    # write metadata
    meta = {
        "query": question,
        "elapsed_s": round(elapsed, 2),
        "r2_scores": r2_scores,
        "merged_tables": merged_tables 
    }
    meta_file = RESULTS_DIR / f"{slug}_meta.json"
    meta_file.write_text(json.dumps(meta, indent=2))

    summary.append(meta)
    print(f" finished in {elapsed:.1f}s  → {slug}_merged.csv")

# ------------------------------------------------------------------
# 3. global summary
# ------------------------------------------------------------------
(Path(RESULTS_DIR) / "summary.json").write_text(json.dumps(summary, indent=2))
print("\nAll done. Results in ./results/")
