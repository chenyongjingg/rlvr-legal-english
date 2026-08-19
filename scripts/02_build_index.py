# -*- coding: utf-8 -*-
"""
LegalGraphRAG index builder v1 (data layer part 2).
- Embed corpus snippets + term definitions with MiniLM-L6-v2 -> FAISS flat index.
- Build a small hierarchical graph (eurlex<->oyez<->terms) as adjacency for graph-signal retrieval.
- Split snippets into train/val/test (80/10/10) for SFT data and eval.
Outputs under data/: faiss.index, embeddings.npy, ids.json, graph.json, split.json.
"""
import os, sys, json, argparse, pickle
import numpy as np

DATA = "/root/autodl-tmp/legal_english/data"
MODELS = "/root/autodl-tmp/legal_english/models"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--models", default=MODELS)
    args = ap.parse_args()
    data, mdir = args.data, args.models

    corpus = json.load(open(os.path.join(data, "corpus_v2.json"), encoding="utf-8"))
    snips = corpus["snippets"]
    terms = corpus["terms"]
    print(f"corpus: {len(snips)} snippets, {len(terms)} terms", flush=True)

    # ---- embeddings ----
    from sentence_transformers import SentenceTransformer
    emb_model = SentenceTransformer(os.path.join(mdir, "MiniLM-L6-v2"), device="cpu")
    texts = [s["text"] for s in snips] + [t["definition"] for t in terms]
    print(f"embedding {len(texts)} items...", flush=True)
    vecs = emb_model.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    np.save(os.path.join(data, "embeddings.npy"), vecs)
    ids = [s["id"] for s in snips] + [f"term_{t['term']}" for t in terms]
    json.dump({"ids": ids, "n_snippets": len(snips), "n_terms": len(terms)},
              open(os.path.join(data, "emb_ids.json"), "w"), ensure_ascii=False)

    import faiss
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine sim (MiniLM normalized)
    vecs_n = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    index.add(vecs_n.astype("float32"))
    faiss.write_index(index, os.path.join(data, "faiss.index"))
    print(f"faiss index: {index.ntotal} vectors, dim {dim}", flush=True)

    # ---- hierarchical graph (simple version: entity<->relationship adjacency) ----
    # nodes: snippets + terms; edges: term appears in snippet text (co-occurrence),
    #        eurlex snippet <-> oyez snippet by shared keyword (legal term).
    node_ids = ids
    graph = {"nodes": node_ids, "edges": []}
    term_to_idx = {t["term"].lower(): f"term_{t['term']}" for t in terms}
    # edge: term appears in a snippet
    for si, s in enumerate(snips):
        low = s["text"].lower()
        for term_name, tid in term_to_idx.items():
            if term_name in low:
                graph["edges"].append({"a": s["id"], "b": tid, "type": "contains_term"})
    # edge: eurlex <-> oyez via shared legal terms (weak link for graph traversal demo)
    oyez = [s for s in snips if s["source"] == "oyez"]
    eur = [s for s in snips if s["source"] == "eurlex"]
    for o in oyez:
        otext = o["text"].lower()
        linked = 0
        for e in eur:
            if linked >= 2:
                break
            if any(t["term"] in otext for t in terms[:20]):
                graph["edges"].append({"a": o["id"], "b": e["id"], "type": "related"})
                linked += 1
    print(f"graph: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges", flush=True)
    json.dump(graph, open(os.path.join(data, "graph.json"), "w"), ensure_ascii=False)

    # ---- train/val/test split (group by source doc title to avoid leakage) ----
    import random
    rng = random.Random(42)
    # group snippets by (source, title)
    groups = {}
    for s in snips:
        key = (s["source"], s["title"])
        groups.setdefault(key, []).append(s)
    keys = list(groups.keys())
    rng.shuffle(keys)
    n = len(keys)
    n_tr = int(n * 0.8); n_va = int(n * 0.1)
    split = {"train": [], "val": [], "test": []}
    for i, k in enumerate(keys):
        target = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
        for s in groups[k]:
            split[target].append(s)
    json.dump(split, open(os.path.join(data, "split.json"), "w"), ensure_ascii=False)
    print(f"split: train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
