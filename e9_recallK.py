#!/usr/bin/env python3
"""How deep is the implicit-reference fragment buried? Reports hybrid recall@K across K,
i.e. the smallest candidate window a reranker would need to even SEE the decisive fragment.
If it's buried beyond practical K, retrieve-then-rerank cannot help (a reranker reorders
the window, it cannot recover a first-stage recall miss). LLM-free.
"""
import json, os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from model2vec import StaticModel
from e7_scale_retrieval import build

MODEL = StaticModel.from_pretrained("minishlab/potion-base-8M")

def ranks(s):
    o = np.argsort(-s); r = np.empty(len(s), dtype=np.int64); r[o] = np.arange(len(s)); return r

def decisive_ranks(M, frag_type, n_probe, seed=0, k_rrf=60):
    frags, owner, probes = build(M, n_probe, frag_type, seed)
    vec = TfidfVectorizer().fit(frags); Xl = normalize(vec.transform(frags))
    Xd = normalize(np.asarray(MODEL.encode(frags), dtype=np.float32))
    out = []
    for pr in probes:
        ql = normalize(vec.transform([pr["q"]])); ls = (Xl @ ql.T).toarray().ravel()
        qd = normalize(np.asarray(MODEL.encode([pr["q"]]), dtype=np.float32)); ds = Xd @ qd[0]
        rrf = 1.0 / (k_rrf + ranks(ls)) + 1.0 / (k_rrf + ranks(ds))
        out.append(int(ranks(rrf)[pr["dec"]]))  # rank of decisive by RRF (0 = top; ranks() already sorts descending)
    return np.array(out)

if __name__ == "__main__":
    res = {}
    for M in [10000, 50000]:
        dr = decisive_ranks(M, "coref_attr", 30)
        row = {"median_rank": int(np.median(dr)), "min_rank": int(dr.min()), "max_rank": int(dr.max())}
        for K in [10, 40, 100, 500, 1000, 2000, 5000]:
            row["recall@%d" % K] = round(float((dr < K).mean()), 3)
        res["coref_attr_M%d" % M] = row
        print("M=%d" % M, json.dumps(row), flush=True)
    os.makedirs("results", exist_ok=True)
    json.dump(res, open("results/e9_recallK.json", "w"), indent=2)
    print("\nFINAL:", json.dumps(res))
