#!/usr/bin/env python3
"""E8 -- fair-baseline retrieval. The critic's key control: stop comparing flat-vs-oracle;
test the system class that actually competes (hybrid lexical+dense via RRF). Same E7 cases.
Question: does hybrid rescue implicit-reference recall, or do BOTH similarity signals lack
identity binding so fusing them adds nothing? Report recall@B for lex / dense / hybrid.
LLM-free, reproducible. (Rerank + real-EL are the LLM follow-ups.)
"""
import json, os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from model2vec import StaticModel
from e7_scale_retrieval import build

MODEL = StaticModel.from_pretrained("minishlab/potion-base-8M")

def ranks(sims):
    order = np.argsort(-sims)
    r = np.empty(len(sims), dtype=np.int64)
    r[order] = np.arange(len(sims))
    return r

def run(M, frag_type, Bs, seed=0, n_probe=50, k_rrf=60):
    frags, owner, probes = build(M, n_probe, frag_type, seed)
    vec = TfidfVectorizer().fit(frags)
    Xl = normalize(vec.transform(frags))
    Xd = normalize(np.asarray(MODEL.encode(frags), dtype=np.float32))
    res = {B: {"lex": 0, "dense": 0, "hybrid": 0} for B in Bs}
    for pr in probes:
        ql = normalize(vec.transform([pr["q"]]))
        ls = (Xl @ ql.T).toarray().ravel()
        qd = normalize(np.asarray(MODEL.encode([pr["q"]]), dtype=np.float32))
        ds = Xd @ qd[0]
        lr = ranks(ls); dr = ranks(ds)
        rrf = 1.0 / (k_rrf + lr) + 1.0 / (k_rrf + dr)
        di = pr["dec"]
        for B in Bs:
            res[B]["lex"] += 1 if lr[di] < B else 0
            res[B]["dense"] += 1 if dr[di] < B else 0
            topB = set(np.argpartition(-rrf, B - 1)[:B].tolist())
            res[B]["hybrid"] += 1 if di in topB else 0
    n = len(probes)
    return {("B%d" % B): {k: round(res[B][k] / n, 3) for k in res[B]} for B in Bs}

if __name__ == "__main__":
    out = {"note": "oracle entity index = 1.00 (feasibility ceiling, not shown)"}
    for ft in ["named", "coref_attr", "token_disjoint"]:
        out[ft] = {}
        for M in [2000, 10000, 50000]:
            r = run(M, ft, [5, 20])
            out[ft]["M%d" % M] = r
            print(ft, "M=%d" % M, json.dumps(r), flush=True)
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/e8_hybrid.json", "w"), indent=2)
    print("\nFINAL:", json.dumps(out))
