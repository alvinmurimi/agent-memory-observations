#!/usr/bin/env python3
"""E7 dense baseline, LOCAL + full scale. Re-runs the scale recall@B test with SEMANTIC
embeddings (model2vec potion-base-8M, static, no quota) on the exact same cases as the
TF-IDF run. Decides whether the structure win over flat retrieval is real or a lexical
artifact. LLM-free, reproducible.
"""
import json, os
import numpy as np
from model2vec import StaticModel
from e7_scale_retrieval import build

MODEL = StaticModel.from_pretrained("minishlab/potion-base-8M")

def nrm(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

def dense_recall(M, frag_type, Bs, seed=0, n_probe=50):
    frags, owner, probes = build(M, n_probe, frag_type, seed)
    F = nrm(np.asarray(MODEL.encode(frags), dtype=np.float32))
    Q = nrm(np.asarray(MODEL.encode([pr["q"] for pr in probes]), dtype=np.float32))
    res = {}
    for B in Bs:
        hit = 0
        for j, pr in enumerate(probes):
            sims = F @ Q[j]
            k = min(B, len(sims))
            top = set(np.argpartition(-sims, k - 1)[:k].tolist())
            hit += 1 if pr["dec"] in top else 0
        res["B%d" % B] = round(hit / len(probes), 3)
    return res

if __name__ == "__main__":
    out = {"embedder": "model2vec/potion-base-8M", "dim": 256}
    for ft in ["named", "coref_attr", "token_disjoint"]:
        out[ft] = {}
        for M in [2000, 10000, 50000]:
            r = dense_recall(M, ft, [5, 20])
            out[ft]["M%d" % M] = r
            print(ft, "M=%d" % M, "dense=", json.dumps(r), flush=True)
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/e7_dense_local.json", "w"), indent=2)
    print("\nFINAL:", json.dumps(out))
