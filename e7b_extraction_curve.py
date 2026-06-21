#!/usr/bin/env python3
"""E7b -- the extraction-cost curve. Don't stop at the oracle: how much of the oracle's
recall gain does a REALISTIC, free, LLM-free write-time resolver recover, and when does
it break? Resolver = write-time LOCALITY: link an implicit-reference fragment to the
nearest preceding named entity in its session. Knob = session ambiguity (a confounding
second entity introduced before the decisive fragment).
  FLAT   = no index (TF-IDF top-B over all M)
  AUTO   = locality index (cheap, fallible)
  ORACLE = perfect entity index (upper bound)
LLM-free, deterministic.
"""
import json, os, random
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from e7_scale_retrieval import ent, val, ATTRS

def decisive_text(frag_type, E, a, v):
    if frag_type == "named":
        return "The " + a + " of " + E + " is " + v + "."
    if frag_type == "coref_attr":
        return "The overseeing party recorded its " + a + " as " + v + "."
    return "The party in question is presently backed by " + v + "."

def build_sessions(M, n_probe, frag_type, confound_p, seed):
    r = random.Random("e7b:%d:%s:%.2f" % (seed, frag_type, confound_p))
    fm = []  # each: {text, sid, intro (entity name introduced or None), owner (true entity)}
    probes = []
    sid = 0
    def add(text, s, intro, owner):
        fm.append({"text": text, "sid": s, "intro": intro, "owner": owner})
        return len(fm) - 1
    for p in range(n_probe):
        E = ent(r); a = r.choice(ATTRS); v = val(r); s = sid; sid += 1
        add(E + " is a registered entity in the network.", s, E, E)
        if r.random() < confound_p:
            D = ent(r)
            add(D + " is a registered entity in the network.", s, D, D)  # confounder, nearer to decisive
        dec = add(decisive_text(frag_type, E, a, v), s, None, E)
        probes.append({"E": E, "a": a, "v": v, "dec": dec})
    while len(fm) < M:
        s = sid; sid += 1; O = ent(r)
        add(O + " is a registered entity in the network.", s, O, O)
        if len(fm) < M:
            add("The " + r.choice(ATTRS) + " of " + O + " is " + val(r) + ".", s, None, O)
    return fm, probes

def auto_owner(fm):
    by_sess = defaultdict(list)
    for i, m in enumerate(fm):
        by_sess[m["sid"]].append(i)
    auto = [None] * len(fm)
    for s, idxs in by_sess.items():
        cur = None
        for i in idxs:  # insertion order == session order
            if fm[i]["intro"] is not None:
                cur = fm[i]["intro"]
            auto[i] = cur
    return auto

def recall(M, frag_type, confound_p, B, seed=0, n_probe=50):
    fm, probes = build_sessions(M, n_probe, frag_type, confound_p, seed)
    texts = [m["text"] for m in fm]
    vec = TfidfVectorizer().fit(texts); X = normalize(vec.transform(texts))
    auto = auto_owner(fm)
    idx_o = defaultdict(list); idx_a = defaultdict(list)
    for i, m in enumerate(fm):
        idx_o[m["owner"]].append(i); idx_a[auto[i]].append(i)
    flat = aut = ora = link = 0
    for pr in probes:
        q = normalize(vec.transform([pr["q"] if "q" in pr else ("What is the " + pr["a"] + " of " + pr["E"] + "?")]))
        sims = (X @ q.T).toarray().ravel()
        if auto[pr["dec"]] == pr["E"]:
            link += 1
        k = min(B, len(sims)); topB = set(np.argpartition(-sims, k - 1)[:k].tolist())
        flat += 1 if pr["dec"] in topB else 0
        co = sorted(idx_o[pr["E"]], key=lambda i: -sims[i])[:B]
        ora += 1 if pr["dec"] in co else 0
        ca = sorted(idx_a[pr["E"]], key=lambda i: -sims[i])[:B]
        aut += 1 if pr["dec"] in ca else 0
    n = len(probes)
    return {"link_acc": round(link / n, 3), "flat": round(flat / n, 3),
            "auto": round(aut / n, 3), "oracle": round(ora / n, 3)}

if __name__ == "__main__":
    out = {}
    for ft in ["coref_attr", "token_disjoint"]:
        for M in [2000, 10000]:
            for cp in [0.0, 0.5]:
                key = "%s_M%d_confound%.1f" % (ft, M, cp)
                out[key] = recall(M, ft, cp, 20)
                print(key, json.dumps(out[key]), flush=True)
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/e7b_extraction_curve.json", "w"), indent=2)
    print("\nFINAL:", json.dumps(out))
