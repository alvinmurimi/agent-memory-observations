#!/usr/bin/env python3
"""E7 -- the MEMORY question (not inference): under scale + a read budget B, does a
write-time entity index surface the decisive fragment better than flat retrieval?
This is the question E2-E6 did NOT test (they handed the reader the evidence).
LLM-free, deterministic, reproducible. recall@B = did the decisive fragment land in
the top-B the reader is allowed to see?

Fragment types for the decisive memory:
  named          -> "The {a} of {E} is {v}."            (entity named; lexical anchor exists)
  coref_attr     -> "The overseeing party recorded its {a} as {v}."  (shares attribute, NOT entity)
  token_disjoint -> "The party in question is presently backed by {v}." (shares neither)

FLAT  = TF-IDF cosine top-B over all M fragments (lexical baseline; dense is the next step).
STRUCT = oracle write-time entity index: scope to the queried entity's fragments, rank, top-B.
         (oracle = perfect write-time entity/coref resolution -> UPPER BOUND on what structure buys.)
"""
import json, os, random
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

SYL = ("tar von mal quen bryn sed wol kor pyx zel dree fenn gorm hask ilv jarr lome nyx orr pell "
       "rund sval threx ulm vask wend yarn zoth brae cind dorn esk fyl grau hev ish kael lorn myr "
       "oss prmagg quoll rhad skel tius uvarn vohl wyrm xil yaric zonn").split()
TAILS = ["Compact","Authority","Trust","Bureau","Consortium","Office","Assembly","Charter",
         "Foundation","Council","Holdings","Partners","Works","Group","Systems","Union"]
RA = ["operating","designated","controlling","assigned","lead","governing","oversight","primary",
      "appointed","registered","supervising","managing","sponsoring","auditing","custodial","executive"]
RB = ["sponsor","custodian","office","steward","underwriter","trustee","authority","guarantor",
      "liquidator","proprietor","registrar","controller","administrator","examiner","overseer","agent"]
ATTRS = [x + " " + y for x in RA for y in RB]

def wd(r):
    s = r.choice(SYL) + r.choice(SYL)
    return s[0].upper() + s[1:]
def ent(r):
    return wd(r) + " " + wd(r) + " " + r.choice(TAILS)
def val(r):
    return wd(r) + " " + r.choice(TAILS)

def build(M, n_probe, frag_type, seed):
    r = random.Random("scale:%d:%s:%d" % (seed, frag_type, M))
    frags = []; owner = []; probes = []
    for p in range(n_probe):
        E = ent(r); a = r.choice(ATTRS); v = val(r)
        if frag_type == "named":
            dec = "The " + a + " of " + E + " is " + v + "."
        elif frag_type == "coref_attr":
            dec = "The overseeing party recorded its " + a + " as " + v + "."
        else:
            dec = "The party in question is presently backed by " + v + "."
        di = len(frags); frags.append(dec); owner.append(p)
        frags.append(E + " is a registered entity in the network."); owner.append(p)  # intro names E
        probes.append({"p": p, "dec": di, "q": "What is the " + a + " of " + E + "?"})
    j = 0
    while len(frags) < M:
        frags.append("The " + r.choice(ATTRS) + " of " + ent(r) + " is " + val(r) + ".")
        owner.append(1000000 + j); j += 1
    return frags, owner, probes

def run(M, B, frag_type, seed=0, n_probe=50):
    frags, owner, probes = build(M, n_probe, frag_type, seed)
    vec = TfidfVectorizer().fit(frags)
    X = normalize(vec.transform(frags))
    ef = defaultdict(list)
    for i, o in enumerate(owner):
        ef[o].append(i)
    flat = struct = 0
    for pr in probes:
        q = normalize(vec.transform([pr["q"]]))
        sims = (X @ q.T).toarray().ravel()
        k = min(B, len(sims))
        topB = set(np.argpartition(-sims, k - 1)[:k].tolist())
        flat += 1 if pr["dec"] in topB else 0
        cand = sorted(ef[pr["p"]], key=lambda i: -sims[i])[:B]
        struct += 1 if pr["dec"] in cand else 0
    n = len(probes)
    return round(flat / n, 3), round(struct / n, 3)

if __name__ == "__main__":
    Ms = [2000, 10000, 50000]; Bs = [5, 20]; types = ["named", "coref_attr", "token_disjoint"]
    out = {}
    for ft in types:
        out[ft] = {}
        for M in Ms:
            for B in Bs:
                fl, st = run(M, B, ft)
                out[ft]["M%d_B%d" % (M, B)] = {"flat": fl, "struct": st}
                print(ft, "M=%d B=%d" % (M, B), "flat=%.3f struct=%.3f" % (fl, st), flush=True)
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/e7_scale_retrieval.json", "w"), indent=2)
    print("\nFINAL:", json.dumps(out))
