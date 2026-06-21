"""E1 - Multi-signal decisive-recall ablation. LLM-free (LSA + deterministic), rate-limit-immune.

For a question whose answer IS in the knowledge base, does a retrieval signal surface the DECISIVE
evidence (the notes needed to answer correctly) within a fixed budget k? This is the retrieval
precondition for "perfect recall on a knowledge base". A signal that lifts decisive-recall from <1 to
~1.0 on a question type where the semantic baseline fails is a genuine, tested epistemic-swim signal.

Decisive sets are fixed by hidden construction, never by the signal, so no signal wins by tautology.
Anti-tautology note: the gold/decisive note is NOT the most-recent, most-frequent, or last line.
"""
from __future__ import annotations
import argparse, json, random, re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

SYL = ("tar von mal quen bryn sed wol kor pyx zel dree fenn gorm hask ilv jarr lome nyx orr pell rund "
       "sval threx ulm vask wend yarn zoth brae cind dorn esk fyl grau hev ish kael lorn myr oss prmagg "
       "quoll rhad skel tius uvarn vohl wyrm xil yaric zonn").split()
TAILS = ["Compact","Authority","Trust","Bureau","Consortium","Office","Assembly","Charter","Foundation",
         "Council","Holdings","Partners","Works","Group","Systems","Union"]
ATTRS = ["operating sponsor","designated custodian","controlling office","governing trustee",
         "oversight authority","lead underwriter","registered proprietor","appointed liquidator"]

def W(rng, n=2): return "".join(rng.choice(SYL) for _ in range(n)).capitalize()
def person(rng): return f"{W(rng)} {W(rng)}"
def org(rng): return ("the " if rng.random() < .5 else "") + f"{W(rng)} {rng.choice(TAILS)}"

def make_embedder(corpus, dims=128, seed=0):
    # Stable lexical retriever: TF-IDF cosine, no SVD. Deterministic and well-behaved in corpus size.
    # (v1 used per-case truncated-SVD, whose basis flips with N and produced a non-monotonic artifact.)
    docs = [c for c in dict.fromkeys(corpus) if c.strip()]
    vec = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=1).fit(docs)
    return lambda texts: normalize(vec.transform(texts))

def rank(emb, query, notes):
    M = emb(notes); q = emb([query])               # sparse, L2-normalized rows
    s = (M @ q.T).toarray().ravel()                # cosine similarity
    return sorted(range(len(notes)), key=lambda i: -float(s[i]))

# ---------- case generators (decisive = indices of the notes needed to answer) ----------
def distractors(rng, n):
    return [f"The {rng.choice(ATTRS)} of {person(rng)} is {org(rng)}." for _ in range(n)]

def case_corrective(rng, N, k=4):
    e, a, vt, vp = person(rng), rng.choice(ATTRS), org(rng), org(rng)
    core = [f"The {a} of {e} is {vt}."]
    for _ in range(k):
        core.append(f"The {a} of {e} is {vp}.")
    core.append(f"An earlier note recording that the {a} of {e} is {vp} was filed in error and withdrawn.")
    dec = {0, len(core) - 1}                       # the truth assertion + the retraction
    notes = core + distractors(rng, N)
    return dict(qtype="corrective", notes=notes, question=f"What is the current {a} of {e}?",
                decisive=dec, entity=e, attr=a)

def case_temporal(rng, N, m=4):
    e, a = person(rng), rng.choice(ATTRS)
    years = sorted(rng.sample(range(2008, 2025), m))
    vals = [org(rng) for _ in range(m)]
    core = [f"As of {years[i]}, the {a} of {e} is {vals[i]}." for i in range(m)]
    j = rng.randrange(m)
    T = rng.randrange(years[j], years[j + 1]) if j < m - 1 else rng.randrange(years[j], 2027)
    notes = core + distractors(rng, N)
    return dict(qtype="temporal", notes=notes, question=f"As of {T}, what was the {a} of {e}?",
                decisive={j}, entity=e, attr=a, asof=T)

def case_multihop(rng, N):
    A, X, C = person(rng), org(rng), org(rng)
    ba, a = rng.choice(ATTRS), rng.choice(ATTRS)
    core = [f"The {ba} of {A} is {X}.", f"The {a} of {X} is {C}."]
    notes = core + distractors(rng, N)
    return dict(qtype="multihop", notes=notes, question=f"What is the {a} of the {ba} of {A}?",
                decisive={0, 1}, entity=A, attr=a, bridge=X, bridge_attr=ba)

GEN = {"corrective": case_corrective, "temporal": case_temporal, "multihop": case_multihop}

# ---------- retrieval signals (return a retrieved index set of size <= k) ----------
def sig_sem(case, emb, k):
    return set(rank(emb, case["question"], case["notes"])[:k])

def sig_entity(case, emb, k):
    ent = case["entity"].lower()
    ent_idx = [i for i, n in enumerate(case["notes"]) if ent in n.lower()]
    sem = rank(emb, case["question"], case["notes"])
    return set((ent_idx[:k] + [i for i in sem if i not in set(ent_idx)])[:k])

def sig_temporal(case, emb, k):
    if case["qtype"] != "temporal":
        return sig_sem(case, emb, k)
    T, ent = case["asof"], case["entity"].lower()
    cand = []
    for i, n in enumerate(case["notes"]):
        mm = re.search(r"As of (\d{4})", n)
        if mm and ent in n.lower() and int(mm.group(1)) <= T:
            cand.append((int(mm.group(1)), i))
    pick = {max(cand)[1]} if cand else set()       # latest validity-window start <= T
    sem = rank(emb, case["question"], case["notes"])
    return set((list(pick) + [i for i in sem if i not in pick])[:k])

def sig_iterative(case, emb, k):
    # the "swim": semantic seed, then re-retrieve using each seed note as a query (follows bridges)
    sem = rank(emb, case["question"], case["notes"])
    seed = sem[: max(2, k // 2)]
    extra = []
    for i in seed:
        extra += [j for j in rank(emb, case["notes"][i], case["notes"])[: max(2, k // 2)] if j != i]
    return set(list(dict.fromkeys(list(seed) + extra))[:k])

def sig_hybrid(case, emb, k):
    s = set(sig_entity(case, emb, max(2, k // 2)))
    if case["qtype"] == "temporal":
        s |= sig_temporal(case, emb, max(2, k // 2))
    if case["qtype"] == "multihop":
        s |= sig_iterative(case, emb, k)
    sem = rank(emb, case["question"], case["notes"])
    return set((list(s) + [i for i in sem if i not in s])[:k])

SIG = {"sem_topk": sig_sem, "entity": sig_entity, "temporal": sig_temporal,
       "iterative": sig_iterative, "hybrid": sig_hybrid}

# ---------- anti-tautology canaries on the decisive note (must NOT trivially equal recall) ----------
def canary_flags(case):
    dec = case["decisive"]; L = len(case["notes"])
    return {"decisive_is_last_line": (L - 1) in dec and case["qtype"] != "corrective"}

def run(qtypes, Ns, ks, per, seed):
    rows = []
    for qt in qtypes:
        for N in Ns:
            cases = [GEN[qt](random.Random(f"{seed}:{qt}:{i}:{N}"), N) for i in range(per)]
            embs = [make_embedder(c["notes"] + [c["question"]], seed=seed) for c in cases]
            for k in ks:
                for sg in SIG:
                    rec = sum(1 for c, e in zip(cases, embs) if c["decisive"] <= SIG[sg](c, e, k))
                    rows.append(dict(qtype=qt, N=N, k=k, signal=sg, n=per, recall=round(rec / per, 3)))
            best = {sg: [r for r in rows if r["qtype"] == qt and r["N"] == N and r["k"] == ks[-1]
                         and r["signal"] == sg][0]["recall"] for sg in SIG}
            print(f"{qt:11s} N={N:4d} k={ks[-1]:2d} | " + "  ".join(f"{sg}={best[sg]:.2f}" for sg in SIG), flush=True)
    return rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qtypes", default="corrective,temporal,multihop")
    p.add_argument("--Ns", default="20,100,500")
    p.add_argument("--ks", default="5,10")
    p.add_argument("--per", type=int, default=40)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="results/exp1_recall.json")
    a = p.parse_args()
    rows = run(a.qtypes.split(","), [int(x) for x in a.Ns.split(",")],
               [int(x) for x in a.ks.split(",")], a.per, a.seed)
    from pathlib import Path
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"config": vars(a), "rows": rows}, indent=2), encoding="utf-8")
    print("\nWROTE", a.out)

if __name__ == "__main__":
    main()
