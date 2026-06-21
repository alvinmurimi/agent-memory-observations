"""E1b - de-rigged multi-hop recall, answering the adversarial critic.

Tests three things the critic flagged:
 (1) Is the single-shot multi-hop recall 'failure' intrinsic, or a token-collision artifact of a small
     attribute (predicate) vocabulary? -> sweep the attribute-pool size.
 (2) Honest retrieval-compute accounting: report ranking passes and docs scored per signal, separately
     from the reader budget k.
 (3) A matched-reader-context baseline: single-shot at large k (sem_k60, reads 60 docs) vs the swim at
     small k (swim_k10, reads 10 docs). Wilson 95% CIs throughout. LLM-free.
"""
from __future__ import annotations
import argparse, json, random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

SYL = ("tar von mal quen bryn sed wol kor pyx zel dree fenn gorm hask ilv jarr lome nyx orr pell rund "
       "sval threx ulm vask wend yarn zoth brae cind dorn esk fyl grau hev ish kael lorn myr oss prmagg "
       "quoll rhad skel tius uvarn vohl wyrm xil yaric zonn").split()
TAILS = ["Compact","Authority","Trust","Bureau","Consortium","Office","Assembly","Charter","Foundation",
         "Council","Holdings","Partners","Works","Group","Systems","Union"]
ROLE_A = ["operating","designated","controlling","assigned","lead","governing","oversight","primary",
          "appointed","registered","supervising","managing","sponsoring","auditing","custodial","executive"]
ROLE_B = ["sponsor","custodian","office","steward","underwriter","trustee","authority","guarantor",
          "liquidator","proprietor","registrar","controller","administrator","examiner","overseer","agent"]
ALL_ATTRS = [f"{a} {b}" for a in ROLE_A for b in ROLE_B]   # 256 distinct attribute phrases

def W(rng, n=2): return "".join(rng.choice(SYL) for _ in range(n)).capitalize()
def person(rng): return f"{W(rng)} {W(rng)}"
def org(rng): return ("the " if rng.random() < .5 else "") + f"{W(rng)} {rng.choice(TAILS)}"

def emb_fit(corpus):
    docs = [c for c in dict.fromkeys(corpus) if c.strip()]
    return TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=1).fit(docs)

def rank(vec, query, notes, cnt):
    cnt["passes"] += 1; cnt["docs"] += len(notes)
    M = normalize(vec.transform(notes)); q = normalize(vec.transform([query]))
    s = (M @ q.T).toarray().ravel()
    return sorted(range(len(notes)), key=lambda i: -float(s[i]))

def case_multihop(rng, N, attrs):
    A, X, C = person(rng), org(rng), org(rng)
    ba, a = rng.choice(attrs), rng.choice(attrs)
    core = [f"The {ba} of {A} is {X}.", f"The {a} of {X} is {C}."]
    dist = [f"The {rng.choice(attrs)} of {person(rng)} is {org(rng)}." for _ in range(N)]
    return dict(notes=core + dist, question=f"What is the {a} of the {ba} of {A}?", decisive={0, 1})

def sem(vec, c, k, cnt):
    return set(rank(vec, c["question"], c["notes"], cnt)[:k])

def swim(vec, c, k, cnt):
    seed = rank(vec, c["question"], c["notes"], cnt)[: max(2, k // 2)]
    extra = []
    for i in seed:
        extra += [j for j in rank(vec, c["notes"][i], c["notes"], cnt)[: max(2, k // 2)] if j != i]
    return set(list(dict.fromkeys(list(seed) + extra))[:k])

def wilson(k, n):
    if not n: return (0.0, 0.0)
    z = 1.96; p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (round(c - h, 3), round(c + h, 3))

def run(pools, N, per, seed, ksm, kbg):
    rows = []
    for P in pools:
        attrs = ALL_ATTRS[:P]
        cases = [case_multihop(random.Random(f"{seed}:{P}:{i}"), N, attrs) for i in range(per)]
        vecs = [emb_fit(c["notes"] + [c["question"]]) for c in cases]
        out = {}
        for name, fn in [(f"sem_k{ksm}", lambda v, c, cnt: sem(v, c, ksm, cnt)),
                         (f"sem_k{kbg}", lambda v, c, cnt: sem(v, c, kbg, cnt)),
                         (f"swim_k{ksm}", lambda v, c, cnt: swim(v, c, ksm, cnt))]:
            rec, cnt = 0, {"passes": 0, "docs": 0}
            for c, v in zip(cases, vecs):
                if c["decisive"] <= fn(v, c, cnt):
                    rec += 1
            out[name] = dict(recall=round(rec / per, 3), ci=wilson(rec, per),
                             passes=round(cnt["passes"] / per, 1), docs=round(cnt["docs"] / per, 1))
        rows.append(dict(attr_pool=P, N=N, n=per, signals=out))
        print(f"attr_pool={P:3d} N={N} per={per} | " +
              "  ".join(f"{k}={out[k]['recall']:.2f}{out[k]['ci']}|{out[k]['passes']}p/{int(out[k]['docs'])}d"
                        for k in out), flush=True)
    return rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pools", default="8,16,32,64,128,256")
    p.add_argument("--N", type=int, default=500)
    p.add_argument("--per", type=int, default=60)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--ksm", type=int, default=10)
    p.add_argument("--kbg", type=int, default=60)
    p.add_argument("--out", default="results/e1b_multihop_fair.json")
    a = p.parse_args()
    rows = run([int(x) for x in a.pools.split(",")], a.N, a.per, a.seed, a.ksm, a.kbg)
    from pathlib import Path
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"config": vars(a), "rows": rows}, indent=2), encoding="utf-8")
    print("\nWROTE", a.out)

if __name__ == "__main__":
    main()
