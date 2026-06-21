"""E1d - the perfect-recall boundary (LLM-free). When does single-shot retrieval find the decisive
note, and when does it fail? We vary exactly which query tokens the decisive note shares:
  BOTH (entity+attribute), ENTITY only, ATTR only, NEITHER (pronoun, document-adjacent to an entity
  intro chunk). For NEITHER we also test neighbor-expansion recovery (retrieve the entity-mentioning
  chunk, add its document neighbor). Big predicate vocabulary (no collision artifact), Wilson CIs.
"""
from __future__ import annotations
import argparse, json, random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

SYL=("tar von mal quen bryn sed wol kor pyx zel dree fenn gorm hask ilv jarr lome nyx orr pell rund sval "
     "threx ulm vask wend yarn zoth brae cind dorn esk fyl grau hev ish kael lorn myr oss prmagg quoll rhad "
     "skel tius uvarn vohl wyrm xil yaric zonn").split()
TAILS=["Compact","Authority","Trust","Bureau","Consortium","Office","Assembly","Charter","Foundation","Council","Holdings","Partners","Works","Group","Systems","Union"]
RA=["operating","designated","controlling","assigned","lead","governing","oversight","primary","appointed","registered","supervising","managing","sponsoring","auditing","custodial","executive"]
RB=["sponsor","custodian","office","steward","underwriter","trustee","authority","guarantor","liquidator","proprietor","registrar","controller","administrator","examiner","overseer","agent"]
ATTRS=[f"{a} {b}" for a in RA for b in RB]
W=lambda r,n=2:"".join(r.choice(SYL) for _ in range(n)).capitalize()
person=lambda r:f"{W(r)} {W(r)}"
org=lambda r:("the " if r.random()<.5 else "")+f"{W(r)} {r.choice(TAILS)}"

def emb_fit(corpus):
    docs=[c for c in dict.fromkeys(corpus) if c.strip()]
    return TfidfVectorizer(lowercase=True,stop_words="english",ngram_range=(1,2),min_df=1).fit(docs)
def rank(vec,q,notes):
    M=normalize(vec.transform(notes)); qq=normalize(vec.transform([q]))
    s=(M@qq.T).toarray().ravel()
    return sorted(range(len(notes)),key=lambda i:-float(s[i]))

def case(rng,N,cond):
    E,a,v=person(rng),rng.choice(ATTRS),org(rng)
    intro=f"{E} is a registered entity in good standing."
    if cond=="BOTH":    dec=f"The {a} of {E} is {v}."
    elif cond=="ENTITY":dec=f"Regarding {E}, the responsible party on record is {v}."
    elif cond=="ATTR":  dec=f"The {a} on record is {v}."
    else:               dec=f"The responsible party on record is {v}."   # NEITHER: pronoun-like, no E/a
    # document: [intro, dec] adjacent at a random position among N distractors
    dist=[f"The {rng.choice(ATTRS)} of {person(rng)} is {org(rng)}." for _ in range(N)]
    pos=rng.randrange(0,N+1)
    notes=dist[:pos]+[intro,dec]+dist[pos:]
    dec_idx=pos+1; intro_idx=pos
    return dict(notes=notes,question=f"What is the {a} of {E}?",dec_idx=dec_idx,intro_idx=intro_idx,entity=E)

def sem_recall(vec,c,k):
    return c["dec_idx"] in set(rank(vec,c["question"],c["notes"])[:k])
def neighbor_recall(vec,c,k):
    # retrieve top-k by query; for any retrieved chunk that mentions the entity, add its doc-neighbors
    top=rank(vec,c["question"],c["notes"])[:k]
    got=set(top); ent=c["entity"].lower()
    for i in top:
        if ent in c["notes"][i].lower():
            if i+1<len(c["notes"]): got.add(i+1)
            if i-1>=0: got.add(i-1)
    return c["dec_idx"] in got
def wilson(k,n):
    if not n:return(0,0)
    z=1.96;p=k/n;d=1+z*z/n;cc=(p+z*z/(2*n))/d;h=z*((p*(1-p)/n+z*z/(4*n*n))**.5)/d
    return(round(cc-h,3),round(cc+h,3))

def run(conds,N,per,k,seed):
    rows=[]
    for cond in conds:
        cs=[case(random.Random(f"{seed}:{cond}:{i}"),N,cond) for i in range(per)]
        vs=[emb_fit(c["notes"]+[c["question"]]) for c in cs]
        sem=sum(sem_recall(v,c,k) for c,v in zip(cs,vs))
        nb=sum(neighbor_recall(v,c,k) for c,v in zip(cs,vs))
        rows.append(dict(cond=cond,N=N,k=k,n=per,sem_recall=round(sem/per,3),sem_ci=wilson(sem,per),
                         neighbor_recall=round(nb/per,3),neighbor_ci=wilson(nb,per)))
        print(f"{cond:7s} N={N} k={k} | sem={sem/per:.2f}{wilson(sem,per)}  neighbor={nb/per:.2f}{wilson(nb,per)}",flush=True)
    return rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--conds",default="BOTH,ENTITY,ATTR,NEITHER")
    p.add_argument("--N",type=int,default=300); p.add_argument("--per",type=int,default=60)
    p.add_argument("--k",type=int,default=10); p.add_argument("--seed",type=int,default=7)
    p.add_argument("--out",default="results/e1d_recall_boundary.json")
    a=p.parse_args()
    rows=run(a.conds.split(","),a.N,a.per,a.k,a.seed)
    from pathlib import Path
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps({"config":vars(a),"rows":rows},indent=2),encoding="utf-8")
    print("\nWROTE",a.out)

if __name__=="__main__":
    main()
