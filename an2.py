import sys
from collections import Counter
from an import cls, load, report
def seg(f):
    ev=load(f)
    k=[e for e in ev if e.get("ph")=="X" and e.get("cat") in ("kernel","gpu_memcpy","gpu_memset")]
    k.sort(key=lambda e:e["ts"])
    print("file:",f); print("total GPU kernel events:",len(k))
    nm=[e["name"] for e in k]
    cand=Counter()
    for n in nm:
        l=n.lower()
        if "gumbel" in l or "argmax" in l or "sampled" in l: cand[n]+=1
    if not cand: print("NO SAMPLER MARKER"); return
    mk,c=cand.most_common(1)[0]
    idx=[i for i,n in enumerate(nm) if n==mk]
    print("marker: %s"%mk[:70]); print("marker count: %d"%c)
    if len(idx)<3: print("TOO FEW MARKERS"); return
    sz=[idx[j+1]-idx[j] for j in range(len(idx)-1)]
    print("last 12 segment sizes:",sz[-12:])
    a,b=idx[-2],idx[-1]
    w=[(k[i]["name"],k[i].get("dur",0)) for i in range(a+1,b+1)]
    tot=sum(d for _,d in w)
    print("STEADY DECODE STEP: %d launches, %.1f us GPU"%(len(w),tot))
    return w,tot
for f in sys.argv[1:]:
    r=seg(f)
    if r: report(*r)
    print("="*70)
