import gzip,json,sys,re
from collections import Counter
C=[("act_quant",r"scale_1x128|scale_128x1|scale_1x64|quant|scaled_fp8|per_token|fp8_group|e8m0"),
("attn",r"flash|fmha|attn|paged|merge_state|varlen"),
("gemm",r"cutlass|gemm|nvjet|cublas|sm90|tma_|matmul|Kernel_"),
("norm",r"rms_?norm|layer_?norm"),
("rope",r"rotary|rope"),
("act",r"silu|gelu|swiglu|act_and_mul"),
("moe",r"moe|expert"),
("comm",r"nccl|allreduce|all_reduce"),
("sample",r"topk|top_k|softmax|argmax|multinomial|curand|sort|cumsum"),
("mem",r"memcpy|memset|Memcpy|Memset"),
("elem",r"elementwise|vectorized|unrolled|index|embed|gather|copy|fill|reduce|cast|cat")]
CC=[(n,re.compile(p,re.I)) for n,p in C]
def cls(n):
    for k,r in CC:
        if r.search(n): return k
    return "other"
def load(p):
    o=gzip.open if p.endswith(".gz") else open
    d=json.load(o(p,"rt"))
    return d["traceEvents"] if isinstance(d,dict) else d
def period(nm,mn=5):
    n=len(nm)
    if n<2*mn: return None,0
    lo,hi=n//4,3*n//4
    s=nm[lo:hi]; m=len(s); b,bs=None,0.0
    for p in range(mn,m//2+1):
        c=m-p
        if c<p: break
        k=sum(1 for i in range(c) if s[i]==s[i+p])/c
        if k>0.98: return p,k
        if k>bs: b,bs=p,k
    return b,bs
def main(f):
    ev=load(f)
    k=[e for e in ev if e.get("ph")=="X" and e.get("cat") in ("kernel","gpu_memcpy","gpu_memset")]
    k.sort(key=lambda e:e["ts"])
    print("file:",f); print("total GPU kernel events:",len(k))
    if not k: print("NO KERNELS"); return
    nm=[e["name"] for e in k]
    p,sc=period(nm)
    print("period (kernels/decode step): %s  selfsim=%.3f"%(p,sc))
    if not p: return
    lo=len(nm)//4
    w=[(k[i]["name"],k[i].get("dur",0)) for i in range(lo,lo+p)]
    tot=sum(d for _,d in w)
    print("one step: %d launches, %.1f us GPU"%(len(w),tot))
    return w,tot
def report(w,tot):
    cn=Counter(); cd=Counter()
    for n,d in w:
        c=cls(n); cn[c]+=1; cd[c]+=d
    print("\n%-11s%9s%10s%8s"%("category","launch","us","%time"))
    for c in sorted(cn,key=lambda x:-cd[x]):
        print("%-11s%9d%10.1f%7.1f%%"%(c,cn[c],cd[c],100*cd[c]/tot if tot else 0))
    print("%-11s%9d%10.1f%7.1f%%"%("TOTAL",sum(cn.values()),tot,100.0))
    aq=[(n,d) for n,d in w if cls(n)=="act_quant"]
    print("\n== ACT-QUANT KERNELS IN ONE STEP ==")
    a=Counter(); ad=Counter()
    for n,d in aq: a[n]+=1; ad[n]+=d
    for n,c in a.most_common(): print("  %3dx %8.1fus  %s"%(c,ad[n],n[:95]))
    print("  -> %d/%d launches, %.1fus (%.1f%% of step)"%(len(aq),len(w),sum(d for _,d in aq),100*sum(d for _,d in aq)/tot if tot else 0))
    print("\n== ALL DISTINCT KERNELS IN ONE STEP ==")
    pc=Counter(); pd=Counter()
    for n,d in w: pc[n]+=1; pd[n]+=d
    print("%4s %9s  %-10s %s"%("n","us","cat","name"))
    for n,c in pc.most_common():
        print("%4d %9.1f  %-10s %s"%(c,pd[n],cls(n),n[:88]))
if __name__=="__main__":
    for f in sys.argv[1:]:
        r=main(f)
        if r: report(*r)
        print("\n"+"="*72+"\n")
