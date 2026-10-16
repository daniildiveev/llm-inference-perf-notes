import gzip, json, sys, collections
t = json.load(gzip.open(sys.argv[1]))
ev = t["traceEvents"]
cats = collections.Counter(e.get("cat") for e in ev)
print(cats.most_common())
names = collections.Counter(e["name"] for e in ev if e.get("cat") in ("user_annotation", "gpu_user_annotation"))
print(names.most_common(30))
k = collections.defaultdict(lambda: [0, 0.0])
for e in ev:
    if e.get("cat") == "kernel":
        k[e["name"]][0] += 1; k[e["name"]][1] += e["dur"]
for n, (c, d) in sorted(k.items(), key=lambda x: -x[1][1])[:40]:
    print(f"{d:10.1f} {c:6d} {n[:150]}")
