"""Summarize GPU clocks/throttle reasons per Phase 2 run window (from logs/p2_clocks.csv + logs/p2_driver.log).
Only samples with power > 150 W (GPU busy) are counted. Output: raw/p2_clock_windows.json + printed table."""
import csv, json, re, datetime as dt, collections

ROOT = "/workspace/fusion-study"
rows = []
for r in csv.reader(open(f"{ROOT}/logs/p2_clocks.csv")):
    if r[0].startswith("timestamp"):
        continue
    try:
        t = dt.datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
        rows.append((t, int(r[1].split()[0]), float(r[3].split()[0]), int(r[4]), r[5].strip()))
    except (ValueError, IndexError):
        pass
starts = []
for line in open(f"{ROOT}/logs/p2_driver.log"):
    m = re.match(r"(\d\d:\d\d:\d\d) (\S+) ", line)
    if m:
        starts.append((m.group(1), m.group(2)))
day = rows[0][0].date() if rows else dt.date.today()
out = {}
for i, (ts, name) in enumerate(starts):
    t0 = dt.datetime.combine(day, dt.time.fromisoformat(ts))
    t1 = dt.datetime.combine(day, dt.time.fromisoformat(starts[i + 1][0])) if i + 1 < len(starts) else rows[-1][0]
    busy = [r for r in rows if t0 <= r[0] < t1 and r[2] > 150]
    if not busy:
        continue
    reasons = collections.Counter(r[4] for r in busy)
    out[name] = dict(n=len(busy), sm_min=min(r[1] for r in busy), sm_mean=sum(r[1] for r in busy) / len(busy),
                     temp_max=max(r[3] for r in busy), reasons=dict(reasons))
    print(f"{name:14s} n={len(busy):4d} sm_mean={out[name]['sm_mean']:6.0f} sm_min={out[name]['sm_min']:5d} "
          f"tmax={out[name]['temp_max']} {dict(reasons)}")
json.dump(out, open(f"{ROOT}/raw/p2_clock_windows.json", "w"), indent=1)
