import sys,collections,os
f=sys.argv[1]
def unit(pid):
    try:
        for l in open(f'/proc/{pid}/cgroup'):
            u=l.strip().rsplit('/',1)[-1]
            if u: return u
    except Exception: pass
    return f'pid{pid}'
per=collections.defaultdict(list); tot=[]; ts=set()
for l in open(f):
    p=l.strip().split(',')
    if len(p)<3: continue
    if p[1]=='TOTAL': tot.append((int(p[0]),int(p[2]),int(p[3]) if len(p)>3 else 0)); continue
    try: per[p[1]].append((int(p[0]),int(p[2])))
    except ValueError: pass
n=len(tot); print(f'samples={n} span={(tot[-1][0]-tot[0][0])/60:.0f}min')
u=sorted(x[1] for x in tot); print(f'TOTAL used MiB p50={u[n//2]} p95={u[int(n*.95)]} max={u[-1]}  gpu_util_mean={sum(x[2] for x in tot)/n:.0f}%')
agg=collections.defaultdict(list)
for pid,v in per.items(): agg[unit(pid)]+= [m for _,m in v]
for k,v in sorted(agg.items(),key=lambda kv:-max(kv[1])):
    s=sorted(v); print(f'{k:45s} n={len(s):6d} min={s[0]:6d} p50={s[len(s)//2]:6d} p95={s[int(len(s)*.95)]:6d} max={s[-1]:6d}')
