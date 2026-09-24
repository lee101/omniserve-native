import re,sys,collections
since=sys.argv[1] if len(sys.argv)>1 else '2026-09-24T00'
rx=re.compile(r'^(\S+) .*?model=(\S+) status=(\d+) dur_ms=([\d.]+).*path="([^"?]+)')
hour=collections.Counter(); paths=collections.Counter(); err=collections.Counter(); dur=collections.defaultdict(list)
for l in open('/var/log/omniserve/access.log',errors='replace'):
    m=rx.match(l)
    if not m or m[1]<since: continue
    ts,model,st,d,p=m.groups()
    if p in('/status','/metrics','/readyz','/healthz','/v1/vram/status','/backend_status'): continue
    k=p if not p.startswith('/v1/jobs') else '/v1/jobs/*'
    paths[k]+=1; dur[k].append(float(d)); hour[(ts[:13],k)]+=1
    if st[0]=='5': err[(ts[:15],k,st)]+=1
for k,v in paths.most_common(25):
    s=sorted(dur[k]); print(f'{v:7d} {k:45s} p50={s[len(s)//2]:.0f}ms p95={s[int(len(s)*.95)]:.0f}ms')
print('5xx by 10min:')
for k,v in sorted(err.items()): print(' ',k,v)
if '-h' in sys.argv:
  for k,v in sorted(hour.items()): print(k,v)
