import re,sys,datetime as D
since=sys.argv[1]; pat=re.compile(sys.argv[2])
ts=[]
for l in open('/var/log/omniserve/access.log',errors='replace'):
    if l[:13]<since or not pat.search(l): continue
    ts.append(D.datetime.fromisoformat(l[:23]).timestamp())
ts.sort(); g=sorted(b-a for a,b in zip(ts,ts[1:]))
span=(ts[-1]-ts[0])/3600 if ts else 0
if not g: print('none'); sys.exit()
idle=lambda th: sum(x for x in g if x>th)/3600
print(f'n={len(ts)} span={span:.1f}h gap p50={g[len(g)//2]:.0f}s p90={g[int(len(g)*.9)]:.0f}s max={g[-1]:.0f}s; hours idle in gaps>120s={idle(120):.1f} >600s={idle(600):.1f}')
