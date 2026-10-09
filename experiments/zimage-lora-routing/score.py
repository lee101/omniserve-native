import json,os,sys,statistics as st
from pathlib import Path
os.environ['AESTHETIC_DEVICE']='cpu';os.environ.setdefault('CUDA_VISIBLE_DEVICES','')
sys.path.insert(0,'/nvme0n1-disk/code/cutedsl-site/inference')
from aesthetic_score import get_scorer
from PIL import Image
out=Path(sys.argv[1]);cache=out/'scores.json'
S=json.load(open(cache)) if cache.exists() else {}
P={json.loads(l)['id']:json.loads(l) for l in open('prompts.jsonl')}
sc=get_scorer();T={}
for i,p in P.items(): T[i]=sc.embed_texts([p['prompt']])[0]
E={}
for vd in sorted(x for x in out.iterdir() if x.is_dir()):
    for f in sorted(vd.glob('*.png')):
        k=f'{vd.name}/{f.stem}'
        if k in S and 'emb' not in S[k]: continue
        im=Image.open(f).convert('RGB');emb,aes=sc.embed_and_score([im])
        S[k]={'clip':float(emb[0]@T[f.stem]),'aes':aes[0]};E[k]=emb[0]
        import torch;torch.save(emb[0],out/f'.{vd.name}.{f.stem}.pt')
for k in list(S):
    v,i=k.split('/');q=out/f'.qwen21.{i}.pt';me=out/f'.{v}.{i}.pt'
    if q.exists() and me.exists():
        import torch;S[k]['cos_q']=float(torch.load(me)@torch.load(q))
json.dump(S,open(cache,'w'))
rows={}
for k,s in S.items():
    v,i=k.split('/');rows.setdefault(v,{})[i]=s
cats=sorted({p['category'] for p in P.values()})
def agg(v,key,ids):
    xs=[rows[v][i][key] for i in ids if i in rows[v] and key in rows[v][i]];return (st.mean(xs),len(xs)) if xs else (None,0)
print(f"{'variant':14s} {'n':>3s} {'clip':>6s} {'aes':>6s} {'cosQ':>6s} {'dclip':>6s} {'daes':>6s} {'winA':>5s} | per-cat d_aes/d_clip vs plain")
for v in sorted(rows):
    ids=list(rows[v]);n=len(ids)
    c=agg(v,'clip',ids)[0];a=agg(v,'aes',ids)[0];cq=agg(v,'cos_q',ids)[0]
    pl=[i for i in ids if i in rows.get('plain',{})]
    dc=st.mean(rows[v][i]['clip']-rows['plain'][i]['clip'] for i in pl) if pl else 0
    da=st.mean(rows[v][i]['aes']-rows['plain'][i]['aes'] for i in pl) if pl else 0
    wa=sum(rows[v][i]['aes']>rows['plain'][i]['aes'] for i in pl)/len(pl) if pl else 0
    per=[]
    for ct in cats:
        cid=[i for i in pl if P[i]['category']==ct]
        if cid: per.append(f"{ct[:4]} {st.mean(rows[v][i]['aes']-rows['plain'][i]['aes'] for i in cid):+.2f}/{st.mean(rows[v][i]['clip']-rows['plain'][i]['clip'] for i in cid):+.3f}")
    f=lambda x:'  -   ' if x is None else f'{x:6.3f}'
    print(f"{v:14s} {n:3d} {f(c)} {f(a)} {f(cq)} {dc:+.3f} {da:+.3f} {wa:5.2f} | {' '.join(per)}")
