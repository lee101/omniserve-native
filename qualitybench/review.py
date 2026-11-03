#!/usr/bin/env python3
import collections, json, os, sys
from PIL import Image

out = sys.argv[1]
prompts = json.load(open(os.environ.get("PROMPTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts_v2.json"))))
rows = [json.loads(l) for l in open(f"{out}/results.jsonl")]
os.makedirs(f"{out}/thumbs", exist_ok=True)
t = collections.defaultdict(list)
cells = collections.defaultdict(dict)
for r in rows:
    t[r["tag"]].append(r["inf_ms"] or r["wall"] * 1000)
    th = f"thumbs/{r['tag']}__{r['prompt']}_s{r['seed']}.jpg"
    if not os.path.exists(f"{out}/{th}"):
        Image.open(f"{out}/{r['tag']}/{r['prompt']}_s{r['seed']}.{r['ext']}").convert("RGB").resize((640, 640), Image.LANCZOS).save(f"{out}/{th}", quality=85)
    cells[(r["prompt"], r["seed"])][r["tag"]] = th
secs = {k: sum(v) / len(v) / 1000 for k, v in t.items()}
tags = sorted(secs, key=lambda k: secs[k])
data = dict(tags=tags, secs=secs, prompts=prompts, cells={f"{p}|{s}": c for (p, s), c in cells.items()})
html = """<!doctype html><meta charset=utf-8><title>review</title><style>
body{font:14px system-ui;background:#111;color:#ddd;margin:0;padding:12px}
.row{margin:0 0 28px}.p{color:#9ab;margin:6px 0}.g{display:grid;grid-template-columns:repeat(auto-fill,minmax(var(--w,420px),1fr));gap:6px}
.c{position:relative;background:#222}.c img{width:100%;display:block;cursor:zoom-in}.l{position:absolute;left:0;top:0;background:#000b;padding:2px 6px;font-size:12px}
.r{position:absolute;right:0;bottom:0;background:#000b}.r b{cursor:pointer;padding:2px 6px;color:#888}.r b.on{color:#ff0;background:#442}
#bar{position:sticky;top:0;background:#111e;padding:6px 0;z-index:9}button,select{background:#333;color:#ddd;border:1px solid #555;padding:3px 8px}
#big{position:fixed;inset:0;background:#000e;display:none;z-index:99;overflow:auto}#big img{max-width:none;width:1024px}
</style><div id=bar>cols <input id=w type=range min=200 max=900 value=420> <button id=blind>blind: off</button> <button id=exp>export ratings</button> <button id=sum>summary</button> <select id=cat></select> <span id=msg></span></div><div id=root></div><div id=big onclick="this.style.display='none'"><img></div><pre id=out></pre>
<script>const D=__DATA__;const K='rev_'+location.pathname;let R=JSON.parse(localStorage.getItem(K)||'{}'),blind=false;
const root=document.getElementById('root');
function shuffle(a,s){a=[...a];let x=s;for(let i=a.length-1;i>0;i--){x=(x*1103515245+12345)&0x7fffffff;const j=x%(i+1);[a[i],a[j]]=[a[j],a[i]]}return a}
function draw(){root.innerHTML='';const cat=document.getElementById('cat').value;
for(const k of Object.keys(D.cells)){const [p,s]=k.split('|');if(cat&&!p.startsWith(cat))continue;const c=D.cells[k];
const div=document.createElement('div');div.className='row';div.innerHTML=`<div class=p>${p} s${s}: ${D.prompts[p]}</div><div class=g></div>`;const g=div.lastChild;
let tags=D.tags.filter(t=>c[t]);if(blind)tags=shuffle(tags,[...k].reduce((a,ch)=>a*31+ch.charCodeAt(0),7)&0x7fffffff);
tags.forEach((t,i)=>{const e=document.createElement('div');e.className='c';const rk=k+'|'+t;
e.innerHTML=`<img src="${c[t]}" loading=lazy><div class=l>${blind?'#'+(i+1):t+' '+D.secs[t].toFixed(1)+'s'}</div><div class=r>${[1,2,3,4,5].map(n=>`<b class="${R[rk]==n?'on':''}" data-n=${n}>${n}</b>`).join('')}</div>`;
e.querySelector('img').onclick=()=>{const b=document.getElementById('big');b.querySelector('img').src=c[t].replace('thumbs/','').replace(/__.*/,'')&&c[t];b.style.display='block'};
e.querySelectorAll('b').forEach(b=>b.onclick=()=>{R[rk]=R[rk]==b.dataset.n?0:+b.dataset.n;localStorage.setItem(K,JSON.stringify(R));draw2(e,rk)});g.appendChild(e)});root.appendChild(div)}}
function draw2(e,rk){e.querySelectorAll('b').forEach(b=>b.className=R[rk]==b.dataset.n?'on':'')}
const cats=['',...new Set(Object.keys(D.prompts).map(p=>p.split('_')[0]))];document.getElementById('cat').innerHTML=cats.map(c=>`<option value="${c}">${c||'all'}</option>`).join('');
document.getElementById('cat').onchange=draw;document.getElementById('w').oninput=e=>{document.body.style.setProperty('--w',e.target.value+'px')};
document.getElementById('blind').onclick=e=>{blind=!blind;e.target.textContent='blind: '+(blind?'on':'off');draw()};
function summary(){const a={};for(const [rk,v] of Object.entries(R)){if(!v)continue;const t=rk.split('|').pop();(a[t]=a[t]||[]).push(v)}
return Object.entries(a).map(([t,v])=>({tag:t,n:v.length,mean:+(v.reduce((x,y)=>x+y,0)/v.length).toFixed(2),secs:+D.secs[t].toFixed(1)})).sort((x,y)=>y.mean-x.mean)}
document.getElementById('sum').onclick=()=>{document.getElementById('out').textContent=summary().map(r=>`${r.tag.padEnd(22)} n=${r.n} mean=${r.mean} ${r.secs}s`).join('\\n')};
document.getElementById('exp').onclick=()=>{const b=new Blob([JSON.stringify({ratings:R,summary:summary()},null,1)]);const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='ratings.json';a.click()};
draw()</script>""".replace("__DATA__", json.dumps(data))
open(f"{out}/review.html", "w").write(html)
print(f"{out}/review.html", len(cells), "rows", len(tags), "tags")
for k in tags:
    print(f"{k:22s} {secs[k]:5.1f}s n={len(t[k])}")
