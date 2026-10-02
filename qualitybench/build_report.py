#!/usr/bin/env python3
import base64, html, io, json, os, sys
from PIL import Image

out = sys.argv[1]
embed = len(sys.argv) > 2 and sys.argv[2] == "embed"
dest = sys.argv[3] if len(sys.argv) > 3 else f"{out}/report"
prompts = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep_prompts.json")))
summary = json.load(open(f"{out}/summary.json"))
per = json.load(open(f"{out}/per_image.json"))
os.makedirs(f"{dest}/thumbs", exist_ok=True)
TH = 448


def thumb(tag, pk):
    p = f"{out}/{tag}/{pk}.png"
    if not os.path.exists(p):
        return None
    tp = f"{dest}/thumbs/{tag}__{pk}.webp"
    if not os.path.exists(tp):
        im = Image.open(p).convert("RGB")
        im.thumbnail((TH, TH))
        im.save(tp, "WEBP", quality=82)
    if embed:
        return "data:image/webp;base64," + base64.b64encode(open(tp, "rb").read()).decode()
    return f"thumbs/{tag}__{pk}.webp"


def full(tag, pk):
    return os.path.relpath(f"{out}/{tag}/{pk}.png", dest)


tags = [s["tag"] for s in summary]
S = {s["tag"]: s for s in summary}


def pareto(items, x, y):
    keep = []
    for a in items:
        if a[x] is None or a[y] is None:
            continue
        if not any(b is not a and b[x] is not None and b[y] is not None and b[x] <= a[x] and b[y] <= a[y] and (b[x] < a[x] or b[y] < a[y]) for b in items):
            keep.append(a["tag"])
    return keep


basefam = [s for s in summary if s["lpips"] is not None and s["extra"].get("turbo") is False]
front = set(pareto(basefam, "wall", "lpips"))


def fmt(v, f):
    return "-" if v is None else f % v


rows = []
for s in summary:
    cls = "front" if s["tag"] in front else ""
    rows.append(
        f"<tr class='{cls}' data-tag='{s['tag']}'><td>{html.escape(s['tag'])}</td><td>{html.escape(s['server'])}</td>"
        f"<td>{s['steps']}</td><td>{s['wall']:.1f}</td><td>{fmt(s['lpips'], '%.3f')}</td><td>{fmt(s['psnr'], '%.1f')}</td>"
        f"<td>{s['clip']:.3f}</td><td>{fmt(s['clip_delta'], '%+.4f')}</td><td>{fmt(s['sharp_ratio'], '%.2f')}</td>"
        f"<td>{s['nyq']:.1e}</td><td class='ex'>{html.escape(json.dumps(s['extra']))}</td></tr>")

galleries = []
for pk, text in prompts.items():
    cells = []
    for s in summary:
        src = thumb(s["tag"], pk)
        if not src:
            continue
        pi = next((p for p in per if p["tag"] == s["tag"] and p["prompt"] == pk), None)
        cap = f"{s['tag']}<br><small>{pi['wall']:.1f}s" + (f" lpips {pi['lpips']:.3f}" if pi and pi["lpips"] is not None else "") + "</small>" if pi else s["tag"]
        link = "#" if embed else full(s["tag"], pk)
        cells.append(f"<figure><a href='{link}' target='_blank'><img loading='lazy' src='{src}'></a><figcaption>{cap}</figcaption></figure>")
    galleries.append(f"<section><h3>{pk}</h3><p class='pr'>{html.escape(text)}</p><div class='grid'>{''.join(cells)}</div></section>")

doc = f"""<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>RA2 Quality Sweep</title>
<style>
:root{{--bg:#fff;--fg:#1a1a1a;--mut:#666;--line:#ddd;--hi:#e8f4e8}}
@media(prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#111;--fg:#eee;--mut:#999;--line:#333;--hi:#1c2b1c}}}}
:root[data-theme=dark]{{--bg:#111;--fg:#eee;--mut:#999;--line:#333;--hi:#1c2b1c}}
body{{background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif;margin:0 auto;max-width:1400px;padding:16px}}
table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{border-bottom:1px solid var(--line);padding:3px 6px;text-align:right;white-space:nowrap}}
td:first-child,td:nth-child(2),th:first-child,th:nth-child(2),.ex{{text-align:left}}.ex{{color:var(--mut);max-width:340px;overflow:hidden;text-overflow:ellipsis}}
tr.front{{background:var(--hi)}}th{{cursor:pointer;position:sticky;top:0;background:var(--bg)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:8px}}figure{{margin:0}}img{{width:100%;display:block;border:1px solid var(--line)}}
figcaption{{font-size:11px}}.pr{{color:var(--mut);font-size:12px}}.wrap{{overflow-x:auto}}small{{color:var(--mut)}}
</style></head><body>
<h1>RA2 quality sweep</h1>
<p>Qwen Image 2.1 Q4_K_M, 1024x1024, seed 0, RTX 3090 Ti (shared), {len(prompts)} prompts, VAE tiling on, notch on unless stated. LPIPS/PSNR are against <b>base30_dense</b> (30 steps, no cache). Green rows are the time-vs-LPIPS Pareto front of the base-model family. LPIPS is meaningless for turbo and premerged-turbo rows (different distilled style). CLIP is ViT-B/32 laion2b prompt similarity (coarse). Sharp = Laplacian variance relative to the reference. Click a header to sort.</p>
<div class=wrap><table id=t><thead><tr><th>tag</th><th>server</th><th>steps</th><th>s/img</th><th>LPIPS</th><th>PSNR</th><th>CLIP</th><th>dCLIP</th><th>sharp</th><th>nyquist</th><th>request</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
{''.join(galleries)}
<script>
const t=document.getElementById('t');t.tHead.onclick=e=>{{const i=e.target.cellIndex;if(i==null)return;const b=t.tBodies[0];const r=[...b.rows];const d=e.target.dir=-(e.target.dir||-1);
r.sort((a,c)=>{{const x=a.cells[i].textContent,y=c.cells[i].textContent;const nx=parseFloat(x),ny=parseFloat(y);return d*((isNaN(nx)||isNaN(ny))?x.localeCompare(y):nx-ny)}});r.forEach(x=>b.appendChild(x))}}
</script></body></html>"""
open(f"{dest}/index.html", "w").write(doc)
print(f"wrote {dest}/index.html ({os.path.getsize(f'{dest}/index.html')//1024} KB)")
