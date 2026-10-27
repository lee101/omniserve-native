"""Self-contained HTML grid (case x config) with judge scores: out/report.html.

uv run --with pillow python qualitybench/edit/report.py [--model gemini-3.1-pro-preview] [--only cfg,..]
"""
import argparse, base64, io, json, os
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))


def thumb(path, edge=384):
    im = Image.open(path).convert("RGB")
    im.thumbnail((edge, edge))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(HERE, "out"))
ap.add_argument("--model", default="gemini-3.1-pro-preview")
ap.add_argument("--only", default="")
a = ap.parse_args()
cases = json.load(open(os.path.join(HERE, "cases.json")))
configs = a.only.split(",") if a.only else sorted(d for d in os.listdir(a.out) if os.path.isdir(os.path.join(a.out, d)))
scores = {}
for line in open(os.path.join(a.out, "judge.jsonl")):
    r = json.loads(line)
    if r.get("model") == a.model and "error" not in r:
        scores[(r["config"], r["case"])] = r
walls = {}
for line in open(os.path.join(a.out, "results.jsonl")):
    r = json.loads(line)
    walls[(r["config"], r["case"])] = r["wall_s"]
html = ["<!doctype html><meta charset=utf-8><title>Edit eval</title><style>body{font:12px system-ui;background:#111;color:#ddd}"
        "table{border-collapse:collapse}td{vertical-align:top;padding:4px}img{width:256px;display:block}"
        "th{position:sticky;top:0;background:#222;padding:4px}.n{color:#999;max-width:256px}</style><table><tr><th>source</th>"]
html += [f"<th>{c}</th>" for c in configs] + ["</tr>"]
for case in cases:
    html.append(f"<tr><td><img src='{thumb(os.path.join(HERE, 'src', case['file']))}'><div class=n>{case['id']}: {case['instruction']}</div></td>")
    for cfg in configs:
        d = os.path.join(a.out, cfg)
        f = next((x for x in os.listdir(d) if x.startswith(case["id"] + ".")), None) if os.path.isdir(d) else None
        if not f:
            html.append("<td></td>")
            continue
        s = scores.get((cfg, case["id"]), {})
        html.append(f"<td><img src='{thumb(os.path.join(d, f))}'><div>SC {s.get('sc','-')} PQ {s.get('pq','-')} PR {s.get('pr','-')}"
                    f" | {walls.get((cfg, case['id']), 0):.0f}s</div><div class=n>{s.get('notes','')}</div></td>")
    html.append("</tr>")
html.append("</table>")
open(os.path.join(a.out, "report.html"), "w").write("".join(html))
print(os.path.join(a.out, "report.html"))
