#!/usr/bin/env python3
"""Compare ra2_residency_bench arms image-by-image (same case/seed file names) against a reference arm.

cutedsl/.venv/bin/python tools/ra2_compare.py RUN_DIR --ref prod [--no-semantic] [--sheet sheet.jpg]
"""
import argparse, json, statistics, sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from image_parity_bench import SemanticScorer, pixel_metrics, load_font  # noqa: E402

PROMPTS = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--no-semantic", action="store_true")
    ap.add_argument("--sheet")
    ap.add_argument("--semantic-module-dir", default=str(Path(__file__).resolve().parents[2] / "cutedsl-site" / "inference"))
    a = ap.parse_args()
    run = Path(a.run)
    rows = [json.loads(l) for l in (run / "runs.jsonl").read_text().splitlines()]
    prompt_of = {}
    for r in rows:
        if "case" in r:
            prompt_of[(r["case"], r["i"])] = r.get("prompt")
    scorer = None if a.no_semantic else SemanticScorer(Path(a.semantic_module_dir))
    from ra2_residency_bench import PROMPTS as T2I, EDIT_PROMPTS
    ref_dir = run / a.ref
    arms = sorted(p.name for p in run.iterdir() if p.is_dir() and p.name != a.ref)
    out, sheet_rows = {}, []
    for arm in arms:
        metrics = []
        for ref_png in sorted(ref_dir.glob("*.png")):
            if "_warm_" in ref_png.name:
                continue
            cand = run / arm / ref_png.name
            if not cand.exists():
                continue
            ri, ci = Image.open(ref_png).convert("RGB"), Image.open(cand).convert("RGB")
            m = pixel_metrics(ri, ci)
            case, idx = ref_png.stem.split("_")[0], int(ref_png.stem.split("_")[1])
            if scorer:
                pi = prompt_of.get((case, idx), 0) or 0
                prompt = (EDIT_PROMPTS if case.startswith("edit") else T2I)[pi]
                m.update(scorer.score(prompt, ri, ci))
            m["file"] = ref_png.name
            metrics.append(m)
            if len(sheet_rows) < 24:
                sheet_rows.append((arm, ref_png, cand, m))
        if not metrics:
            continue
        agg = {k: round(statistics.fmean(x[k] for x in metrics if isinstance(x.get(k), (int, float)) and x[k] != float("inf")), 4)
               for k in ("psnr_db", "ssim_global", "clip_image_cosine", "clip_prompt_delta", "aesthetic_delta")
               if any(isinstance(x.get(k), (int, float)) and x[k] != float("inf") for x in metrics)}
        agg["identical"] = sum(1 for x in metrics if x.get("identical"))
        agg["n"] = len(metrics)
        agg["min_clip_image_cosine"] = round(min((x.get("clip_image_cosine", 1) for x in metrics)), 4)
        out[arm] = agg
        print(json.dumps({"arm": arm, **agg}), flush=True)
    (run / f"compare_vs_{a.ref}.json").write_text(json.dumps(out, indent=2))
    if a.sheet and sheet_rows:
        cw, ch, lh, pad = 300, 300, 34, 8
        sheet = Image.new("RGB", (pad + 2 * (cw + pad), pad + len(sheet_rows) * (ch + lh + pad)), "white")
        draw, font = ImageDraw.Draw(sheet), load_font(13)
        for i, (arm, rp, cp, m) in enumerate(sheet_rows):
            y = pad + i * (ch + lh + pad)
            for j, (lab, p) in enumerate(((a.ref, rp), (arm, cp))):
                im = Image.open(p).convert("RGB"); im.thumbnail((cw, ch))
                sheet.paste(im, (pad + j * (cw + pad), y))
                draw.text((pad + j * (cw + pad), y + ch + 2), f"{lab} {p.name}", fill=(0, 0, 0), font=font)
            draw.text((pad + cw + pad, y + ch + 17), f"PSNR {m.get('psnr_db', 0):.1f} CLIP {m.get('clip_image_cosine', 0):.3f}", fill=(80, 80, 80), font=font)
        sheet.save(a.sheet, quality=88)


if __name__ == "__main__":
    main()
