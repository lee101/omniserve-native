#!/usr/bin/env python3
"""Compare a candidate image backend with the current production reference.

The benchmark keeps request payloads identical, stores both outputs and a
contact sheet, and combines retention metrics with CPU CLIP/aesthetic scores.
References can be captured before a resident backend is stopped, then compared
later. It never changes routing and marks requests as background work.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "performance" / "image-parity-corpus.json"


def now_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def decode_image_response(body: bytes, content_type: str, timeout: float) -> tuple[Image.Image, dict]:
    stripped = body.lstrip()
    if "json" not in content_type.lower() and stripped[:1] not in (b"{", b"["):
        return Image.open(io.BytesIO(body)).convert("RGB"), {"transport": "raw"}
    doc = json.loads(body)
    if isinstance(doc, list):
        row = doc[0] if doc else {}
    else:
        rows = doc.get("data") or []
        row = rows[0] if rows else doc
    encoded = row.get("b64_json") or row.get("image_base64")
    if encoded:
        return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB"), {
            "transport": "b64_json",
            "model": doc.get("model") if isinstance(doc, dict) else None,
            "seed": row.get("seed"),
            "inference_time_ms": row.get("inference_time_ms"),
            "format": row.get("format"),
            "teleport": row.get("teleport"),
        }
    url = row.get("url") or row.get("path") or doc.get("url") or doc.get("path")
    if not url:
        raise ValueError("image response contains neither bytes, b64_json, nor URL")
    with urllib.request.urlopen(url, timeout=timeout) as response:
        image = Image.open(io.BytesIO(response.read())).convert("RGB")
    return image, {
        "transport": "url",
        "url": url,
        "model": doc.get("model") if isinstance(doc, dict) else None,
        "seed": row.get("seed"),
        "teleport": row.get("teleport"),
    }


def request_image(base: str, payload: dict, secret: str | None, timeout: float) -> tuple[Image.Image, dict]:
    headers = {
        "Accept": "application/json, image/*",
        "Content-Type": "application/json",
        "X-Omniserve-Tier": "background",
    }
    if secret:
        headers["Authorization"] = f"Bearer {secret}"
    request = urllib.request.Request(
        f"{base.rstrip('/')}/v1/images/generations",
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers=headers,
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            content_type = response.headers.get("Content-Type", "")
            status = response.status
    except urllib.error.HTTPError as exc:
        body = exc.read(4096).decode("utf-8", "replace").strip()
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    image, meta = decode_image_response(body, content_type, timeout)
    meta.update({
        "http_status": status,
        "wall_ms": round((time.perf_counter() - started) * 1000, 3),
        "content_type": content_type,
        "encoded_bytes": len(body),
    })
    return image, meta


def request_economics(meta: dict, size: str, monthly_gpu_cost: float) -> dict[str, float | str]:
    width_text, height_text = size.lower().split("x", 1)
    megapixels = int(width_text) * int(height_text) / 1_000_000
    inference_ms = meta.get("inference_time_ms")
    timing_source = "inference_time_ms" if isinstance(inference_ms, (int, float)) else "wall_ms"
    billed_ms = float(inference_ms if timing_source == "inference_time_ms" else meta["wall_ms"])
    cost = billed_ms / 1000 * monthly_gpu_cost / (730 * 3600)
    return {
        "monthly_gpu_cost_usd": monthly_gpu_cost,
        "megapixels": megapixels,
        "timing_source": timing_source,
        "billed_ms": billed_ms,
        "cost_per_image_usd": cost,
        "cost_per_megapixel_usd": cost / megapixels,
    }


def entropy(image: Image.Image) -> float:
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    probabilities = hist[hist > 0] / gray.size
    return float(-np.sum(probabilities * np.log2(probabilities)))


def global_ssim(left: np.ndarray, right: np.ndarray) -> float:
    left = left.astype(np.float64)
    right = right.astype(np.float64)
    mu_l, mu_r = float(left.mean()), float(right.mean())
    var_l, var_r = float(left.var()), float(right.var())
    covariance = float(np.mean((left - mu_l) * (right - mu_r)))
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    return ((2 * mu_l * mu_r + c1) * (2 * covariance + c2)) / (
        (mu_l * mu_l + mu_r * mu_r + c1) * (var_l + var_r + c2)
    )


def edge_cosine(left: np.ndarray, right: np.ndarray) -> float:
    def edges(array: np.ndarray) -> np.ndarray:
        gray = array.astype(np.float32).mean(axis=2)
        dx = np.diff(gray, axis=1, append=gray[:, -1:])
        dy = np.diff(gray, axis=0, append=gray[-1:, :])
        return np.hypot(dx, dy).ravel()

    a, b = edges(left), edges(right)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def histogram_intersection(left: np.ndarray, right: np.ndarray) -> float:
    scores = []
    for channel in range(3):
        a, _ = np.histogram(left[:, :, channel], bins=64, range=(0, 256), density=False)
        b, _ = np.histogram(right[:, :, channel], bins=64, range=(0, 256), density=False)
        a = a / max(1, a.sum())
        b = b / max(1, b.sum())
        scores.append(float(np.minimum(a, b).sum()))
    return statistics.fmean(scores)


def pixel_metrics(reference: Image.Image, candidate: Image.Image) -> dict[str, float | bool]:
    if reference.size != candidate.size:
        return {"same_dimensions": False}
    ref = np.asarray(reference, dtype=np.float32)
    cand = np.asarray(candidate, dtype=np.float32)
    diff = cand - ref
    mse = float(np.mean(diff * diff))
    return {
        "same_dimensions": True,
        "identical": bool(mse == 0.0),
        "mse": mse,
        "mae": float(np.mean(np.abs(diff))),
        "psnr_db": float("inf") if mse == 0 else float(20 * math.log10(255 / math.sqrt(mse))),
        "ssim_global": float(global_ssim(ref, cand)),
        "edge_cosine": edge_cosine(ref, cand),
        "histogram_intersection": histogram_intersection(ref, cand),
    }


class SemanticScorer:
    def __init__(self, module_dir: Path):
        os.environ.setdefault("AESTHETIC_DEVICE", "cpu")
        sys.path.insert(0, str(module_dir))
        from aesthetic_score import get_scorer

        self.scorer = get_scorer()
        if self.scorer is None:
            raise RuntimeError("aesthetic/CLIP scorer is unavailable")

    def score(self, prompt: str, reference: Image.Image, candidate: Image.Image) -> dict[str, float]:
        embeddings, aesthetics = self.scorer.embed_and_score([reference, candidate])
        text = self.scorer.embed_texts([prompt])[0]
        ref, cand = embeddings[0], embeddings[1]
        return {
            "clip_image_cosine": float(ref @ cand),
            "clip_prompt_reference": float(ref @ text),
            "clip_prompt_candidate": float(cand @ text),
            "clip_prompt_delta": float(cand @ text - ref @ text),
            "aesthetic_reference": float(aesthetics[0]),
            "aesthetic_candidate": float(aesthetics[1]),
            "aesthetic_delta": float(aesthetics[1] - aesthetics[0]),
        }


def load_font(size: int):
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def write_contact_sheet(rows: list[dict], output: Path) -> None:
    if not rows:
        return
    cell_w, cell_h, label_h, pad = 320, 240, 58, 12
    sheet = Image.new("RGB", (pad + 3 * (cell_w + pad), pad + len(rows) * (cell_h + label_h + pad)), "white")
    draw = ImageDraw.Draw(sheet)
    font = load_font(14)
    for row_index, row in enumerate(rows):
        y = pad + row_index * (cell_h + label_h + pad)
        reference = Image.open(row["reference_path"]).convert("RGB")
        candidate_path = row.get("candidate_path")
        if candidate_path:
            candidate = Image.open(candidate_path).convert("RGB")
            diff = ImageChops.difference(reference, candidate)
        else:
            candidate = Image.new("RGB", reference.size, (127, 29, 29))
            diff = candidate.copy()
        for column, (label, image) in enumerate((("reference", reference), ("candidate", candidate), ("difference", diff))):
            x = pad + column * (cell_w + pad)
            image.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
            sheet.paste(image, (x, y))
            draw.text((x, y + cell_h + 4), f"{row['id']} · {label}", fill=(15, 23, 42), font=font)
            if column == 1:
                score = row.get("semantic", {}).get("clip_image_cosine")
                suffix = "" if score is None else f" · CLIP {score:.3f}"
                error = row["candidate"].get("error")
                timing = "error" if error else f"{row['candidate']['wall_ms']/1000:.2f}s"
                draw.text((x, y + cell_h + 25), f"{timing}{suffix}", fill=(71, 85, 105), font=font)
    sheet.save(output, quality=92)


def markdown_report(report: dict) -> str:
    lines = [
        "# OmniServe Image Parity",
        "",
        f"- reference: `{report['reference_base']}`",
        f"- candidate: `{report['candidate_base']}`",
        f"- passed: `{report['passed']}`",
        f"- semantic scorer: `{report['semantic_available']}`",
        "",
        "| Case | Size | Reference s | Candidate s | Candidate $/MP | CLIP image | Prompt Δ | Aesthetic Δ | Result |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    latency = report.get("latency") or {}
    if latency:
        lines[6:6] = [
            f"- median latency ratio ({latency['timing_source']}): "
            f"`{latency['candidate_to_reference_ratio']:.3f}` "
            f"(maximum `{latency['maximum_ratio']:.3f}`)",
        ]
    for row in report["rows"]:
        semantic = row.get("semantic") or {}
        candidate_economics = (row.get("economics") or {}).get("candidate") or {}
        cost_per_mp = candidate_economics.get("cost_per_megapixel_usd")
        cost_text = "—" if cost_per_mp is None else f"{cost_per_mp:.6f}"
        def metric(name: str) -> str:
            value = semantic.get(name)
            return "—" if value is None else f"{value:.3f}"
        lines.append(
            f"| `{row['id']}` | {row['size']} | {row['reference']['wall_ms']/1000:.2f} | "
            f"{row['candidate']['wall_ms']/1000:.2f} | "
            f"{cost_text} | "
            f"{metric('clip_image_cosine')} | "
            f"{metric('clip_prompt_delta')} | {metric('aesthetic_delta')} | "
            f"{'PASS' if row['passed'] else 'FAIL'} |"
        )
    if report["failures"]:
        lines.extend(["", "## Failures", ""] + [f"- {failure}" for failure in report["failures"]])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-base", default="http://127.0.0.1:8791")
    parser.add_argument("--candidate-base")
    parser.add_argument("--capture-reference", action="store_true")
    parser.add_argument("--reference-run", type=Path,
                        help="directory created by --capture-reference")
    parser.add_argument("--reference-secret-env", default="OMNISERVE_NATIVE_SECRET")
    parser.add_argument("--candidate-secret-env", default="OMNISERVE_NATIVE_SECRET")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evals")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--reference-steps", type=int,
                        help="override the corpus step count for the reference")
    parser.add_argument("--candidate-steps", type=int,
                        help="override the corpus step count for the candidate")
    parser.add_argument("--monthly-gpu-cost", type=float, default=1000.0,
                        help="amortized GPU host cost used for per-megapixel economics")
    parser.add_argument("--semantic-module-dir", type=Path,
                        default=ROOT.parent / "cutedsl-site" / "inference")
    parser.add_argument("--no-semantic", action="store_true")
    args = parser.parse_args()
    if args.capture_reference and args.reference_run:
        parser.error("--capture-reference and --reference-run are mutually exclusive")
    if not args.capture_reference and not args.candidate_base:
        parser.error("--candidate-base is required unless --capture-reference is used")
    if args.reference_steps is not None and args.reference_steps < 1:
        parser.error("--reference-steps must be positive")
    if args.candidate_steps is not None and args.candidate_steps < 1:
        parser.error("--candidate-steps must be positive")
    if args.monthly_gpu_cost <= 0:
        parser.error("--monthly-gpu-cost must be positive")

    corpus = json.loads(args.corpus.read_text())
    cases = corpus["cases"][: args.limit or None]
    prefix = "image_reference" if args.capture_reference else "image_parity"
    run_dir = args.output_dir / f"{prefix}_{now_slug()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    scorer = None
    semantic_error = None
    if not args.no_semantic and not args.capture_reference:
        try:
            scorer = SemanticScorer(args.semantic_module_dir)
        except Exception as exc:
            semantic_error = f"{type(exc).__name__}: {exc}"

    rows = []
    failures = []
    reference_secret = os.getenv(args.reference_secret_env)
    candidate_secret = os.getenv(args.candidate_secret_env)
    defaults = corpus.get("defaults", {})
    gates = corpus.get("gates", {})
    captured_rows = None
    if args.reference_run:
        manifest = json.loads((args.reference_run / "reference_manifest.json").read_text())
        captured_rows = {row["id"]: row for row in manifest["rows"]}
    for index, case in enumerate(cases):
        payload = {**defaults, **{key: case[key] for key in ("prompt", "size", "seed")}, "n": 1}
        reference_payload = {**payload}
        candidate_payload = {**payload}
        if args.reference_steps is not None:
            reference_payload["steps"] = args.reference_steps
        if args.candidate_steps is not None:
            candidate_payload["steps"] = args.candidate_steps
        if captured_rows is not None:
            captured = captured_rows.get(case["id"])
            if not captured or captured.get("payload") != reference_payload:
                raise ValueError(f"captured reference does not match case {case['id']}")
            reference = Image.open(captured["reference_path"]).convert("RGB")
            reference_meta = captured["reference"]
        else:
            reference, reference_meta = request_image(
                args.reference_base, reference_payload, reference_secret, args.timeout)
        if args.capture_reference:
            ref_path = run_dir / f"{index:02d}_{case['id']}_reference.png"
            reference.save(ref_path, format="PNG")
            rows.append({
                **case,
                "payload": reference_payload,
                "reference": reference_meta,
                "reference_path": str(ref_path.resolve()),
                "sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
            })
            print(json.dumps({
                "case": case["id"], "reference_ms": reference_meta["wall_ms"],
                "reference_path": str(ref_path),
            }, sort_keys=True), flush=True)
            continue
        ref_path = run_dir / f"{index:02d}_{case['id']}_reference.png"
        cand_path = run_dir / f"{index:02d}_{case['id']}_candidate.png"
        reference.save(ref_path, format="PNG")
        try:
            candidate, candidate_meta = request_image(
                args.candidate_base, candidate_payload, candidate_secret, args.timeout)
        except Exception as exc:
            message = f"candidate request failed: {type(exc).__name__}: {exc}"
            row = {
                **case,
                "reference": reference_meta,
                "candidate": {"wall_ms": 0.0, "error": message},
                "reference_path": str(ref_path),
                "candidate_path": None,
                "candidate_quality": {},
                "pixel": {},
                "semantic": {},
                "failures": [message],
                "passed": False,
            }
            rows.append(row)
            failures.append(f"{case['id']}: {message}")
            print(json.dumps({"case": case["id"], "passed": False,
                              "error": message}, sort_keys=True), flush=True)
            continue
        candidate.save(cand_path, format="PNG")
        candidate_array = np.asarray(candidate, dtype=np.float32)
        quality = {
            "entropy": entropy(candidate),
            "stddev": float(candidate_array.std()),
            "sha256": hashlib.sha256(candidate.tobytes()).hexdigest(),
        }
        metrics = pixel_metrics(reference, candidate)
        semantic = scorer.score(case["prompt"], reference, candidate) if scorer else {}
        economics = {
            "reference": request_economics(
                reference_meta, case["size"], args.monthly_gpu_cost),
            "candidate": request_economics(
                candidate_meta, case["size"], args.monthly_gpu_cost),
        }
        row_failures = []
        if not metrics.get("same_dimensions"):
            row_failures.append("dimensions differ")
        if quality["entropy"] < gates.get("candidate_entropy_min", 3.0):
            row_failures.append(f"entropy {quality['entropy']:.3f}")
        if quality["stddev"] < gates.get("candidate_stddev_min", 8.0):
            row_failures.append(f"stddev {quality['stddev']:.3f}")
        if semantic_error:
            row_failures.append(f"semantic scorer unavailable: {semantic_error}")
        if scorer:
            for name, comparator in (
                ("clip_image_cosine", "clip_image_cosine_min"),
                ("clip_prompt_delta", "clip_prompt_delta_min"),
                ("aesthetic_delta", "aesthetic_delta_min"),
            ):
                if semantic[name] < gates[comparator]:
                    row_failures.append(f"{name} {semantic[name]:.3f} < {gates[comparator]:.3f}")
        row = {
            **case,
            "reference": reference_meta,
            "candidate": candidate_meta,
            "reference_payload": reference_payload,
            "candidate_payload": candidate_payload,
            "economics": economics,
            "reference_path": str(ref_path),
            "candidate_path": str(cand_path),
            "candidate_quality": quality,
            "pixel": metrics,
            "semantic": semantic,
            "failures": row_failures,
            "passed": not row_failures,
        }
        rows.append(row)
        failures.extend(f"{case['id']}: {failure}" for failure in row_failures)
        print(json.dumps({
            "case": case["id"], "passed": row["passed"],
            "reference_ms": reference_meta["wall_ms"],
            "candidate_ms": candidate_meta["wall_ms"],
            **semantic,
        }, sort_keys=True), flush=True)

    if args.capture_reference:
        manifest = {
            "timestamp": now_slug(),
            "reference_base": args.reference_base,
            "corpus": str(args.corpus.resolve()),
            "rows": rows,
        }
        manifest_path = run_dir / "reference_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({"captured": len(rows), "run_dir": str(run_dir),
                          "manifest": str(manifest_path)}, sort_keys=True))
        return 0

    if semantic_error:
        failures.append(f"semantic scorer unavailable: {semantic_error}")
    timed_rows = [row for row in rows if not row["candidate"].get("error")]
    latency = {}
    if timed_rows:
        has_inference_timing = all(
            isinstance(row[side].get("inference_time_ms"), (int, float))
            for row in timed_rows for side in ("reference", "candidate")
        )
        timing_source = "inference_time_ms" if has_inference_timing else "wall_ms"
        reference_median = statistics.median(
            row["reference"][timing_source] for row in timed_rows)
        candidate_median = statistics.median(
            row["candidate"][timing_source] for row in timed_rows)
        ratio = candidate_median / reference_median if reference_median > 0 else float("inf")
        maximum = gates.get("median_latency_ratio_max", 1.5)
        latency = {
            "timing_source": timing_source,
            "reference_median_ms": reference_median,
            "candidate_median_ms": candidate_median,
            "reference_wall_median_ms": statistics.median(
                row["reference"]["wall_ms"] for row in timed_rows),
            "candidate_wall_median_ms": statistics.median(
                row["candidate"]["wall_ms"] for row in timed_rows),
            "candidate_to_reference_ratio": ratio,
            "maximum_ratio": maximum,
        }
        if ratio > maximum:
            failures.append(f"median latency ratio {ratio:.3f} > {maximum:.3f}")
    report = {
        "timestamp": now_slug(),
        "reference_base": (str(args.reference_run) if args.reference_run else args.reference_base),
        "candidate_base": args.candidate_base,
        "corpus": str(args.corpus),
        "semantic_available": scorer is not None,
        "semantic_error": semantic_error,
        "latency": latency,
        "rows": rows,
        "failures": failures,
        "passed": not failures,
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    (run_dir / "report.md").write_text(markdown_report(report))
    write_contact_sheet(rows, run_dir / "contact_sheet.jpg")
    print(json.dumps({"passed": report["passed"], "run_dir": str(run_dir), "failures": failures}, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
