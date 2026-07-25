#!/usr/bin/env python3
"""Quality bench for the models omniserve-native can inference.

Transport throughput is already covered by tools/bench_http.c. This harness
covers the other half: whether the served models still answer *correctly*
after a KV-cache, sampler, template, or model-artifact change. Every check
runs against a live gateway over HTTP, so it exercises the same C data plane
production uses.

    ./tools/quality_bench.py --port 8791
    ./tools/quality_bench.py --suite embedding --json report.json
    ./tools/quality_bench.py --update-baseline      # accept current scores

Exits non-zero when a graded metric falls below its baseline (minus
--tolerance) or a hard invariant fails, so CI can gate on it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request

DEFAULT_BASELINE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "performance", "quality-baseline.json"
)

# --- graded data -----------------------------------------------------------
# Small and deterministic on purpose: a quality gate has to run in seconds
# beside the transport bench, and a roleplay-tuned model is scored on
# instruction following rather than trivia depth.

TASKS = [
    # (prompt, accepted answer patterns)
    ("What is 17 + 25? Reply with the number only.", [r"\b42\b"]),
    ("What is 12 * 12? Reply with the number only.", [r"\b144\b"]),
    ("What is the capital city of France? Reply with the city name only.", [r"paris"]),
    ("What is the capital city of Japan? Reply with the city name only.", [r"tokyo"]),
    ("Which is larger, 9.11 or 9.9? Reply with the number only.", [r"9\.9(?!\d)"]),
    ("How many letters are in the word 'omniserve'? Reply with the number only.", [r"\b9\b"]),
    ("Complete the sequence with one number: 2, 4, 8, 16,", [r"\b32\b"]),
    ("Reply with exactly one word: the opposite of 'hot'.", [r"cold"]),
]

# Sentence pairs for embedding separation. Positives paraphrase each other;
# negatives share vocabulary but not meaning, which is what catches a
# pooling or quantization regression that raw self-similarity misses.
STS_POSITIVE = [
    ("A man is playing a guitar.", "Someone is strumming a guitar."),
    ("The server returned an error.", "The backend responded with a failure."),
    ("She bought a red car.", "She purchased a crimson automobile."),
    ("How do I reset my password?", "What is the process for changing my password?"),
    ("The cat slept on the couch.", "A cat was napping on the sofa."),
]
STS_NEGATIVE = [
    ("A man is playing a guitar.", "A man is frying an egg."),
    ("The server returned an error.", "The server rack was delivered on Tuesday."),
    ("She bought a red car.", "She sold her blue bicycle."),
    ("How do I reset my password?", "How do I reset the oven timer?"),
    ("The cat slept on the couch.", "The dog barked at the mailman."),
]

DRIFT_TEXTS = [
    "omniserve native quality bench reference vector",
    "The quick brown fox jumps over the lazy dog.",
    "Stable diffusion and large language models share one GPU.",
]


class Bench:
    def __init__(self, base: str, secret: str | None, timeout: float):
        self.base = base.rstrip("/")
        self.secret = secret
        self.timeout = timeout
        self.results: list[dict] = []
        self.metrics: dict[str, float] = {}

    # --- transport ---------------------------------------------------------
    def call(self, path: str, payload=None, method=None, raw=False):
        url = f"{self.base}{path}"
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
        if self.secret:
            headers["X-API-Key"] = self.secret
        req = urllib.request.Request(url, data=data, headers=headers,
                                     method=method or ("POST" if data else "GET"))
        started = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read()
                elapsed = (time.monotonic() - started) * 1000.0
                if raw:
                    return resp.status, body, elapsed
                return resp.status, json.loads(body) if body else None, elapsed
        except urllib.error.HTTPError as exc:
            body = exc.read()
            elapsed = (time.monotonic() - started) * 1000.0
            if raw:
                return exc.code, body, elapsed
            try:
                return exc.code, json.loads(body), elapsed
            except Exception:
                return exc.code, {"error": body[:200].decode("utf-8", "replace")}, elapsed
        except Exception as exc:  # connection refused, timeout, reset
            return 0, {"error": str(exc)}, (time.monotonic() - started) * 1000.0

    def chat(self, content: str, **kw):
        payload = {"messages": [{"role": "user", "content": content}],
                   "max_tokens": kw.pop("max_tokens", 64), "temperature": kw.pop("temperature", 0.0)}
        payload.update(kw)
        status, body, ms = self.call("/v1/chat/completions", payload)
        text = ""
        usage = {}
        if isinstance(body, dict):
            usage = body.get("usage") or {}
            try:
                text = body["choices"][0]["message"]["content"] or ""
            except Exception:
                text = ""
        return status, text, usage, ms

    # --- reporting ---------------------------------------------------------
    def unauthorized(self, name: str, status: int) -> bool:
        """A 401/403 means credentials were not supplied, not a broken model."""
        if status not in (401, 403):
            return False
        self.results.append({"check": name, "ok": True, "detail": "skipped: auth required",
                             "skipped": True})
        print(f"  [SKIP] {name} — http={status}; pass --secret to grade this route", flush=True)
        return True

    def check(self, name: str, ok: bool, detail: str = "", metric: float | None = None):
        self.results.append({"check": name, "ok": bool(ok), "detail": detail})
        if metric is not None:
            self.metrics[name] = round(float(metric), 6)
        mark = "PASS" if ok else "FAIL"
        line = f"  [{mark}] {name}"
        if detail:
            line += f" — {detail}"
        print(line, flush=True)
        return ok

    def score(self, name: str, value: float, detail: str = ""):
        """A graded metric: recorded, compared against the baseline later."""
        self.metrics[name] = round(float(value), 6)
        self.results.append({"check": name, "ok": True, "detail": detail, "score": value})
        print(f"  [SCORE] {name} = {value:.4f}" + (f" — {detail}" if detail else ""), flush=True)


# --- suites ----------------------------------------------------------------

def suite_gateway(b: Bench, status_doc: dict) -> None:
    print("gateway")
    ok, health, _ = b.call("/health")
    b.check("gateway.health", ok == 200 and isinstance(health, dict),
            f"status={(health or {}).get('status')}")
    gpu = status_doc.get("gpu") or {}
    # Newer builds report placement; older ones simply omit the field.
    if gpu:
        b.check("gateway.gpu_placement", not gpu.get("degraded", False),
                f"placement={gpu.get('placement')} device={gpu.get('device')} "
                f"kv={gpu.get('kv_type')} flash_attn={gpu.get('flash_attn')}")
    if "vram_available" in status_doc:
        b.check("gateway.vram_readable", bool(status_doc["vram_available"]),
                f"free={status_doc.get('vram_free_gib')} GiB")
    code, models, _ = b.call("/v1/models")
    served = [m.get("id") for m in ((models or {}).get("data") or [])]
    b.check("gateway.models_listed", code == 200 and bool(served), ", ".join(served[:6]))


def suite_llm(b: Bench) -> None:
    print("llm")
    status, text, usage, ms = b.chat("Say hello in one short sentence.", max_tokens=32)
    if b.unauthorized("llm.reachable", status):
        return
    if not b.check("llm.reachable", status == 200 and bool(text.strip()),
                   f"http={status} {ms:.0f}ms"):
        return

    prompt = "List three primary colours, comma separated."
    buster = "Unrelated priming text about geology and sedimentary rocks."

    # Cold-path determinism: two full prefills of the same prompt must be
    # byte-identical at temperature 0. A mismatch means slot state or the
    # sampler chain leaks between requests.
    b.chat(buster, max_tokens=8)
    _, cold_a, cold_usage_a, cold_ms = b.chat(prompt, max_tokens=64)
    b.chat(buster, max_tokens=8)
    _, cold_b, _, _ = b.chat(prompt, max_tokens=64)
    b.check("llm.cold_determinism", cold_a == cold_b,
            "identical" if cold_a == cold_b
            else f"diverges at char {common_prefix_len(cold_a, cold_b)}/{len(cold_a)}")

    # Warm path: the same prompt served off a reused KV prefix. Re-decoding one
    # token against cached keys is not bit-identical to a chunked prefill, so
    # greedy near-ties can flip and the continuation legitimately differs.
    # Tracked as a score, because agreement collapsing to near zero means the
    # reuse offset is wrong rather than merely imprecise.
    _, warm, warm_usage, warm_ms = b.chat(prompt, max_tokens=64)
    diverge = common_prefix_len(cold_a, warm)
    agreement = diverge / max(len(cold_a), len(warm), 1)
    b.score("llm.prefix_cache_agreement", agreement,
            f"diverges at char {diverge}/{max(len(cold_a), len(warm))}")
    b.check("llm.prefix_cache_not_corrupt", agreement > 0.05 or cold_a == warm,
            f"agreement={agreement:.3f}")
    cached = int(warm_usage.get("cached_prompt_tokens") or 0)
    b.check("llm.prefix_cache_active", cached > 0,
            f"cached_prompt_tokens={cached} cold={cold_ms:.0f}ms warm={warm_ms:.0f}ms "
            f"(cold reported {cold_usage_a.get('cached_prompt_tokens')})")
    if cold_ms > 0:
        b.score("llm.prefix_cache_speedup", cold_ms / max(warm_ms, 1e-6),
                f"{cold_ms:.0f}ms -> {warm_ms:.0f}ms")

    # Contract adherence. These are the knobs the public API promises.
    _, capped, capped_usage, _ = b.chat("Write a long paragraph about the sea.", max_tokens=16)
    b.check("llm.max_tokens_honored", int(capped_usage.get("completion_tokens") or 0) <= 16,
            f"completion_tokens={capped_usage.get('completion_tokens')}")

    _, stopped, _, _ = b.chat("Count: one two three four five", max_tokens=48,
                              stop=["three"])
    b.check("llm.stop_sequence_honored", "three" not in stopped.lower(),
            f"text={stopped[:60]!r}")

    status, legacy, ms = b.call("/api/v1/generate", {
        "text": "Hi I am bored so looking", "number_of_results": 1, "max_length": 60,
        "max_sentences": 1, "min_probability": 0.7, "model": "best",
        "enable_thinking": False,
    })
    generated = ""
    if isinstance(legacy, list) and legacy:
        generated = (legacy[0] or {}).get("generated_text") or ""
    b.check("llm.legacy_generate", status == 200 and bool(generated.strip()),
            f"http={status} {ms:.0f}ms text={generated[:50]!r}")
    b.check("llm.max_sentences_honored", generated.count(".") <= 1,
            f"terminators={generated.count('.')}")
    # Reasoning tags must never reach the public field.
    b.check("llm.no_reasoning_leak",
            not re.search(r"<\s*/?\s*(think|thought|reasoning)\b", generated, re.I),
            f"text={generated[:50]!r}")

    correct = 0
    misses = []
    for prompt, patterns in TASKS:
        _, answer, _, _ = b.chat(prompt, max_tokens=32)
        hit = any(re.search(p, answer, re.I) for p in patterns)
        correct += hit
        if not hit:
            misses.append(f"{prompt[:28]}->{answer.strip()[:24]!r}")
    b.score("llm.task_accuracy", correct / len(TASKS),
            f"{correct}/{len(TASKS)}" + (f"; missed {misses[0]}" if misses else ""))


def common_prefix_len(a: str, b_: str) -> int:
    limit = min(len(a), len(b_))
    i = 0
    while i < limit and a[i] == b_[i]:
        i += 1
    return i


def cosine(a: list[float], b_: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b_))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b_))
    return dot / (na * nb) if na and nb else 0.0


def suite_embedding(b: Bench, baseline: dict) -> dict:
    print("embedding")

    def embed(text, **kw):
        payload = {"input": text}
        payload.update(kw)
        status, body, ms = b.call("/v1/embeddings", payload)
        vectors = []
        if isinstance(body, dict):
            for row in body.get("data") or []:
                vectors.append(row.get("embedding") or [])
        return status, vectors, ms

    status, vectors, ms = embed("omniserve native embedding probe")
    if b.unauthorized("embedding.reachable", status):
        return {}
    if not b.check("embedding.reachable", status == 200 and vectors and len(vectors[0]) > 0,
                   f"http={status} dims={len(vectors[0]) if vectors else 0} {ms:.0f}ms"):
        return {}
    dims = len(vectors[0])

    _, again, _ = embed("omniserve native embedding probe")
    b.check("embedding.determinism", again and cosine(vectors[0], again[0]) > 0.999999,
            f"self_cosine={cosine(vectors[0], again[0]):.6f}" if again else "no vector")

    # Batched shape must equal the scalar shape element-wise, or callers get
    # silently different vectors depending on how they batch.
    status, batched, _ = embed(["omniserve native embedding probe", DRIFT_TEXTS[1]])
    if b.check("embedding.batch_shape", status == 200 and len(batched) == 2,
               f"returned={len(batched)}"):
        b.check("embedding.batch_matches_single",
                cosine(batched[0], vectors[0]) > 0.999999,
                f"cosine={cosine(batched[0], vectors[0]):.6f}")

    status, legacy, _ = b.call("/api/v1/feature-extraction",
                               {"text": "omniserve native embedding probe", "num_features": 128})
    flat = legacy if isinstance(legacy, list) else (legacy or {}).get("embedding")
    b.check("embedding.legacy_num_features",
            status == 200 and isinstance(flat, list) and len(flat) == 128,
            f"http={status} len={len(flat) if isinstance(flat, list) else 'n/a'}")

    pos, neg = [], []
    for left, right in STS_POSITIVE:
        _, vecs, _ = embed([left, right])
        if len(vecs) == 2:
            pos.append(cosine(vecs[0], vecs[1]))
    for left, right in STS_NEGATIVE:
        _, vecs, _ = embed([left, right])
        if len(vecs) == 2:
            neg.append(cosine(vecs[0], vecs[1]))
    if pos and neg:
        margin = statistics.fmean(pos) - statistics.fmean(neg)
        pairs = [(p, n) for p in pos for n in neg]
        ranking = sum(1 for p, n in pairs if p > n) / len(pairs)
        b.score("embedding.sts_margin", margin,
                f"pos={statistics.fmean(pos):.3f} neg={statistics.fmean(neg):.3f}")
        b.score("embedding.sts_ranking_accuracy", ranking,
                f"{int(ranking * len(pairs))}/{len(pairs)} pairs ordered")

    # Drift: the same texts must keep producing the same vectors across
    # requantization and backend changes.
    drift_now = []
    for text in DRIFT_TEXTS:
        _, vecs, _ = embed(text)
        drift_now.append(vecs[0] if vecs else [])
    reference = (baseline.get("embedding_reference") or {}).get("vectors")
    if reference and len(reference) == len(drift_now):
        similarities = [cosine(r, n) for r, n in zip(reference, drift_now) if r and n]
        worst = min(similarities) if similarities else 0.0
        b.score("embedding.drift_cosine", worst, f"worst of {len(similarities)} reference texts")
    else:
        print("  [INFO] no embedding reference stored; run --update-baseline to record one")
    return {"dims": dims, "vectors": drift_now}


def suite_image(b: Bench, status_doc: dict) -> None:
    embedded = (status_doc.get("diffusion") or {}).get("ready")
    upstream = (status_doc.get("upstreams") or {}).get("image")
    if not embedded and not upstream:
        print("image\n  [SKIP] no diffusion backend configured")
        return
    print("image")
    status, body, ms = b.call("/v1/images/generations", {
        "prompt": "a red cube on a white background, product photo",
        "n": 1, "size": "512x512", "steps": 4,
    }, raw=True)
    if b.unauthorized("image.reachable", status):
        return
    if not b.check("image.reachable", status == 200 and bool(body), f"http={status} {ms:.0f}ms"):
        return
    magic_png = body[:8] == b"\x89PNG\r\n\x1a\n"
    magic_webp = body[:4] == b"RIFF" and body[8:12] == b"WEBP"
    magic_jpeg = body[:2] == b"\xff\xd8"
    is_json = body.lstrip()[:1] in (b"{", b"[")
    if is_json:
        # Providers may return a URL/base64 envelope instead of raw bytes.
        try:
            doc = json.loads(body)
            payload = (doc.get("data") or [{}])[0]
            b.check("image.envelope", bool(payload.get("url") or payload.get("b64_json")),
                    f"keys={sorted(payload)[:4]}")
        except Exception as exc:
            b.check("image.envelope", False, str(exc)[:60])
        return
    b.check("image.encoded_bytes", magic_png or magic_webp or magic_jpeg,
            f"{len(body)} bytes magic={body[:4]!r}")
    # A uniform buffer is what a failed decode or a black image looks like.
    sample = body[: 64 * 1024]
    distinct = len(set(sample))
    b.check("image.non_degenerate", distinct > 64, f"{distinct} distinct byte values")
    b.score("image.latency_ms", ms, f"{ms:.0f}ms")


def suite_audio(b: Bench, status_doc: dict) -> None:
    upstreams = status_doc.get("upstreams") or {}
    if not upstreams.get("tts"):
        print("audio\n  [SKIP] no tts upstream configured")
        return
    print("audio")
    status, body, ms = b.call("/v1/audio/speech", {
        "model": "tts-1", "input": "OmniServe quality bench.", "voice": "default",
    }, raw=True)
    if b.unauthorized("audio.tts_reachable", status):
        return
    if not b.check("audio.tts_reachable", status == 200 and len(body) > 1024,
                   f"http={status} {len(body)} bytes {ms:.0f}ms"):
        return
    wav = body[:4] == b"RIFF"
    mp3 = body[:3] == b"ID3" or body[:2] in (b"\xff\xfb", b"\xff\xf3")
    ogg = body[:4] == b"OggS"
    b.check("audio.tts_container", wav or mp3 or ogg, f"magic={body[:4]!r}")
    b.score("audio.tts_latency_ms", ms, f"{ms:.0f}ms")


# --- baseline gating -------------------------------------------------------

# Higher is better unless listed here; latencies regress upward.
LOWER_IS_BETTER = {"image.latency_ms", "audio.tts_latency_ms"}


def compare_baseline(metrics: dict, baseline: dict, tolerance: float) -> list[str]:
    stored = baseline.get("metrics") or {}
    regressions = []
    for name, value in metrics.items():
        if name not in stored:
            continue
        was = stored[name]
        if name in LOWER_IS_BETTER:
            allowed = was * (1.0 + max(tolerance, 0.25))  # latency is noisy
            if value > allowed:
                regressions.append(f"{name}: {value:.4f} > allowed {allowed:.4f} (was {was:.4f})")
        else:
            allowed = was - abs(was) * tolerance - 1e-9
            if value < allowed:
                regressions.append(f"{name}: {value:.4f} < allowed {allowed:.4f} (was {was:.4f})")
    return regressions


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=None, help="gateway base URL")
    ap.add_argument("--port", type=int, default=int(os.environ.get("OMNISERVE_NATIVE_PORT", 8791)))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--secret", default=os.environ.get("OMNISERVE_NATIVE_SECRET"))
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--suite", action="append",
                    choices=["gateway", "llm", "embedding", "image", "audio"],
                    help="run only these suites (repeatable)")
    ap.add_argument("--baseline", default=DEFAULT_BASELINE)
    ap.add_argument("--tolerance", type=float, default=0.05,
                    help="fractional slack allowed against a stored score")
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--json", default=None, help="write the full report here")
    args = ap.parse_args()

    base = args.base or f"http://{args.host}:{args.port}"
    baseline = {}
    if os.path.exists(args.baseline):
        with open(args.baseline) as fh:
            baseline = json.load(fh)

    b = Bench(base, args.secret, args.timeout)
    code, status_doc, _ = b.call("/status")
    if code != 200 or not isinstance(status_doc, dict):
        print(f"cannot reach {base}/status: {status_doc}", file=sys.stderr)
        return 2
    print(f"omniserve-native quality bench -> {base}\n")

    suites = args.suite or ["gateway", "llm", "embedding", "image", "audio"]
    embed_reference = {}
    if "gateway" in suites:
        suite_gateway(b, status_doc)
    if "llm" in suites and ((status_doc.get("llm") or {}).get("ready")
                            or (status_doc.get("upstreams") or {}).get("llm")):
        suite_llm(b)
    elif "llm" in suites:
        print("llm\n  [SKIP] no llm backend configured")
    if "embedding" in suites and ((status_doc.get("embedding") or {}).get("ready")
                                 or (status_doc.get("upstreams") or {}).get("embedding")):
        embed_reference = suite_embedding(b, baseline)
    elif "embedding" in suites:
        print("embedding\n  [SKIP] no embedding backend configured")
    if "image" in suites:
        suite_image(b, status_doc)
    if "audio" in suites:
        suite_audio(b, status_doc)

    failures = [r for r in b.results if not r["ok"]]
    regressions = [] if args.update_baseline else compare_baseline(
        b.metrics, baseline, args.tolerance)

    print()
    print(f"checks: {len(b.results) - len(failures)}/{len(b.results)} passed, "
          f"{len(b.metrics)} scores recorded")
    for failure in failures:
        print(f"  FAILED {failure['check']}: {failure['detail']}")
    for regression in regressions:
        print(f"  REGRESSED {regression}")

    report = {
        "base": base,
        "model": {
            "llm": (status_doc.get("llm") or {}).get("model"),
            "embedding": (status_doc.get("embedding") or {}).get("model"),
            "diffusion": (status_doc.get("diffusion") or {}).get("model"),
        },
        "gpu": status_doc.get("gpu"),
        "checks": b.results,
        "metrics": b.metrics,
        "failures": [f["check"] for f in failures],
        "regressions": regressions,
    }
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"  report -> {args.json}")

    if args.update_baseline:
        payload = {"metrics": b.metrics, "model": report["model"]}
        if embed_reference.get("vectors"):
            payload["embedding_reference"] = {
                "texts": DRIFT_TEXTS,
                "dims": embed_reference.get("dims"),
                "vectors": [[round(v, 6) for v in vec] for vec in embed_reference["vectors"]],
            }
        os.makedirs(os.path.dirname(os.path.abspath(args.baseline)), exist_ok=True)
        with open(args.baseline, "w") as fh:
            json.dump(payload, fh, indent=1)
        print(f"  baseline updated -> {args.baseline}")

    return 1 if failures or regressions else 0


if __name__ == "__main__":
    sys.exit(main())
