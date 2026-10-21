#!/usr/bin/env python3
"""Concurrent chat throughput against a running omniserve-native.

Each stream sends roleplay-shaped requests (persona system prompt + a few
turns) with SSE streaming and records TTFT, completion tokens and wall time.
Prints one JSON line per concurrency level.

  scripts/llm_throughput_bench.py --port 8795 --conc 1,4,8,16 --requests 32
"""
import argparse
import http.client
import json
import statistics
import threading
import time

PERSONA = (
    "You are Mira, a sharp-tongued ship's navigator on the airship Kestrel. You speak in short, "
    "vivid sentences, love maps and storms, distrust the captain's new first mate, and hide a "
    "soft spot for stray cats. Stay in character, write in the present tense, describe actions "
    "between asterisks, and never mention being an AI. The Kestrel is crossing the Shattered "
    "Expanse, a sky full of floating islands, lightning reefs and pirate kites. "
) * 4
TURNS = [
    "*I climb up to the navigation deck, soaked from the rain.* Mira, we've lost the eastern beacon. What now?",
    "The first mate says we should cut through the lightning reef to save a day. Thoughts?",
    "*A small grey cat jumps onto the chart table and sits right on our course line.*",
    "Tell me about the worst storm you ever flew through.",
    "Pirate kites on the horizon, three of them. How do we lose them?",
    "*I hand you a cup of hot cocoa.* You look like you haven't slept in two days.",
]


HEADERS = {"Content-Type": "application/json"}


def one_request(host, port, i, max_tokens, temperature, seed):
    turn = TURNS[i % len(TURNS)]
    body = json.dumps({
        "messages": [
            {"role": "system", "content": PERSONA},
            {"role": "user", "content": TURNS[(i + 3) % len(TURNS)]},
            {"role": "assistant", "content": "*Mira glances up from the charts.* Make it quick."},
            {"role": "user", "content": turn},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed + i,
        "stream": True,
        "stream_options": {"include_usage": True},
    })
    conn = http.client.HTTPConnection(host, port, timeout=600)
    t0 = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", body, HEADERS)
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError(f"status {resp.status}: {resp.read()[:200]!r}")
    ttft = None
    text = []
    usage = None
    buf = b""
    while True:
        chunk = resp.read1(65536)
        if not chunk:
            break
        buf += chunk
        while b"\n\n" in buf:
            ev, buf = buf.split(b"\n\n", 1)
            for line in ev.split(b"\n"):
                if not line.startswith(b"data: "):
                    continue
                data = line[6:]
                if data == b"[DONE]":
                    continue
                obj = json.loads(data)
                if obj.get("usage"):
                    usage = obj["usage"]
                for ch in obj.get("choices") or []:
                    piece = (ch.get("delta") or {}).get("content")
                    if piece:
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        text.append(piece)
    elapsed = time.perf_counter() - t0
    conn.close()
    toks = usage.get("completion_tokens") if usage else None
    return {"ttft": ttft or elapsed, "elapsed": elapsed, "tokens": toks, "text": "".join(text)}


def run_level(args, conc):
    results = []
    lock = threading.Lock()
    counter = iter(range(args.requests))
    errors = []

    def worker():
        while True:
            with lock:
                i = next(counter, None)
            if i is None:
                return
            try:
                r = one_request(args.host, args.port, i, args.max_tokens, args.temperature, args.seed)
            except Exception as exc:  # noqa: BLE001
                with lock:
                    errors.append(str(exc))
                continue
            with lock:
                results.append(r)

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(conc)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    toks = [r["tokens"] or max(1, len(r["text"]) // 4) for r in results]
    per_stream = [t / max(1e-6, r["elapsed"] - r["ttft"]) for t, r in zip(toks, results) if t > 1]
    ttfts = sorted(r["ttft"] for r in results)
    return {
        "conc": conc,
        "requests": len(results),
        "errors": len(errors),
        "agg_tok_s": round(sum(toks) / wall, 1),
        "stream_tok_s": round(statistics.median(per_stream), 1) if per_stream else None,
        "ttft_p50_ms": round(1000 * ttfts[len(ttfts) // 2], 1) if ttfts else None,
        "ttft_p95_ms": round(1000 * ttfts[min(len(ttfts) - 1, int(len(ttfts) * 0.95))], 1) if ttfts else None,
        "mean_tokens": round(sum(toks) / max(1, len(toks)), 1),
        "wall_s": round(wall, 2),
        "first_error": errors[0] if errors else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8795)
    ap.add_argument("--conc", default="1,2,4,8,16")
    ap.add_argument("--requests", type=int, default=0, help="per level; default 2x concurrency (min 4)")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--label", default="")
    ap.add_argument("--header", action="append", default=[], help="Name: value, repeatable")
    args = ap.parse_args()
    for h in args.header:
        k, _, v = h.partition(":")
        HEADERS[k.strip()] = v.strip()
    fixed = args.requests
    for conc in [int(c) for c in args.conc.split(",")]:
        args.requests = fixed or max(4, 2 * conc)
        row = run_level(args, conc)
        if args.label:
            row = {"label": args.label, **row}
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
