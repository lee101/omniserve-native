#!/usr/bin/env python3
"""One-process image request for tools/make-image.sh.

The bash pipeline spent about 0.4 s in interpreter starts and subshells around
a request; this does request, retry, local fallback and file writing in one
stdlib process. Exit codes: 0 ok, 1 failure, 77 the API rejected the key.

Transient failures (connection refused or reset, 429, 502, 503, 504) are
retried with exponential backoff and jitter. When the API stays unavailable the
same request is rendered on the local RA2 lane (starting it when it is down),
so a flaky gateway still produces an image. A retried POST can be charged twice
by the API if the first attempt completed upstream; --retries 0 disables it.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import subprocess
import sys
import time
import urllib.error
import urllib.request

TRANSIENT = {429, 500, 502, 503, 504}
USER_AGENT = "make-image/2 (+omniserve-native)"


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def post(url: str, body: dict, headers: dict, timeout: float) -> tuple[int, bytes, dict]:
    data = json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, method="POST", headers={
        "Content-Type": "application/json", "User-Agent": USER_AGENT, **headers})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read(), dict(response.headers)
    except urllib.error.HTTPError as error:
        return error.code, error.read(), dict(error.headers)
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as error:
        return 0, str(error).encode(), {}


def with_retries(url: str, body: dict, headers: dict, timeout: float, retries: int, label: str):
    status, payload, response_headers = 0, b"", {}
    for attempt in range(retries + 1):
        status, payload, response_headers = post(url, body, headers, timeout)
        if status and status not in TRANSIENT:
            return status, payload
        if attempt == retries:
            break
        wait = min(30.0, 1.5 * (2 ** attempt)) * (0.75 + random.random() / 2)
        retry_after = response_headers.get("Retry-After", "")
        if retry_after.isdigit():
            wait = max(wait, min(30.0, float(retry_after)))
        reason = f"HTTP {status}" if status else payload.decode(errors="replace")[:80]
        log(f"{label}: {reason}; retry {attempt + 1}/{retries} in {wait:.1f}s")
        time.sleep(wait)
    return status, payload


def lane_up(base: str) -> bool:
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/health", timeout=3):
            return True
    except (urllib.error.URLError, OSError):
        return False


def ensure_lane(base: str) -> bool:
    if lane_up(base):
        return True
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "start-ra2-lane.sh")
    if not os.access(script, os.X_OK):
        log(f"local lane {base} is down and {script} is missing")
        return False
    log(f"local lane {base} is down; starting it (first start takes about a minute)")
    return subprocess.call([script], stdout=sys.stderr, stderr=sys.stderr) == 0 and lane_up(base)


def sniff(blob: bytes) -> str:
    if blob.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if blob.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if blob[:4] == b"RIFF" and blob[8:12] == b"WEBP":
        return "webp"
    return "bin"


def decode(value: str) -> bytes:
    if value.startswith("data:"):
        value = value.partition(",")[2]
    return base64.b64decode(value)


def fetch_url(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def artifacts(document: dict) -> list[str]:
    """Every image the response carries, whichever lane shape produced it."""
    found: list[str] = []

    def take(row) -> None:
        if not isinstance(row, dict):
            return
        for key in ("image_base64", "b64_json", "image_url", "url"):
            value = row.get(key)
            if isinstance(value, str) and value:
                found.append(value)
                return

    result = document.get("result") if isinstance(document.get("result"), dict) else document
    for container in (result, document):
        rows = container.get("images") or container.get("data")
        if isinstance(rows, list) and rows:
            for row in rows:
                take(row)
            return found
    take(result)
    if not found:
        value = document.get("saved_image_url")
        if isinstance(value, str) and value:
            found.append(value)
    return found


def out_path(args, index: int, count: int, ext: str) -> str:
    if args.out:
        stem, _ = os.path.splitext(args.out)
        return f"{stem}.{ext}"
    os.makedirs(args.out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(args.out_dir, f"image-{stamp}.{ext}")
    suffix = 1
    while os.path.exists(path):
        path = os.path.join(args.out_dir, f"image-{stamp}-{suffix}.{ext}")
        suffix += 1
    return path


def save(args, document: dict) -> list[str]:
    values = artifacts(document)
    if not values:
        raise SystemExit("response carried no image")
    paths = []
    for index, value in enumerate(values[: args.count]):
        blob = fetch_url(value) if value.startswith("http") else decode(value)
        if len(blob) < 64:
            raise SystemExit(f"image {index} is only {len(blob)} bytes")
        path = out_path(args, index, len(values), sniff(blob))
        with open(path, "wb") as handle:
            handle.write(blob)
        paths.append(path)
    return paths


def local_body(args) -> dict:
    body = {"prompt": args.prompt, "width": args.width, "height": args.height, "n": args.count,
            "output_format": "webp", "seed": args.seed}
    if args.steps:
        body["steps"] = args.steps
    if args.guidance:
        body["guidance_scale"] = args.guidance
    if args.quality == "hq":
        body["turbo"] = False
    return body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", required=True)
    parser.add_argument("--key", default="")
    parser.add_argument("--service", default="image")
    parser.add_argument("--backend", default="ra2")
    parser.add_argument("--model", default="")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--guidance", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quality", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--retries", type=int, default=int(os.environ.get("MG_RETRIES", "3")))
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("MG_TIMEOUT", "900")))
    parser.add_argument("--direct", action="store_true", help="skip the API, render on the local lane")
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--lane", default=os.environ.get("RA2_BACKEND_URL", "http://127.0.0.1:8792"))
    parser.add_argument("prompt")
    args = parser.parse_args()

    document = None
    engine = credits = "?"
    if not args.direct:
        body = {"service": args.service, "prompt": args.prompt, "model": args.model or None,
                "image_backend": args.backend, "width": args.width, "height": args.height,
                "n": args.count, "num_steps": args.steps or None, "guidance": args.guidance or None,
                "seed": args.seed or None, "quality": args.quality or None}
        body = {k: v for k, v in body.items() if v is not None}
        started = time.time()
        status, payload = with_retries(args.api.rstrip("/") + "/api/service", body,
                                       {"Authorization": f"Bearer {args.key}"},
                                       args.timeout, args.retries, "api")
        if status == 401:
            return 77
        if status in (200, 201, 202):
            document = json.loads(payload)
            result = document.get("result") if isinstance(document.get("result"), dict) else {}
            engine, credits = result.get("engine", "?"), document.get("credits_used", "?")
        else:
            try:
                message = json.loads(payload).get("error", payload[:200])
            except (ValueError, AttributeError):
                message = payload[:200].decode(errors="replace")
            log(f"api: HTTP {status or 'unreachable'} after {time.time() - started:.1f}s: {message}")
            if args.no_fallback:
                return 1
    if document is None:
        if not ensure_lane(args.lane):
            log("no local lane available")
            return 1
        log("rendering on the local lane")
        headers = {}
        secret = os.environ.get("RA2_BACKEND_SECRET", "")
        if secret:
            headers = {"Authorization": f"Bearer {secret}", "X-API-Key": secret}
        status, payload = with_retries(args.lane.rstrip("/") + "/v1/images/generations", local_body(args),
                                       headers, args.timeout, 1, "lane")
        if status != 200:
            log(f"lane: HTTP {status}: {payload[:200].decode(errors='replace')}")
            return 1
        document = json.loads(payload)
        engine, credits = "omniserve-native(local)", 0

    if args.json:
        json.dump(document, sys.stdout)
        return 0
    paths = save(args, document)
    log(f"engine={engine} credits={credits}")
    print("\n".join(paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
