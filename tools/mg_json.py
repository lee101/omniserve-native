#!/usr/bin/env python3
"""JSON plumbing for the tools/ ManifoldGen CLIs.

Every tool script talks to the API with curl and pipes the response body into
one of the ops below, so the shell never parses JSON itself (jaq and jq differ
in syntax and the image payloads are megabytes of base64). The body arrives on
stdin, never as an argument: an argv entry has a hard size limit that a single
1024x1024 image blows through. Stdlib only.

    mg_json.py get    PATH [PATH...]     first non-empty value at PATH, raw
    mg_json.py count  PATH               length of the value at PATH
    mg_json.py jget   PATH               the value at PATH, as JSON
    mg_json.py err    [STATUS]           human error from the response
    mg_json.py fetch  OUT PATH [PATH...] write the first usable artifact
    mg_json.py keys                     top-level keys, for debugging

PATH is dotted: `result.images.0.image_base64`. A numeric segment indexes a
list. Every op exits non-zero when it cannot satisfy itself, so callers can use
plain `set -e` and never have to test for emptiness by hand.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import urllib.request

USER_AGENT = "omniserve-tools/1.0"
FETCH_TIMEOUT_S = float(os.environ.get("MG_FETCH_TIMEOUT_S", "600"))


def load(body: str):
    body = body.strip()
    if not body:
        raise SystemExit("empty response body")
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"response was not JSON ({exc}); first 300 bytes: {body[:300]}")


def dig(value, dotted: str):
    for segment in dotted.split("."):
        if isinstance(value, list):
            if not segment.lstrip("-").isdigit():
                raise KeyError(dotted)
            index = int(segment)
            if not -len(value) <= index < len(value):
                raise KeyError(dotted)
            value = value[index]
        elif isinstance(value, dict):
            value = value[segment]
        else:
            raise KeyError(dotted)
    return value


def lookup(body: str, path: str):
    return dig(load(body), path)


def stringify(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)) or value is None:
        return json.dumps(value)
    return json.dumps(value)


def op_get(body: str, paths: list[str]) -> int:
    try:
        document = load(body)
    except SystemExit as exc:
        print(f"mg_json: {exc}", file=sys.stderr)
        return 1
    for path in paths:
        try:
            value = dig(document, path)
        except KeyError:
            continue
        text = stringify(value)
        if text and text != "null" and text != "{}" and text != "[]":
            print(text)
            return 0
    print(f"mg_json: none of {', '.join(paths)} present", file=sys.stderr)
    return 1


def op_count(body: str, path: str) -> int:
    try:
        value = lookup(body, path)
    except (KeyError, SystemExit) as exc:
        print(f"mg_json: {exc}", file=sys.stderr)
        return 1
    try:
        print(len(value))
    except TypeError:
        print("mg_json: value has no length", file=sys.stderr)
        return 1
    return 0


def op_jget(body: str, path: str) -> int:
    try:
        value = lookup(body, path)
    except (KeyError, SystemExit) as exc:
        print(f"mg_json: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(value, indent=2))
    return 0


def op_keys(body: str) -> int:
    try:
        document = load(body)
    except SystemExit as exc:
        print(f"mg_json: {exc}", file=sys.stderr)
        return 1
    if isinstance(document, dict):
        print(" ".join(sorted(document)))
    else:
        print(type(document).__name__)
    return 0


def op_err(body: str, status: str) -> int:
    """Print the server's error. Two envelopes coexist: {"error":"msg"} and the
    paywall shape {"error":{"code","message","subscribe_url"}}."""
    prefix = f"HTTP {status}: " if status and status != "200" else ""
    try:
        document = json.loads(body.strip() or "{}")
    except json.JSONDecodeError:
        print(f"{prefix}{body.strip()[:400]}", file=sys.stderr)
        return 1
    if not isinstance(document, dict):
        print(f"{prefix}{body.strip()[:400]}", file=sys.stderr)
        return 1
    error = document.get("error", document.get("message"))
    if isinstance(error, dict):
        parts = [str(error.get(k)) for k in ("code", "message") if error.get(k)]
        text = ": ".join(parts) or json.dumps(error)
        if error.get("subscribe_url"):
            text += f" (subscribe: {error['subscribe_url']})"
    elif error:
        text = str(error)
    elif document.get("detail"):
        text = str(document["detail"])
    else:
        text = json.dumps(document)[:400]
    print(f"{prefix}{text}", file=sys.stderr)
    return 0


def decode_artifact(value: str) -> bytes:
    if value.startswith("data:"):
        _, _, payload = value.partition(",")
        if not payload:
            raise SystemExit("data URI had no payload")
        return base64.b64decode(payload)
    return base64.b64decode(value, validate=False)



def sniff(blob: bytes) -> str | None:
    """Name the container from its magic bytes.

    The backend lane picks the container (the native gateway hands back WebP,
    fal hands back MP3, the TTS lane returns WAV), so a caller-chosen extension
    is only a guess. Returning the real one keeps every saved file openable by
    the tool that claims its type.
    """
    if blob.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if blob.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if blob.startswith(b"GIF8"):
        return "gif"
    if blob[:4] == b"RIFF" and blob[8:12] == b"WEBP":
        return "webp"
    if blob[:4] == b"RIFF" and blob[8:12] == b"WAVE":
        return "wav"
    if blob.startswith(b"OggS"):
        return "ogg"
    if blob.startswith(b"fLaC"):
        return "flac"
    if blob.startswith(b"ID3") or (len(blob) > 1 and blob[0] == 0xFF and blob[1] & 0xE0 == 0xE0):
        return "mp3"
    if blob[4:8] == b"ftyp":
        brand = blob[8:12]
        return "m4a" if brand in (b"M4A ", b"M4B ") else "mp4"
    if blob.startswith(b"\x1a\x45\xdf\xa3"):
        return "webm"
    if blob[:4] == b"RIFF" and blob[8:12] == b"AVI ":
        return "avi"
    return None


def op_fetch(body: str, paths: list[str], out: str) -> int:
    """Write the first usable artifact among PATHS and print its final path.

    An artifact is a data URI, a base64 blob or an http(s) URL; the API returns
    whichever the backend lane produced, so all three have to be accepted. The
    file is renamed when the real container differs from OUT's extension.
    """
    try:
        document = load(body)
    except SystemExit as exc:
        print(f"mg_json: {exc}", file=sys.stderr)
        return 1
    candidates: list[tuple[str, str]] = []
    for path in paths:
        try:
            value = dig(document, path)
        except KeyError:
            continue
        if isinstance(value, str) and value:
            candidates.append((path, value))
    if not candidates:
        print(f"mg_json: none of {', '.join(paths)} present", file=sys.stderr)
        return 1

    last_error = "no candidate"
    for path, value in candidates:
        try:
            if value.startswith("http://") or value.startswith("https://"):
                request = urllib.request.Request(value, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT_S) as response:
                    blob = response.read()
                if len(blob) < 64:
                    raise SystemExit(f"only {len(blob)} bytes fetched")
            else:
                blob = decode_artifact(value)
                if len(blob) < 64:
                    raise SystemExit(f"only {len(blob)} bytes decoded")
        except Exception as exc:  # noqa: BLE001 - report and try the next candidate
            last_error = f"{path}: {exc}"
            continue
        with open(out, "wb") as handle:
            handle.write(blob)
        kind = sniff(blob)
        stem, ext = os.path.splitext(out)
        final = out
        if kind and ext.lower().lstrip(".") != kind:
            final = f"{stem}.{kind}"
            os.replace(out, final)
        print(f"{len(blob)} bytes of {kind or 'unknown'} from {path} -> {final}", file=sys.stderr)
        print(final)
        return 0
    print(f"mg_json: no usable artifact ({last_error})", file=sys.stderr)
    return 1


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    op, rest = argv[1], argv[2:]
    try:
        body = sys.stdin.read()
    except Exception as exc:  # noqa: BLE001 - a read failure is just a bad body
        print(f"mg_json: could not read the response body: {exc}", file=sys.stderr)
        return 1
    if op == "get":
        return op_get(body, rest)
    if op == "count":
        return op_count(body, rest[0] if rest else "")
    if op == "jget":
        return op_jget(body, rest[0] if rest else "")
    if op == "keys":
        return op_keys(body)
    if op == "err":
        return op_err(body, rest[0] if rest else "")
    if op == "fetch":
        if len(rest) < 2:
            print("fetch needs OUT and at least one PATH", file=sys.stderr)
            return 2
        return op_fetch(body, rest[1:], rest[0])
    print(f"mg_json: unknown op {op}", file=sys.stderr)
    return 2

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))