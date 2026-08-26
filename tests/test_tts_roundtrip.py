"""End-to-end TTS quality gate: generate speech, transcribe it back, compare.

Round-trips text through an OpenAI-compatible speech server:
  1. POST /v1/audio/speech            -> audio (base64 JSON or audio_url)
  2. POST /v1/audio/transcriptions    -> text
  3. normalized similarity >= --min-ratio passes

Env or flags:
  OPENPATHS_TTS_URL   base URL of the speech server (default http://127.0.0.1:8092)
  OPENPATHS_API_KEY   bearer token
Prints tts_ms / stt_ms / bytes / similarity; exit 1 on quality failure.
"""

import argparse
import base64
import json
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request


def http_json(url, payload, key, timeout=120):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def transcribe(url, key, wav_path, model, timeout=180):
    boundary = "omnibench"
    with open(wav_path, "rb") as f:
        audio = f.read()
    body = b""
    for name, value in (("model", model),):
        body += (f"--{boundary}\r\nContent-Disposition: form-data; "
                 f'name="{name}"\r\n\r\n{value}\r\n').encode()
    body += (f"--{boundary}\r\nContent-Disposition: form-data; "
             'name="file"; filename="audio.wav"\r\n'
             "Content-Type: audio/wav\r\n\r\n").encode() + audio + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{url}/v1/audio/transcriptions",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                 "Authorization": f"Bearer {key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read()).get("text", "")


def normalize(text):
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).split()


def similarity(expected_words, got_text):
    got = normalize(got_text)
    if not expected_words:
        return 1.0 if got else 0.0
    hits = sum(1 for w in expected_words if w in got)
    return hits / len(expected_words)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="The quick brown fox jumps over the lazy dog.")
    ap.add_argument("--voice", default="Kore")
    ap.add_argument("--model", default=os.environ.get("OPENPATHS_TTS_MODEL",
                                                      "gemini-3.1-flash-tts-preview"))
    ap.add_argument("--stt-model", default="whisper-large-v3-turbo")
    ap.add_argument("--min-ratio", type=float, default=0.6)
    args = ap.parse_args()

    base = os.environ.get("OPENPATHS_TTS_URL", "http://127.0.0.1:8092").rstrip("/")
    key = os.environ.get("OPENPATHS_API_KEY", "")
    if not key:
        print("OPENPATHS_API_KEY is required", file=sys.stderr)
        return 2

    t0 = time.perf_counter()
    try:
        speech = http_json(f"{base}/v1/audio/speech",
                           {"model": args.model, "input": args.text,
                            "voice": args.voice}, key)
    except urllib.error.HTTPError as e:
        print(f"tts failed: HTTP {e.code}: {e.read().decode()[:300]}", file=sys.stderr)
        return 2
    tts_ms = (time.perf_counter() - t0) * 1000

    encoded = str(speech.get("audio", "")).strip()
    audio_url = str(speech.get("audio_url", "")).strip()
    if encoded.startswith("http"):
        audio_url, encoded = encoded, ""
    if encoded:
        audio = base64.b64decode(encoded)
    elif audio_url:
        with urllib.request.urlopen(audio_url, timeout=120) as r:
            audio = r.read()
    else:
        print("speech response had no audio", file=sys.stderr)
        return 2

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio)
        wav_path = f.name

    t1 = time.perf_counter()
    try:
        text = transcribe(base, key, wav_path, args.stt_model)
    except urllib.error.HTTPError as e:
        print(f"stt failed: HTTP {e.code}: {e.read().decode()[:300]}", file=sys.stderr)
        os.unlink(wav_path)
        return 2
    stt_ms = (time.perf_counter() - t1) * 1000

    ratio = similarity(normalize(args.text), text)
    print(f"tts {tts_ms:7.0f} ms | {len(audio)/1024:.0f} KiB | "
          f"stt {stt_ms:7.0f} ms | ratio {ratio:.2f}")
    print(f"expected : {args.text}")
    print(f"heard    : {text}")
    os.unlink(wav_path)

    passed = ratio >= args.min_ratio
    print("PASS" if passed else f"FAIL below --min-ratio {args.min_ratio}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
