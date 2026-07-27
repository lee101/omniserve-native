#!/usr/bin/env python3
"""WER regression harness for the OmniServe ASR serving path.

This measures the deployed HTTP endpoint rather than a model object, so a
rewrite of the hot path (Python worker -> native C gateway) can be proved
accuracy-neutral: run it against the old path, keep the report, run it against
the new one, and gate on the diff.

    # record a baseline from the current worker
    tools/wer_bench.py --manifest corpus.jsonl --out baseline.json

    # after changing the serving path, fail if accuracy regressed
    tools/wer_bench.py --manifest corpus.jsonl --out candidate.json \\
        --baseline baseline.json --max-wer-regression 0.005

Manifest is JSONL with the same field names train_asr.py uses:

    {"audio_filepath": "clips/a.wav", "text": "the reference transcript"}

Exit status is 0 when every gate passes, 1 when a gate fails, 2 on a usage or
transport error, so it can be dropped straight into CI.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import unicodedata
import urllib.error
import urllib.request
import uuid
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---- text normalization -----------------------------------------------------

# WER is only comparable when both sides are normalized identically (see
# docs/asr-training.md). This is the single canonical normalizer: casefold,
# strip punctuation, collapse whitespace. It deliberately does NOT rewrite
# numbers or expand contractions -- those are opinionated transforms that can
# flatter a model, and applying them inconsistently is how WER comparisons
# quietly become meaningless.
_PUNCT = re.compile(r"[^\w\s']", flags=re.UNICODE)
_SPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    text = _PUNCT.sub(" ", text)
    return _SPACE.sub(" ", text).strip()


def tokens(text: str) -> list[str]:
    normalized = normalize(text)
    return normalized.split() if normalized else []


# ---- edit distance ----------------------------------------------------------


@dataclass
class Counts:
    """Word-level edit counts. Aggregating these is what makes corpus WER right."""

    hits: int = 0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0

    @property
    def reference_words(self) -> int:
        return self.hits + self.substitutions + self.deletions

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    def wer(self) -> float:
        if self.reference_words == 0:
            # No reference words: only insertions can be wrong.
            return 1.0 if self.insertions else 0.0
        return self.errors / self.reference_words

    def add(self, other: "Counts") -> None:
        self.hits += other.hits
        self.substitutions += other.substitutions
        self.deletions += other.deletions
        self.insertions += other.insertions


def align(reference: list[str], hypothesis: list[str]) -> Counts:
    """Levenshtein alignment over words, returning edit counts.

    Kept dependency-free on purpose: the gate must run in a bare CI container.
    Uses two rows plus a backtrace-free counting pass via full DP when the
    inputs are small enough, which they are for evaluation clips.
    """
    n, m = len(reference), len(hypothesis)
    if n == 0:
        return Counts(insertions=m)
    if m == 0:
        return Counts(deletions=n)

    # dp[i][j] = (cost, hits, subs, dels, ins)
    prev: list[tuple[int, int, int, int, int]] = [(j, 0, 0, 0, j) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur: list[tuple[int, int, int, int, int]] = [(i, 0, 0, i, 0)]
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                c, h, s, d, ins = prev[j - 1]
                cur.append((c, h + 1, s, d, ins))
                continue
            sub = prev[j - 1]
            dele = prev[j]
            insert = cur[j - 1]
            best = min(
                (sub[0] + 1, sub[1], sub[2] + 1, sub[3], sub[4]),
                (dele[0] + 1, dele[1], dele[2], dele[3] + 1, dele[4]),
                (insert[0] + 1, insert[1], insert[2], insert[3], insert[4] + 1),
                key=lambda t: t[0],
            )
            cur.append(best)
        prev = cur
    _, hits, subs, dels, ins = prev[m]
    return Counts(hits=hits, substitutions=subs, deletions=dels, insertions=ins)


# ---- audio ------------------------------------------------------------------


def audio_seconds(path: Path) -> float:
    """Duration of a WAV clip; 0.0 when it cannot be determined (RTF is skipped)."""
    try:
        with wave.open(str(path), "rb") as w:
            rate = w.getframerate()
            return w.getnframes() / rate if rate else 0.0
    except (wave.Error, OSError):
        return 0.0


# ---- transport --------------------------------------------------------------


def post_audio(url: str, path: Path, field_name: str, timeout: float) -> str:
    """POST one clip as multipart/form-data and pull the transcript out."""
    body_bytes = path.read_bytes()
    boundary = uuid.uuid4().hex
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field_name}"; filename="{path.name}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode()
    tail = f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        url,
        data=head + body_bytes + tail,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8", "replace"))
    return extract_text(payload)


def extract_text(payload: Any) -> str:
    """Accept the shapes the ASR endpoints actually return."""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        for key in ("text", "transcript", "transcription", "output"):
            if key in payload:
                found = extract_text(payload[key])
                if found:
                    return found
    return ""


# ---- benchmark --------------------------------------------------------------


@dataclass
class Result:
    counts: Counts = field(default_factory=Counts)
    per_clip: list[dict[str, Any]] = field(default_factory=list)
    latencies: list[float] = field(default_factory=list)
    audio_total: float = 0.0
    wall_total: float = 0.0
    failures: list[dict[str, str]] = field(default_factory=list)


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{lineno}: invalid JSON: {exc}")
        if "audio_filepath" not in row or "text" not in row:
            raise SystemExit(f"{path}:{lineno}: needs audio_filepath and text")
        rows.append(row)
    if not rows:
        raise SystemExit(f"{path}: manifest is empty")
    return rows


def run(args: argparse.Namespace) -> Result:
    rows = load_manifest(args.manifest)
    root = args.audio_root or args.manifest.parent
    result = Result()

    for row in rows:
        clip = Path(row["audio_filepath"])
        if not clip.is_absolute():
            clip = root / clip
        if not clip.exists():
            result.failures.append({"clip": str(clip), "error": "missing audio file"})
            continue

        started = time.perf_counter()
        try:
            hypothesis = post_audio(args.url, clip, args.field, args.timeout)
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            result.failures.append({"clip": str(clip), "error": str(exc)})
            continue
        elapsed = time.perf_counter() - started

        counts = align(tokens(row["text"]), tokens(hypothesis))
        result.counts.add(counts)
        result.latencies.append(elapsed)
        result.wall_total += elapsed
        duration = audio_seconds(clip)
        result.audio_total += duration
        result.per_clip.append(
            {
                "clip": str(clip),
                "wer": round(counts.wer(), 6),
                "reference_words": counts.reference_words,
                "errors": counts.errors,
                "seconds": round(elapsed, 4),
                "audio_seconds": round(duration, 4),
                "reference": normalize(row["text"]),
                "hypothesis": normalize(hypothesis),
            }
        )
    return result


def report(result: Result, args: argparse.Namespace) -> dict[str, Any]:
    latencies = sorted(result.latencies)

    def pct(p: float) -> float:
        if not latencies:
            return 0.0
        idx = min(len(latencies) - 1, int(round((p / 100) * (len(latencies) - 1))))
        return round(latencies[idx], 4)

    # Corpus WER aggregates edits across the whole set. The mean of per-clip WERs
    # is reported separately because it weights a three-word clip the same as a
    # three-minute one -- useful context, wrong as a headline number.
    per_clip_wers = [c["wer"] for c in result.per_clip]
    return {
        "url": args.url,
        "manifest": str(args.manifest),
        "clips_scored": len(result.per_clip),
        "clips_failed": len(result.failures),
        "wer": round(result.counts.wer(), 6),
        "mean_clip_wer": round(statistics.fmean(per_clip_wers), 6) if per_clip_wers else 0.0,
        "counts": {
            "hits": result.counts.hits,
            "substitutions": result.counts.substitutions,
            "deletions": result.counts.deletions,
            "insertions": result.counts.insertions,
            "reference_words": result.counts.reference_words,
        },
        "latency_seconds": {
            "mean": round(statistics.fmean(latencies), 4) if latencies else 0.0,
            "p50": pct(50),
            "p90": pct(90),
            "p99": pct(99),
        },
        "audio_seconds": round(result.audio_total, 3),
        "wall_seconds": round(result.wall_total, 3),
        "rtf": round(result.wall_total / result.audio_total, 4) if result.audio_total else None,
        "normalization": "nfkc+casefold+strip-punctuation+collapse-space",
        "failures": result.failures,
        "per_clip": result.per_clip if args.per_clip else [],
    }


def gate(current: dict[str, Any], args: argparse.Namespace) -> list[str]:
    """Return the list of gate failures; empty means the run passes."""
    problems: list[str] = []
    if current["clips_scored"] == 0:
        problems.append("no clips were scored")
    if current["clips_failed"] and not args.allow_failures:
        problems.append(f"{current['clips_failed']} clip(s) failed to transcribe")
    if args.max_wer is not None and current["wer"] > args.max_wer:
        problems.append(f"WER {current['wer']:.4f} exceeds --max-wer {args.max_wer:.4f}")

    if args.baseline:
        try:
            base = json.loads(args.baseline.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"could not read baseline: {exc}")
            return problems
        if base.get("normalization") != current["normalization"]:
            # Comparing WER across different normalizers is meaningless.
            problems.append("baseline used a different normalization; re-record it")
            return problems
        delta = current["wer"] - float(base.get("wer", 0.0))
        current["baseline_wer"] = base.get("wer")
        current["wer_delta"] = round(delta, 6)
        if delta > args.max_wer_regression:
            problems.append(
                f"WER regressed by {delta:.4f} "
                f"({base.get('wer'):.4f} -> {current['wer']:.4f}), "
                f"limit {args.max_wer_regression:.4f}"
            )
        base_rtf, cur_rtf = base.get("rtf"), current.get("rtf")
        if base_rtf and cur_rtf:
            current["rtf_speedup"] = round(base_rtf / cur_rtf, 4)
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True, help="JSONL of audio_filepath + text")
    ap.add_argument("--audio-root", type=Path, help="root for relative audio paths (default: manifest dir)")
    ap.add_argument(
        "--url",
        default="http://127.0.0.1:8791/v1/audio/transcriptions",
        help="ASR endpoint to measure (default: local worker)",
    )
    ap.add_argument("--field", default="file", help="multipart field name (default: file)")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--out", type=Path, help="write the JSON report here")
    ap.add_argument("--per-clip", action="store_true", help="include per-clip rows in the report")
    ap.add_argument("--baseline", type=Path, help="previous report to compare against")
    ap.add_argument(
        "--max-wer-regression",
        type=float,
        default=0.0,
        help="allowed absolute WER increase vs baseline (default: 0, no regression)",
    )
    ap.add_argument("--max-wer", type=float, help="absolute WER ceiling")
    ap.add_argument("--allow-failures", action="store_true", help="do not fail on unusable clips")
    args = ap.parse_args()

    if not args.manifest.exists():
        print(f"manifest not found: {args.manifest}", file=sys.stderr)
        return 2

    result = run(args)
    current = report(result, args)
    problems = gate(current, args)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")

    rtf = current["rtf"]
    print(
        f"clips={current['clips_scored']} "
        f"WER={current['wer']:.4f} "
        f"p50={current['latency_seconds']['p50']}s "
        f"RTF={rtf if rtf is not None else 'n/a'}"
    )
    if "wer_delta" in current:
        direction = "worse" if current["wer_delta"] > 0 else "better"
        print(f"vs baseline {current['baseline_wer']:.4f}: {current['wer_delta']:+.4f} ({direction})")
    if "rtf_speedup" in current:
        print(f"RTF speedup vs baseline: {current['rtf_speedup']}x")

    for problem in problems:
        print(f"FAIL: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
