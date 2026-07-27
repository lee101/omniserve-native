#!/usr/bin/env python3
"""Score several ASR backends on one corpus with identical normalization.

wer_bench.py answers "did this change to the serving path move WER?".
This answers "which model should we serve at all?" -- it runs N backends over
the same clips and prints a comparison table, reusing wer_bench's normalizer and
corpus-level aggregation so the numbers are comparable to the gate.

    tools/asr_model_bench.py --corpus ground_truth.json --audio-dir clips \\
        --backend whisper-base=hf:openai/whisper-base.en \\
        --backend whisper-small=hf:openai/whisper-small.en \\
        --backend gemini=gemini:gemini-2.5-flash \\
        --out performance/asr-models.json

Backend specs:
    hf:<model-id>       transformers automatic-speech-recognition pipeline
    http:<url>          an OpenAI-style multipart endpoint (a served worker)
    gemini:<model>      Gemini generateContent with inline audio + a transcribe
                        prompt; needs GEMINI_API_KEY

A corpus is either JSONL ({"audio_filepath":..., "text":...}) or the flat
{"file.wav": "reference"} map DictatorFlow's e2e corpus uses.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.request
import wave
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

from wer_bench import Counts, align, audio_seconds, extract_text, normalize, post_audio, tokens


# ---- corpus -----------------------------------------------------------------


def load_corpus(path: Path, audio_dir: Path | None) -> list[tuple[Path, str]]:
    raw = path.read_text(encoding="utf-8")
    root = audio_dir or path.parent
    rows: list[tuple[Path, str]] = []

    stripped = raw.lstrip()
    if stripped.startswith("{") and "\n" not in stripped.split("}")[0][:200]:
        pass  # fall through to the generic attempts below

    # Flat {"file.wav": "reference"} map.
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and all(isinstance(v, str) for v in obj.values()):
            for name, text in obj.items():
                rows.append((root / name, text))
            return rows
    except json.JSONDecodeError:
        pass

    # JSONL manifest.
    for lineno, line in enumerate(raw.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{lineno}: {exc}")
        clip = Path(row["audio_filepath"])
        rows.append(((clip if clip.is_absolute() else root / clip), row["text"]))
    if not rows:
        raise SystemExit(f"{path}: no usable rows")
    return rows


# ---- backends ---------------------------------------------------------------


def hf_backend(model_id: str) -> Callable[[Path], str]:
    """A local transformers ASR pipeline. Loaded once, reused for every clip."""
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=device,
        dtype=torch.float16 if device == 0 else torch.float32,
    )

    def run(clip: Path) -> str:
        # No chunk_length_s: this mirrors what the worker actually calls, so the
        # measured number is the served number. Whisper handles >30s clips via
        # its own long-form path in current transformers.
        out = pipe(str(clip))
        return out.get("text", "") if isinstance(out, dict) else str(out)

    return run


def http_backend(url: str) -> Callable[[Path], str]:
    def run(clip: Path) -> str:
        return post_audio(url, clip, "file", timeout=300.0)

    return run


# Kept explicit and boring: the model is being asked for a transcript and
# nothing else. Any instruction to "clean up" or "summarise" would change what
# WER is measuring.
GEMINI_PROMPT = (
    "Transcribe this audio verbatim. Output only the transcript text, "
    "with no preamble, commentary, timestamps, or speaker labels."
)


def gemini_backend(model: str) -> Callable[[Path], str]:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise SystemExit("gemini backend needs GEMINI_API_KEY")
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def run(clip: Path) -> str:
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": GEMINI_PROMPT},
                        {
                            "inline_data": {
                                "mime_type": "audio/wav",
                                "data": base64.b64encode(clip.read_bytes()).decode(),
                            }
                        },
                    ]
                }
            ],
            # Deterministic: a sampled transcript makes the comparison noisy.
            "generationConfig": {"temperature": 0.0},
        }
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "x-goog-api-key": key},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read().decode("utf-8", "replace"))
        candidates = body.get("candidates") or []
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts") or []
        return " ".join(p.get("text", "") for p in parts).strip()

    return run


def audio_lm_backend(model_id: str) -> Callable[[Path], str]:
    """An audio-capable instruction model (Gemma 3n, Qwen2-Audio, Phi-4-mm).

    These are not ASR pipelines: the clip goes in as a modality alongside a text
    instruction and the transcript comes out of ordinary generation. Decoding is
    greedy so the comparison is not sampling noise.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    def run(clip: Path) -> str:
        import soundfile as sf

        audio, rate = sf.read(str(clip), dtype="float32")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": GEMINI_PROMPT},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            sampling_rate=rate,
        ).to(model.device)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        return processor.batch_decode(out[:, input_len:], skip_special_tokens=True)[0].strip()

    return run


def build_backend(spec: str) -> Callable[[Path], str]:
    kind, _, rest = spec.partition(":")
    if kind == "hf":
        return hf_backend(rest)
    if kind == "audiolm":
        return audio_lm_backend(rest)
    if kind == "http":
        return http_backend(rest)
    if kind == "gemini":
        return gemini_backend(rest)
    raise SystemExit(f"unknown backend spec {spec!r} (hf:, audiolm:, http:, gemini:)")


# ---- scoring ----------------------------------------------------------------


def score(name: str, run: Callable[[Path], str], corpus: list[tuple[Path, str]], verbose: bool) -> dict[str, Any]:
    total = Counts()
    clips: list[dict[str, Any]] = []
    wall = 0.0
    audio = 0.0
    failures = 0

    for clip, reference in corpus:
        if not clip.exists():
            failures += 1
            continue
        started = time.perf_counter()
        try:
            hypothesis = run(clip)
        except Exception as exc:  # one bad clip must not lose the whole backend
            failures += 1
            clips.append({"clip": clip.name, "error": str(exc)[:200]})
            continue
        elapsed = time.perf_counter() - started

        counts = align(tokens(reference), tokens(hypothesis))
        total.add(counts)
        wall += elapsed
        audio += audio_seconds(clip)
        clips.append(
            {
                "clip": clip.name,
                "wer": round(counts.wer(), 4),
                "reference": normalize(reference),
                "hypothesis": normalize(hypothesis),
                "seconds": round(elapsed, 3),
            }
        )
        if verbose:
            print(f"  {clip.name:<22} wer={counts.wer():.3f}  {normalize(hypothesis)[:70]}")

    if total.reference_words == 0:
        # Every clip failed. Counts.wer() would return 0.0 here, which reads as a
        # perfect score for a backend that never produced a transcript, so refuse
        # to report a number at all.
        return {
            "backend": name,
            "wer": None,
            "error": f"no clips scored ({failures} failed)",
            "reference_words": 0,
            "clips_failed": failures,
            "per_clip": clips,
        }
    return {
        "backend": name,
        "wer": round(total.wer(), 4),
        "reference_words": total.reference_words,
        "errors": total.errors,
        "substitutions": total.substitutions,
        "deletions": total.deletions,
        "insertions": total.insertions,
        "clips_failed": failures,
        "audio_seconds": round(audio, 2),
        "wall_seconds": round(wall, 2),
        "rtf": round(wall / audio, 3) if audio else None,
        "per_clip": clips,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--audio-dir", type=Path)
    ap.add_argument("--backend", action="append", default=[], metavar="NAME=SPEC", required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    corpus = load_corpus(args.corpus, args.audio_dir)
    words = sum(len(tokens(t)) for _, t in corpus)
    print(f"corpus: {len(corpus)} clips, {words} reference words")
    if words < 2000:
        # One word is worth 1/words of WER. On a corpus this small, differences
        # smaller than a few points are noise, not model quality.
        print(f"note: {words} words is small; treat differences under ~{100/words:.1f}% WER as noise")

    results = []
    for entry in args.backend:
        name, _, spec = entry.partition("=")
        if not spec:
            raise SystemExit(f"--backend needs NAME=SPEC, got {entry!r}")
        print(f"\n== {name} ({spec})")
        try:
            run = build_backend(spec)
        except SystemExit:
            raise
        except Exception as exc:
            print(f"  unavailable: {exc}")
            results.append({"backend": name, "spec": spec, "error": str(exc)[:300]})
            continue
        result = score(name, run, corpus, args.verbose)
        result["spec"] = spec
        results.append(result)
        if result.get("wer") is None:
            print(f"  {result.get('error', 'no result')}")
        else:
            print(
                f"  WER={result['wer']:.4f}  errors={result['errors']}/{result['reference_words']}"
                f"  RTF={result['rtf']}"
            )

    scored = [r for r in results if r.get("wer") is not None]
    scored.sort(key=lambda r: r["wer"])
    print("\n" + "=" * 62)
    print(f"{'backend':<22}{'WER':>9}{'errors':>10}{'RTF':>9}")
    print("-" * 62)
    for r in scored:
        print(f"{r['backend']:<22}{r['wer']:>9.4f}{r['errors']:>10}{str(r['rtf']):>9}")
    for r in results:
        if r.get("wer") is None:
            reason = r.get("error", "unavailable")
            print(f"{r['backend']:<22}{'n/a':>9}   {reason[:60]}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {"corpus": str(args.corpus), "clips": len(corpus), "reference_words": words, "results": results},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
