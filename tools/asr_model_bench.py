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
        --backend gpt-transcribe=openai:gpt-transcribe \\
        --out performance/asr-models.json

Backend specs:
    hf:<model-id>       transformers automatic-speech-recognition pipeline
    http:<url>          an OpenAI-style multipart endpoint (a served worker)
    gemini:<model>      Gemini generateContent with inline audio + a transcribe
                        prompt; needs GEMINI_API_KEY
    openai:<model>      OpenAI audio transcriptions; needs OPENAI_API_KEY

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
import uuid
import wave
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import asr_stats as st
import asr_textnorm as tn
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


def openai_backend(model: str) -> Callable[[Path], str]:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise SystemExit("openai backend needs OPENAI_API_KEY")
    endpoint = os.environ.get(
        "OPENAI_TRANSCRIBE_URL", "https://api.openai.com/v1/audio/transcriptions"
    ).strip()

    def run(clip: Path) -> str:
        boundary = uuid.uuid4().hex
        fields = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="model"\r\n\r\n{model}\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="response_format"\r\n\r\njson\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{clip.name}"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        ).encode()
        body = fields + clip.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(
            endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            return extract_text(json.loads(resp.read().decode("utf-8", "replace")))

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
    if kind == "openai":
        return openai_backend(rest)
    raise SystemExit(f"unknown backend spec {spec!r} (hf:, audiolm:, http:, gemini:, openai:)")


# ---- scoring ----------------------------------------------------------------


def score(name: str, run: Callable[[Path], str], corpus: list[tuple[Path, str]], verbose: bool) -> dict[str, Any]:
    total = Counts()
    lenient_total = Counts()
    clip_scores: list[st.ClipScore] = []
    hyp_tokens: dict[str, list[str]] = {}
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

        ref_toks, hyp_toks = tokens(reference), tokens(hypothesis)
        counts = align(ref_toks, hyp_toks)
        total.add(counts)
        clip_scores.append(st.ClipScore(counts.errors, counts.reference_words))
        hyp_tokens[clip.name] = hyp_toks
        lenient = align(tn.lenient_tokens(reference, tokens), tn.lenient_tokens(hypothesis, tokens))
        lenient_total.add(lenient)
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
        # Formatting-only normalization: same words, different spelling. Shown
        # next to the strict number so a gap between them is visible rather than
        # silently baked into the headline.
        "wer_lenient": round(lenient_total.wer(), 4),
        "per_clip": clips,
        "_clip_scores": clip_scores,
        "_hyp_tokens": hyp_tokens,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--audio-dir", type=Path)
    ap.add_argument("--backend", action="append", default=[], metavar="NAME=SPEC", required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--resamples", type=int, default=10000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-ensemble", action="store_true", help="skip the ROVER row")
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
        if spec == "openai:gpt-transcribe" and result.get("audio_seconds") is not None:
            result["price_per_audio_minute_usd"] = 0.0045
            result["estimated_api_cost_usd"] = round(result["audio_seconds"] / 60 * 0.0045, 8)
        results.append(result)
        if result.get("wer") is None:
            print(f"  {result.get('error', 'no result')}")
        else:
            print(
                f"  WER={result['wer']:.4f}  errors={result['errors']}/{result['reference_words']}"
                f"  RTF={result['rtf']}"
            )

    scored = [r for r in results if r.get("wer") is not None]

    # ROVER: vote a combined transcript out of every scored backend. On a corpus
    # where systems make *different* mistakes this lands below the best single
    # system; where they share a blind spot it cannot help, and saying so is the
    # point of measuring it rather than assuming.
    if len(scored) >= 3 and not args.no_ensemble:
        combined = Counts()
        combined_scores = []
        agreements = []
        for clip, reference in corpus:
            hyps = [r["_hyp_tokens"].get(clip.name) for r in scored]
            hyps = [h for h in hyps if h is not None]
            if len(hyps) < 3:
                continue
            voted = st.rover_combine(hyps)
            agreements.append(st.agreement_rate(hyps))
            c = align(tokens(reference), voted)
            combined.add(c)
            combined_scores.append(st.ClipScore(c.errors, c.reference_words))
        if combined_scores:
            results.append(
                {
                    "backend": f"rover({len(scored)} systems)",
                    "spec": "ensemble",
                    "wer": round(combined.wer(), 4),
                    "errors": combined.errors,
                    "reference_words": combined.reference_words,
                    "rtf": None,
                    "mean_agreement": round(sum(agreements) / len(agreements), 3),
                    "_clip_scores": combined_scores,
                    "_hyp_tokens": {},
                }
            )
            scored = [r for r in results if r.get("wer") is not None]

    scored.sort(key=lambda r: r["wer"])

    # Bootstrap CI per system. The interval is the honest version of the number.
    for r in scored:
        cs = r.get("_clip_scores") or []
        if cs:
            ci = st.bootstrap_wer_ci(cs, resamples=args.resamples, seed=args.seed)
            r["wer_ci_low"], r["wer_ci_high"] = round(ci.low, 4), round(ci.high, 4)

    print("\n" + "=" * 78)
    print(f"{'backend':<24}{'WER':>8}{'95% CI':>20}{'lenient':>10}{'errors':>8}{'RTF':>8}")
    print("-" * 78)
    for r in scored:
        ci = (
            f"[{r['wer_ci_low']:.3f}, {r['wer_ci_high']:.3f}]"
            if "wer_ci_low" in r
            else ""
        )
        lenient = f"{r['wer_lenient']:.4f}" if r.get("wer_lenient") is not None else "-"
        print(
            f"{r['backend']:<24}{r['wer']:>8.4f}{ci:>20}{lenient:>10}"
            f"{r['errors']:>8}{str(r['rtf']):>8}"
        )
    for r in results:
        if r.get("wer") is None:
            reason = r.get("error", "unavailable")
            print(f"{r['backend']:<22}{'n/a':>9}   {reason[:60]}")

    # Pairwise paired-bootstrap against the leader: which gaps are real?
    if len(scored) >= 2:
        best = scored[0]
        print("\nPaired bootstrap vs the leader (same resampled clips for both):")
        for other in scored[1:]:
            a, b = other.get("_clip_scores"), best.get("_clip_scores")
            if not a or not b or len(a) != len(b):
                continue
            cmp = st.paired_bootstrap(a, b, resamples=args.resamples, seed=args.seed)
            other["vs_leader"] = {
                "leader": best["backend"],
                "delta": round(cmp.delta, 4),
                "ci": [round(cmp.low, 4), round(cmp.high, 4)],
                "p_value": round(cmp.p_value, 4),
                "significant": cmp.significant,
            }
            print("  " + cmp.verdict(other["backend"], best["backend"], 0.05))

    for r in results:
        r.pop("_clip_scores", None)
        r.pop("_hyp_tokens", None)

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
