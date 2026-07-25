#!/usr/bin/env python3
"""Build a private trainer manifest from DictatorFlow consent sidecars."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def duration_seconds(path: Path) -> float:
    if path.suffix.lower() == ".wav":
        import wave

        with wave.open(str(path), "rb") as handle:
            return handle.getnframes() / float(handle.getframerate())
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def capture_split(speaker: str, capture: str) -> str:
    # Capture-level holdout supports a single owner's adaptation run. Public
    # release still requires a separate speaker-disjoint benchmark.
    bucket = int(hashlib.sha256(f"{speaker}:{capture}".encode()).hexdigest()[:8], 16) % 10
    return "validation" if bucket == 0 else ("test" if bucket == 1 else "train")


def eligible(meta: dict[str, Any]) -> bool:
    return (
        meta.get("consent_scope") == "public_model_weights"
        and int(meta.get("consent_version", 0)) >= 1
        and meta.get("speaker_rights_confirmed") is True
        and not meta.get("revoked", False)
        and not meta.get("consent_revoked_at")
    )


def build(source: Path, output: Path) -> dict[str, Any]:
    rows = []
    skipped = 0
    for meta_path in sorted(source.glob("**/*.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not eligible(meta):
            skipped += 1
            continue
        audio = (meta_path.parent / meta["audio_file"]).resolve()
        transcript = (meta_path.parent / meta["transcript_file"]).resolve()
        if not audio.is_file() or not transcript.is_file():
            skipped += 1
            continue
        text = transcript.read_text(encoding="utf-8").strip()
        if not text:
            skipped += 1
            continue
        rows.append(
            {
                "audio_filepath": str(audio),
                "text": text,
                "duration": round(duration_seconds(audio), 4),
                "speaker_id": meta["speaker_id"],
                "split": capture_split(meta["speaker_id"], meta_path.stem),
                "consent_scope": meta["consent_scope"],
                "consent_version": meta["consent_version"],
                "consented_at": meta["consented_at"],
                "speaker_rights_confirmed": True,
            }
        )
    # Tiny corpora can hash to one bucket. Move deterministic tail captures so
    # the trainer can run, while recording that this is not release-grade eval.
    if len(rows) >= 2 and not any(row["split"] == "validation" for row in rows):
        rows[-1]["split"] = "validation"
    if rows and not any(row["split"] == "train" for row in rows):
        rows[0]["split"] = "train"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return {
        "eligible_rows": len(rows),
        "skipped_rows": skipped,
        "duration_seconds": round(sum(row["duration"] for row in rows), 3),
        "speakers": len({row["speaker_id"] for row in rows}),
        "speaker_disjoint_evaluation": False,
        "contains_transcripts": True,
        "publish_manifest_or_audio": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = build(args.source.resolve(), args.output.resolve())
    audit_path = args.output.with_suffix(args.output.suffix + ".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, separators=(",", ":")))


if __name__ == "__main__":
    main()
