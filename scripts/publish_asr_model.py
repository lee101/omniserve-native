#!/usr/bin/env python3
"""Publish reviewed model artifacts, never corpus files."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def validate_release(artifact: Path, approval: Path) -> dict:
    decision = json.loads(approval.read_text(encoding="utf-8"))
    required_true = (
        "privacy_review_approved",
        "license_review_approved",
        "consent_revalidated",
        "speaker_disjoint_eval_completed",
        "model_card_reviewed",
    )
    missing = [field for field in required_true if decision.get(field) is not True]
    if missing:
        raise ValueError(f"release approvals missing: {', '.join(missing)}")
    wer = float(decision.get("wer", 1))
    baseline_wer = float(decision.get("baseline_wer", 0))
    if wer <= 0 or baseline_wer <= 0 or wer > baseline_wer:
        raise ValueError("candidate WER must be positive and no worse than the declared baseline")
    reviewed_at = datetime.fromisoformat(str(decision["reviewed_at"]).replace("Z", "+00:00"))
    if reviewed_at.tzinfo is None or reviewed_at > datetime.now(timezone.utc):
        raise ValueError("reviewed_at is invalid")
    forbidden = {".wav", ".flac", ".opus", ".mp3", ".jsonl"}
    leaked = [path for path in artifact.rglob("*") if path.suffix.lower() in forbidden]
    if leaked:
        raise ValueError("artifact directory contains corpus/audio files")
    if not (artifact / "README.md").is_file():
        raise ValueError("artifact README.md model card is missing")
    if not any((artifact / name).exists() for name in ("model.safetensors", "pytorch_model.bin")):
        raise ValueError("model weights are missing")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument(
        "--confirm-public-release",
        required=True,
        choices=["I_HAVE_REVIEWED_PRIVACY_LICENSE_AND_WER"],
    )
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    validate_release(artifact, args.approval.resolve())
    from huggingface_hub import HfApi

    HfApi().create_repo(args.repo_id, repo_type="model", exist_ok=True)
    HfApi().upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=artifact,
        commit_message="Release reviewed DictatorFlow ASR variant",
        ignore_patterns=["*.wav", "*.flac", "*.opus", "*.mp3", "*.jsonl", "corpus-*"],
    )
    print(f"published https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
