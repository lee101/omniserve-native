#!/usr/bin/env python3
"""Consent-gated, resumable Parakeet CTC fine-tuning.

The input is a private JSONL manifest. Every row must carry explicit consent
for public model weights; legacy recordings and generic product-improvement
consent are intentionally rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import signal
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_SCOPE = "public_model_weights"
MIN_CONSENT_VERSION = 1
PREEMPT_REQUESTED = False


def request_preempt(_signum: int, _frame: Any) -> None:
    global PREEMPT_REQUESTED
    PREEMPT_REQUESTED = True


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("consented_at must include a timezone")
    return parsed.astimezone(timezone.utc)


def validate_manifest(path: Path, audio_root: Path) -> dict[str, Any]:
    root = audio_root.resolve()
    rows = 0
    duration_s = 0.0
    speakers: set[str] = set()
    splits: dict[str, int] = {}
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            digest.update(raw)
            if not raw.strip():
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(f"line {line_number}: invalid JSON: {error}") from error
            required = {
                "audio_filepath",
                "text",
                "duration",
                "speaker_id",
                "split",
                "consent_scope",
                "consent_version",
                "consented_at",
            }
            missing = sorted(required - item.keys())
            if missing:
                raise ValueError(f"line {line_number}: missing {', '.join(missing)}")
            audio_path = Path(item["audio_filepath"])
            if not audio_path.is_absolute():
                audio_path = root / audio_path
            resolved = audio_path.resolve()
            if resolved != root and root not in resolved.parents:
                raise ValueError(f"line {line_number}: audio path escapes audio root")
            if not resolved.is_file():
                raise ValueError(f"line {line_number}: audio file does not exist")
            if item["consent_scope"] != REQUIRED_SCOPE:
                raise ValueError(f"line {line_number}: public-weight consent is absent")
            if int(item["consent_version"]) < MIN_CONSENT_VERSION:
                raise ValueError(f"line {line_number}: unsupported consent version")
            parse_time(str(item["consented_at"]))
            if item.get("consent_revoked_at") or item.get("revoked", False):
                raise ValueError(f"line {line_number}: consent was revoked")
            if item.get("speaker_rights_confirmed") is not True:
                raise ValueError(f"line {line_number}: speaker rights are not confirmed")
            text = str(item["text"]).strip()
            if not text:
                raise ValueError(f"line {line_number}: empty transcript")
            duration = float(item["duration"])
            if not 0.1 <= duration <= 1800:
                raise ValueError(f"line {line_number}: invalid duration")
            split = str(item["split"])
            if split not in {"train", "validation", "test"}:
                raise ValueError(f"line {line_number}: invalid split")
            rows += 1
            duration_s += duration
            speakers.add(str(item["speaker_id"]))
            splits[split] = splits.get(split, 0) + 1
    if rows == 0:
        raise ValueError("manifest has no eligible recordings")
    if not splits.get("train") or not splits.get("validation"):
        raise ValueError("manifest needs non-empty train and validation splits")
    return {
        "rows": rows,
        "duration_seconds": round(duration_s, 3),
        "speakers": len(speakers),
        "splits": splits,
        "sha256": digest.hexdigest(),
        "consent_scope": REQUIRED_SCOPE,
    }


@dataclass
class CTCDataCollator:
    processor: Any
    input_name: str

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        input_features = [{self.input_name: row[self.input_name]} for row in features]
        batch = self.processor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": row["labels"]} for row in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


def training_main(args: argparse.Namespace) -> None:
    # Heavy imports happen only after the cheap consent audit passes.
    audit = validate_manifest(args.manifest, args.audio_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "corpus-audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )

    signal.signal(signal.SIGUSR1, request_preempt)
    import numpy as np
    from datasets import Audio, load_dataset
    from jiwer import wer as word_error_rate
    from transformers import (
        AutoModelForCTC,
        AutoProcessor,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    processor = AutoProcessor.from_pretrained(args.base_model)
    model = AutoModelForCTC.from_pretrained(args.base_model)
    if args.freeze_encoder:
        freezer = getattr(model, "freeze_feature_encoder", None)
        if callable(freezer):
            freezer()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    dataset = load_dataset("json", data_files=str(args.manifest), split="train")
    dataset = dataset.cast_column("audio_filepath", Audio(sampling_rate=16000))
    input_name = processor.model_input_names[0]

    def prepare(row: dict[str, Any]) -> dict[str, Any]:
        audio = row["audio_filepath"]
        encoded = processor(
            audio=audio["array"],
            text=row["text"],
            sampling_rate=audio["sampling_rate"],
        )
        return {
            input_name: encoded[input_name][0],
            "labels": encoded["labels"],
        }

    prepared = dataset.map(
        prepare,
        remove_columns=dataset.column_names,
        num_proc=args.preprocessing_workers,
    )
    source = load_dataset("json", data_files=str(args.manifest), split="train")
    train_indices = [i for i, split in enumerate(source["split"]) if split == "train"]
    eval_indices = [i for i, split in enumerate(source["split"]) if split == "validation"]
    train_dataset = prepared.select(train_indices)
    eval_dataset = prepared.select(eval_indices)
    def compute_metrics(prediction: Any) -> dict[str, float]:
        logits = prediction.predictions
        predicted_ids = np.argmax(logits, axis=-1)
        labels = prediction.label_ids.copy()
        labels[labels == -100] = processor.tokenizer.pad_token_id
        predicted = processor.batch_decode(predicted_ids)
        references = processor.batch_decode(labels, group_tokens=False)
        return {"wer": float(word_error_rate(references, predicted))}

    class PreemptCallback(TrainerCallback):
        def on_step_end(self, _args: Any, _state: Any, control: Any, **_kwargs: Any) -> Any:
            if PREEMPT_REQUESTED:
                control.should_save = True
                control.should_training_stop = True
            return control

    kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "max_steps": args.max_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "logging_steps": args.logging_steps,
        "save_total_limit": 3,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "load_best_model_at_end": True,
        "metric_for_best_model": "wer",
        "greater_is_better": False,
        "report_to": [],
        "remove_unused_columns": False,
    }
    parameter = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments).parameters else "evaluation_strategy"
    kwargs[parameter] = "steps"
    training_args = TrainingArguments(**kwargs)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CTCDataCollator(processor, input_name),
        compute_metrics=compute_metrics,
        processing_class=processor,
        callbacks=[PreemptCallback()],
    )
    checkpoint = args.resume_from_checkpoint
    if checkpoint == "auto":
        checkpoints = sorted(args.output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        checkpoint = str(checkpoints[-1]) if checkpoints else None
    trainer.train(resume_from_checkpoint=checkpoint or None)
    metrics = trainer.evaluate()
    trainer.save_model(str(args.output_dir / "final"))
    processor.save_pretrained(str(args.output_dir / "final"))
    (args.output_dir / "eval-results.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--audio-root", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--base-model", default="nvidia/parakeet-ctc-0.6b")
    result.add_argument("--resume-from-checkpoint", default="auto")
    result.add_argument("--max-steps", type=int, default=1000)
    result.add_argument("--save-steps", type=int, default=25)
    result.add_argument("--eval-steps", type=int, default=25)
    result.add_argument("--logging-steps", type=int, default=5)
    result.add_argument("--batch-size", type=int, default=1)
    result.add_argument("--eval-batch-size", type=int, default=1)
    result.add_argument("--gradient-accumulation-steps", type=int, default=16)
    result.add_argument("--learning-rate", type=float, default=1e-5)
    result.add_argument("--preprocessing-workers", type=int, default=1)
    result.add_argument("--freeze-encoder", action=argparse.BooleanOptionalAction, default=False)
    result.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    precision = result.add_mutually_exclusive_group()
    precision.add_argument("--fp16", action="store_true")
    precision.add_argument("--bf16", action="store_true")
    return result


if __name__ == "__main__":
    training_main(parser().parse_args())
