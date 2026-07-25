#!/usr/bin/env python3
"""Export only the encoder trunk from an *ForMaskedLM Hugging Face model.

llama.cpp embeds encoder models but intentionally does not use masked-language
modeling heads. Some checkpoints still store those head tensors and older
converter paths reject them instead of ignoring them. Saving through AutoModel
produces an architecture-correct, byte-for-byte-equivalent encoder checkpoint.
This is conversion tooling only; OmniServe's serving path remains C-only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoModel, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    model = AutoModel.from_pretrained(args.source, local_files_only=True)
    model.save_pretrained(args.destination, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    tokenizer.save_pretrained(args.destination)


if __name__ == "__main__":
    main()
