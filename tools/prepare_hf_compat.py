#!/usr/bin/env python3
"""Stage a Hugging Face checkpoint with converter-compatible metadata.

Large tensor files are symlinked, never copied. The current Gemma 4 training
checkpoint stores ``extra_special_tokens`` in an older list form; recent
Transformers expects a keyed object. For the text-only model the video marker
is already present in tokenizer.json, so dropping the legacy convenience field
does not alter token IDs or weights.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    for source_file in args.source.iterdir():
        destination_file = args.destination / source_file.name
        if source_file.name == "tokenizer_config.json":
            config = json.loads(source_file.read_text(encoding="utf-8"))
            if isinstance(config.get("extra_special_tokens"), list):
                config.pop("extra_special_tokens")
            destination_file.write_text(
                json.dumps(config, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        else:
            os.symlink(source_file.resolve(), destination_file)


if __name__ == "__main__":
    main()
