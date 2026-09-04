#!/usr/bin/env python3
import argparse
import pathlib
import tempfile

from huggingface_hub import HfApi


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPACE = ROOT / "spaces" / "depth-anything-v2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        target = pathlib.Path(directory)
        for source in SPACE.iterdir():
            if source.is_file():
                (target / source.name).write_bytes(source.read_bytes())
        (target / "depth_anything_worker.py").write_bytes((ROOT / "workers" / "depth_anything_worker.py").read_bytes())
        api = HfApi()
        api.create_repo(args.repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
        api.upload_folder(repo_id=args.repo_id, repo_type="space", folder_path=target)


if __name__ == "__main__":
    main()
