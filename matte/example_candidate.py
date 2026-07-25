#!/usr/bin/env python3
"""Worked example of the --candidate-cmd contract.

    <prog> <image.npy> <alpha.npy> <out.npy> [fg|bg]

Reads the fixture inputs, runs the pure-Python transcription of the algorithm,
writes the requested output as .npy. A C/CUDA port should expose the same CLI
shape so it can be dropped straight into eval_foreground.py:

    python eval_foreground.py --case C \
        --candidate-cmd "./build/matte_ml {image} {alpha} {out} fg"
"""

import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from ref_python import estimate_fb_ml_py  # noqa: E402


def main():
    if len(sys.argv) < 4:
        print(__doc__, file=sys.stderr)
        return 2
    image_path, alpha_path, out_path = sys.argv[1:4]
    target = sys.argv[4] if len(sys.argv) > 4 else "fg"

    F, B = estimate_fb_ml_py(np.load(image_path), np.load(alpha_path))
    np.save(out_path, F if target == "fg" else B)
    return 0


if __name__ == "__main__":
    sys.exit(main())
