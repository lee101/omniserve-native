#!/usr/bin/env python3
"""One-shot official TRELLIS.2 inference with immediate VRAM release."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resolution", type=int, default=512, choices=(512, 1024, 1536))
    parser.add_argument("--texture-size", type=int, default=1024, choices=(1024, 2048, 4096))
    parser.add_argument("--decimation-target", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("ATTN_BACKEND", "xformers")

    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    import o_voxel

    model_ref = os.getenv("OMNISERVE_3D_TRELLIS_MODEL", "microsoft/TRELLIS.2-4B")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_ref)
    pipeline.low_vram = True
    pipeline.cuda()
    pipeline_type = {
        512: "512",
        1024: "1024_cascade",
        1536: "1536_cascade",
    }[args.resolution]
    image = Image.open(args.image)
    mesh = pipeline.run(
        image,
        seed=args.seed,
        pipeline_type=pipeline_type,
        max_num_tokens=49152,
    )[0]
    mesh.simplify(16_777_216)
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=args.decimation_target,
        texture_size=args.texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=True,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    glb.export(args.output, extension_webp=True)


if __name__ == "__main__":
    main()
