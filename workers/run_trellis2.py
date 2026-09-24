#!/usr/bin/env python3
"""One-shot official TRELLIS.2 inference with immediate VRAM release."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def override_model_names() -> None:
    from trellis2.modules import image_feature_extractor
    from trellis2.pipelines import rembg

    for owner, name, env in (
        (image_feature_extractor, "DinoV3FeatureExtractor", "OMNISERVE_3D_DINOV3_REPO"),
        (rembg, "BiRefNet", "OMNISERVE_3D_REMBG_REPO"),
    ):
        replacement = os.getenv(env, "").strip()
        if not replacement:
            continue
        cls = getattr(owner, name)
        original = cls.__init__

        def init(self, model_name=None, *args, _original=original, _replacement=replacement, **kwargs):
            _original(self, _replacement, *args, **kwargs)

        cls.__init__ = init


def export_one(pipeline, o_voxel, image, output, seed, pipeline_type, args) -> None:
    mesh = pipeline.run(
        image,
        seed=seed,
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
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    partial = f"{output}.partial.glb"
    glb.export(partial, extension_webp=True)
    os.replace(partial, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--image")
    parser.add_argument("--output")
    parser.add_argument("--batch", help="JSON list of {image, output, seed}; models load once")
    parser.add_argument("--resolution", type=int, default=512, choices=(512, 1024, 1536))
    parser.add_argument("--texture-size", type=int, default=1024, choices=(1024, 2048, 4096))
    parser.add_argument("--decimation-target", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.batch:
        jobs = json.loads(Path(args.batch).read_text())
    elif args.image and args.output:
        jobs = [{"image": args.image, "output": args.output, "seed": args.seed}]
    else:
        parser.error("--image and --output, or --batch, are required")

    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("ATTN_BACKEND", "xformers")

    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    import o_voxel

    override_model_names()

    started = time.monotonic()
    model_ref = os.getenv("OMNISERVE_3D_TRELLIS_MODEL", "microsoft/TRELLIS.2-4B")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_ref)
    pipeline.low_vram = True
    pipeline.cuda()
    pipeline_type = {
        512: "512",
        1024: "1024_cascade",
        1536: "1536_cascade",
    }[args.resolution]
    loaded = time.monotonic()
    for job in jobs:
        if Path(job["output"]).exists():
            continue
        minimum = int(os.getenv("OMNISERVE_3D_MIN_FREE_MIB", "0"))
        if minimum:
            import torch
            torch.cuda.empty_cache()
            while True:
                free = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", "0"], text=True).strip())
                print(json.dumps({"omniserve_3d_vram": {"free_mib": free, "minimum_mib": minimum, "output": job["output"]}}), flush=True)
                if free >= minimum:
                    break
                time.sleep(30)
        job_started = time.monotonic()
        try:
            export_one(pipeline, o_voxel, Image.open(job["image"]), job["output"], int(job.get("seed", args.seed)), pipeline_type, args)
        except Exception as error:
            print(json.dumps({"omniserve_3d_job": {"output": job["output"], "error": str(error)[-500:]}}), flush=True)
            continue
        print(json.dumps({"omniserve_3d_job": {"output": job["output"], "seconds": round(time.monotonic() - job_started, 1)}}), flush=True)
    sampled = loaded
    import torch

    print(json.dumps({
        "omniserve_3d_stats": {
            "load_s": round(loaded - started, 1),
            "sample_s": round(sampled - loaded, 1),
            "export_s": round(time.monotonic() - sampled, 1),
            "peak_allocated_mib": torch.cuda.max_memory_allocated() >> 20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() >> 20,
        }
    }), flush=True)


if __name__ == "__main__":
    main()
