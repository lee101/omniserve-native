#!/usr/bin/env bash
set -euo pipefail

# Isolated TRELLIS.2 runtime for RTX 5090 / Blackwell. This intentionally uses
# xFormers instead of the repository's older pinned FlashAttention build.
repo="${OMNISERVE_3D_TRELLIS_REPO:-/nvme0n1-disk/code/TRELLIS.2}"
venv="${OMNISERVE_3D_VENV:-$repo/.venv}"
extension_root="${OMNISERVE_3D_EXTENSION_ROOT:-/nvme0n1-disk/code/trellis2-extensions}"
cuda_root="${CUDA_HOME:-/usr/local/cuda-12.9}"

test -d "$repo/.git"
test -x "$(command -v uv)"
test -x "$cuda_root/bin/nvcc"

export CUDA_HOME="$cuda_root"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export MAX_JOBS="${MAX_JOBS:-4}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-4}"

if [ ! -x "$venv/bin/python" ]; then
  uv venv --python 3.11 "$venv"
fi
uv pip install --python "$venv/bin/python" \
  torch==2.11.0 torchvision==0.26.0 xformers==0.0.35 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$venv/bin/python" \
  setuptools wheel ninja packaging \
  imageio imageio-ffmpeg tqdm easydict opencv-python-headless \
  transformers gradio==6.0.1 tensorboard pandas lpips zstandard \
  pillow kornia timm trimesh pygltflib plyfile \
  git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

mkdir -p "$extension_root"
if [ ! -d "$extension_root/nvdiffrast/.git" ]; then
  git clone --branch v0.4.0 https://github.com/NVlabs/nvdiffrast.git "$extension_root/nvdiffrast"
fi
if [ ! -d "$extension_root/nvdiffrec/.git" ]; then
  git clone --branch renderutils https://github.com/JeffreyXiang/nvdiffrec.git "$extension_root/nvdiffrec"
fi
if [ ! -d "$extension_root/CuMesh/.git" ]; then
  git clone --recursive https://github.com/JeffreyXiang/CuMesh.git "$extension_root/CuMesh"
fi
if [ ! -d "$extension_root/FlexGEMM/.git" ]; then
  git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git "$extension_root/FlexGEMM"
fi

uv pip install --python "$venv/bin/python" --no-build-isolation "$extension_root/nvdiffrast"
uv pip install --python "$venv/bin/python" --no-build-isolation "$extension_root/nvdiffrec"
uv pip install --python "$venv/bin/python" --no-build-isolation "$extension_root/CuMesh"
uv pip install --python "$venv/bin/python" --no-build-isolation "$extension_root/FlexGEMM"
# o-voxel declares Git URLs for CuMesh and FlexGEMM. They are already built
# above from pinned local checkouts, so install o-voxel without re-resolving
# those dependencies (which would both duplicate builds and conflict in uv).
uv pip install --python "$venv/bin/python" --no-build-isolation --no-deps \
  "$repo/o-voxel"

ATTN_BACKEND=xformers "$venv/bin/python" - <<'PY'
import torch
import xformers
import cumesh
import flex_gemm
import nvdiffrast.torch
import o_voxel
from trellis2.pipelines import Trellis2ImageTo3DPipeline
print({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "architectures": torch.cuda.get_arch_list(),
    "attention": "xformers",
    "imports": "ok",
})
PY
