# Third-party notices

OmniServe Native source is Apache-2.0. It integrates with optional projects and
models that retain their own licenses and are not relicensed by this repository.

Review the exact version and license before distributing a combined binary or
model artifact:

- llama.cpp and GGML
- stable-diffusion.cpp
- NVIDIA CUDA, NVML, PyTorch, NeMo, and Parakeet models
- Hugging Face Transformers, Datasets, Hub, and model repositories
- Microsoft TRELLIS.2 and its gated DINOv3 dependency
- BiRefNet
- FastAPI, Uvicorn, SoundFile, jiwer, and python-multipart

The ASR training template derives from `nvidia/parakeet-ctc-0.6b`, whose model
card declares CC BY 4.0. A released fine-tune must preserve attribution and
document modifications. Model weights, generated assets, and datasets are
separate works with separate terms.

This notice is informational and does not replace the license text shipped by
each dependency.
