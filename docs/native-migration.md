# Native migration and release gate

This document defines what “native” means for OmniServe, records the remaining
Python boundaries, and makes the open-source release gate explicit. It is a
shipping checklist, not a claim that model-specific runtimes can all be replaced
by portable C.

## Current boundary

The request data plane is native C today: HTTP parsing, authentication, JSON
tokenization, admission, VRAM leases, request routing, pooled proxying, metrics,
access logging, embedded llama.cpp/stable-diffusion.cpp inference, image encoding,
and speculative decode. CUDA is used where the algorithm is device-resident.

The August 2026 ASR-router cutover removes the Python HTTP relay from every
transcription request. `omniserve-asr-router` now owns local-worker health,
gateway VRAM checks, transient-only fallback, byte-preserving response relay,
and backend-attempt headers. The model worker remains separate because it owns a
PyTorch/Transformers checkpoint, not because the gateway needs Python.

| Boundary | Current implementation | Native target | Release condition |
| --- | --- | --- | --- |
| Gateway and scheduler | C | complete | portable, sanitizer, and routing tests pass |
| LLM and embeddings | C over llama.cpp | complete | CPU and requested GPU placement both fail honestly |
| Diffusion | C over stable-diffusion.cpp | active | image parity gate passes for each promoted checkpoint |
| Matte colour solve/composite | C + CUDA | active | CPU/CUDA fixture parity and device-pointer benchmark pass |
| ASR capacity/fallback router | C | complete, canary pending | native integration suite and production canary pass |
| ASR inference | Python model worker | C/C++ backend when model support is equivalent | WER, timestamps, formats, and GPU placement match |
| BiRefNet orchestration | Python model worker; C/CUDA postprocess | isolate model calls, move cache/job plumbing native | alpha/edge quality and image parity gate pass |
| Object storage | Python boto3 helper | native content-addressed local store first; remote storage through a stable service or audited SigV4 client | cache-key parity and conditional-put tests pass |
| 3D orchestration | Python process worker | keep model runtime external; move validation, job state, and admission callbacks native | TRELLIS/Pixal contracts and cancellation survive restart |
| ASR training | Python background control plane | no hot-path C requirement | stays isolated, background-only, checkpointable, and optional |

Training code and model-vendor launchers are not automatically migration
failures. They are off the request hot path and often bind to Python-only model
libraries. The useful boundary is a small, versioned HTTP contract behind native
admission. Porting control logic that executes for every request has priority over
rewriting an offline trainer in C.

## Ordered migration

1. Canary `omniserve-asr-router`, compare backend choice and response bytes with
   the Python reference, then remove the Python router from packaged services.
2. Finish the embedded Z-Image parity matrix: checkpoint/artifact combinations,
   guidance mapping, LoRA validation, WebP/PNG envelopes, VAE tiling, latent replay,
   peak VRAM, denoiser-only latency, and end-to-end latency.
3. Split BiRefNet into a minimal tensor inference worker and native job/cache
   control. The foreground/background solve and compositing already stay on the
   CUDA device.
4. Port the content-addressed local object cache. Treat S3 signing as a separate
   dependency decision; do not grow an unaudited TLS and SigV4 stack inside the
   inference gateway.
5. Move 3D request validation, durable job state, and GPU hold callbacks native.
   Keep TRELLIS/Pixal inference as an external runtime until a supported native
   backend exists.

Every cutover needs contract tests against both implementations, a quality gate
appropriate to the modality, peak-memory measurement, and rollback through one
upstream environment variable. “It compiles” is not a cutover criterion.

## Open-source release gate

The repository already has Apache-2.0 source licensing, contribution and security
policies, a code of conduct, third-party notices, portable CPU builds, sanitizer
CI, and no tracked model weights. The checked local optional runtimes use MIT
licenses for llama.cpp, stable-diffusion.cpp, and TRELLIS.2; they remain separate
projects and their exact revisions must be recorded for a binary release.

Before a public tag:

- run `sh scripts/release_audit.sh` and the full `ci` and `sanitize` presets;
- record exact external-runtime commits and licenses in the release notes;
- publish a model manifest with repository, revision, weight license, intended
  use, and redistribution status for every example configuration;
- keep weights, credentials, production logs, evaluation source images, cache
  directories, virtual environments, and built binaries out of Git;
- verify fixture provenance and that generated/evaluation images are either
  redistributable or omitted from the source release;
- build from a clean clone using only documented dependencies;
- generate an SBOM for the shipped binary/container and retain its compiler and
  CUDA versions;
- replace host-specific systemd paths in release packaging with configured
  install prefixes; and
- run the API contract and quality suites against the exact release artifact.

The automated audit is deliberately narrow: it catches common secret formats,
tracked model/build artifacts, unexpectedly large tracked files, and missing
policy documents. It cannot determine model-license compatibility or establish
ownership of training/evaluation data; those remain explicit human release
checks.
