# Local-first ASR and interruptible training

OmniServe keeps speech inference local when the machine has capacity and
replays the same request to a configured managed provider when it does not.
Fine-tuning is an exclusive background workload: it can use the GPU only after
interactive work drains, and it cooperatively checkpoints and exits as soon as
paid, subscription, or free serving work queues.

## Processes

| Process | Default port | Responsibility |
|---|---:|---|
| `omniserve-native` | 8791 | Authentication, weighted admission, exclusive background lease |
| `asr_router.py` | 9096 | Local health/VRAM decision and transient remote fallback |
| `asr_worker.py` | 9097 | Lazy Parakeet inference on CUDA, ROCm, or CPU |
| `asr_training_manager.py` | 9098 | Persistent jobs, checkpoint preemption, low-priority process launch |

`SIGSTOP` is deliberately not used because it leaves model tensors in VRAM.
When interactive demand appears, the manager sends `SIGUSR1`; the trainer saves
a Transformers checkpoint at the next step boundary and exits. A bounded grace
period prevents a broken trainer from blocking serving indefinitely.

## Install

```bash
python -m venv .venv
.venv/bin/pip install -r workers/requirements-asr.txt
sudo cp systemd/omniserve-asr-*.service \
  systemd/omniserve-training-* /etc/systemd/system/
```

Configure the native gateway:

```ini
Environment=OMNISERVE_NATIVE_STT_UPSTREAM=http://127.0.0.1:9096
Environment=OMNISERVE_NATIVE_TRAINING_UPSTREAM=http://127.0.0.1:9098
Environment=OMNISERVE_NATIVE_TRAINING_PERMITS=8
Environment=OMNISERVE_NATIVE_TRAINING_SWAP_EMBEDDED_MODELS=1
```

Then enable the workers and resume timer. Do not enable the timer until a
reviewed job is queued:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now omniserve-asr-worker omniserve-asr-router
sudo systemctl enable --now omniserve-training-manager
sudo systemctl enable --now omniserve-training-runner.timer
```

## Build an eligible private manifest

DictatorFlow writes sidecars only after an account explicitly opts in to
`public_model_weights`. Legacy recordings do not have these sidecars and are
not eligible.

```bash
./scripts/build_asr_manifest.py \
  --source /private/dictatorflow-audio-history/public-model-v1 \
  --output /private/training/manifests/batch-001.jsonl
```

The manifest contains private transcripts and absolute audio paths. Never
commit or upload it.

## Queue and run a job

```bash
curl -X POST http://127.0.0.1:8791/v1/training/jobs \
  -H 'Content-Type: application/json' \
  -d '{
    "manifest":"/private/training/manifests/batch-001.jsonl",
    "audio_root":"/private/dictatorflow-audio-history/public-model-v1",
    "output_dir":"/nvme0n1-disk/models/dictatorflow-training/run-001",
    "base_model":"nvidia/parakeet-ctc-0.6b",
    "max_steps":1000,
    "bf16":true
  }'
```

The timer calls `/v1/training/jobs/run`. That endpoint is forcibly assigned the
background tier regardless of caller headers. The gateway unloads its embedded
models after exclusive admission, the manager requires the configured free
VRAM floor, and the ASR worker is held/unloaded for the segment.

On a CUDA build, PyTorch reports `device=cuda`. ROCm builds intentionally use
the same PyTorch device API and report `runtime=rocm`. If neither backend is
available, local inference falls back to CPU; training should normally keep an
18 GiB free-VRAM floor and remain queued on CPU-only hosts.

## Evaluation and publication

A same-speaker holdout is useful for iteration but insufficient for a public
release. Before uploading weights:

1. Revalidate consent and remove revoked samples.
2. Run a speaker-disjoint private evaluation and a licensed public benchmark.
3. Compare WER against the unchanged base model using identical normalization.
4. Test memorization and inspect rare-name/proper-noun outputs.
5. Complete the model card from `docs/asr-model-card-template.md`.
6. Complete a release approval JSON and use `scripts/publish_asr_model.py`.

The publisher refuses corpus/audio files, requires explicit privacy, license,
consent, model-card, and speaker-disjoint-evaluation approvals, and rejects a
candidate whose declared WER is worse than its baseline.
