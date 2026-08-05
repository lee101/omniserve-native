# OmniServe Native

A small native C control plane for local model serving. It exposes a browser
model selector and admits a model only when its configured weight and runtime
budget fit the live NVIDIA GPU memory after retaining a 2 GiB safety reserve.

It binds only to `127.0.0.1:8080`. Selection is deliberately separate from
inference: an inference backend must consume the admitted `active_model` state.
This keeps a failed or overloaded backend from bypassing VRAM admission.

## Run

```bash
make test
make run
```

Open `http://127.0.0.1:8080`. The service calls `nvidia-smi` on every status
and selection request. If NVIDIA telemetry is missing or errors, all model
admissions fail closed rather than risking VRAM exhaustion.

## Model Catalog

`models/models.csv` contains `id,label,weights_mib,runtime_mib`. The runtime
budget must include KV cache, compute workspace, and backend overhead for the
intended context length. Models that do not fit remain visible in the selector
but cannot be admitted.
