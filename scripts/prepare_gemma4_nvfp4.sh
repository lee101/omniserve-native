#!/usr/bin/env bash
set -euo pipefail

# ModelOpt's HF PTQ example is maintained in the Model-Optimizer repository,
# not shipped in the Python wheel. This wrapper validates the large source
# checkpoint before starting calibration and keeps the NVFP4 output separate
# from the llama.cpp GGUF artifacts.
project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
code_dir=$(cd "$project_dir/.." && pwd)

source_dir=${ONATIVE_GEMMA4_SOURCE:-/nvme0n1-disk/models/omniserve-native/gemma4-meromero-v2-31b}
output_dir=${ONATIVE_MODEL_DIR:-/nvme0n1-disk/models/omniserve-native}
modelopt_root=${MODEL_OPT_ROOT:-"$code_dir/Model-Optimizer"}
ptq_script=${MODEL_OPT_HF_PTQ:-"$modelopt_root/examples/hf_ptq/hf_ptq.py"}
python_bin=${MODEL_OPT_PYTHON:-"$project_dir/.venv-trtllm/bin/python"}
qformat=${ONATIVE_GEMMA4_NVFP4_QFORMAT:-nvfp4}
kv_qformat=${ONATIVE_GEMMA4_NVFP4_KV_QFORMAT:-nvfp4}
calib_size=${ONATIVE_GEMMA4_CALIB_SIZE:-128}
calib_seq=${ONATIVE_GEMMA4_CALIB_SEQ:-512}
batch_size=${ONATIVE_GEMMA4_CALIB_BATCH_SIZE:-1}
gpu_mem_percentage=${ONATIVE_GEMMA4_GPU_MEM_PERCENTAGE:-0.2}
device=${ONATIVE_GEMMA4_DEVICE:-cuda}
reserve_gib=${ONATIVE_MODEL_SPACE_RESERVE_GIB:-8}

[[ -x "$python_bin" ]] || {
    echo "missing ModelOpt Python: $python_bin" >&2
    exit 1
}
awk -v value="$gpu_mem_percentage" 'BEGIN { exit !(value > 0 && value <= 1) }' || {
    echo "ONATIVE_GEMMA4_GPU_MEM_PERCENTAGE must be in (0, 1]: $gpu_mem_percentage" >&2
    exit 1
}
[[ -f "$ptq_script" ]] || {
    cat >&2 <<EOF
missing ModelOpt HF PTQ example: $ptq_script
Clone the example checkout first, for example:
  git clone --depth 1 --filter=blob:none --sparse \\
    https://github.com/NVIDIA/Model-Optimizer.git "$modelopt_root"
  git -C "$modelopt_root" sparse-checkout set examples/hf_ptq modelopt_recipes
EOF
    exit 1
}

[[ -s "$source_dir/model.safetensors.index.json" ]] || {
    echo "missing checkpoint index: $source_dir/model.safetensors.index.json" >&2
    exit 2
}
[[ -s "$source_dir/model-00001-of-00002.safetensors" \
   && -s "$source_dir/model-00002-of-00002.safetensors" ]] || {
    echo "checkpoint is incomplete; resume prepare_gemma4_model.sh first" >&2
    exit 2
}

mkdir -p "$output_dir"
weight_bytes=$(stat -c '%s' "$source_dir"/model-*.safetensors | awk '{ total += $1 } END { print total + 0 }')
available_bytes=$(df -P -B1 "$output_dir" | awk 'NR == 2 { print $4 }')
# low_memory_mode compresses during calibration, but leave room for the export
# and CUDA/runtime overhead. This is a disk guard, not a VRAM guarantee.
required_bytes=$((weight_bytes * 55 / 100 + reserve_gib * 1024 * 1024 * 1024))
if (( available_bytes < required_bytes )); then
    printf 'insufficient disk space: have %.1f GiB, need about %.1f GiB\n' \
        "$((available_bytes / 1024 / 1024 / 1024))" \
        "$((required_bytes / 1024 / 1024 / 1024))" >&2
    exit 3
fi

echo "ModelOpt version: $("$python_bin" -c 'import modelopt; print(getattr(modelopt, "__version__", "unknown"))')"
echo "source: $source_dir"
echo "export: $output_dir/G4-MeroMero-v2-31B-${qformat^^}"
echo "calibration: size=$calib_size seq=$calib_seq batch=$batch_size device=$device gpu_mem=${gpu_mem_percentage} qformat=$qformat kv=$kv_qformat"

exec "$python_bin" "$ptq_script" \
    --pyt_ckpt_path "$source_dir" \
    --qformat "$qformat" \
    --kv_cache_qformat "$kv_qformat" \
    --export_path "$output_dir/G4-MeroMero-v2-31B-${qformat^^}" \
    --device "$device" \
    --calib_size "$calib_size" \
    --calib_seq "$calib_seq" \
    --batch_size "$batch_size" \
    --gpu_max_mem_percentage "$gpu_mem_percentage" \
    --low_memory_mode \
    --use_seq_device_map \
    --trust_remote_code
