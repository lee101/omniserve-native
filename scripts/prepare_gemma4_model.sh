#!/usr/bin/env bash
set -euo pipefail

# Download a Gemma 4 checkpoint and make a llama.cpp GGUF quantization. This
# deliberately calculates capacity before starting a 60+ GB download: the
# source checkpoint, 16-bit GGUF, quantized GGUF, and safety margin coexist
# during conversion.

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
code_dir=$(cd "$project_dir/.." && pwd)

repo_id=${ONATIVE_GEMMA4_REPO:-zerofata/G4-MeroMero-v2-31B}
model_root=${ONATIVE_GEMMA4_SOURCE:-/nvme0n1-disk/models/omniserve-native/gemma4-meromero-v2-31b}
output_dir=${ONATIVE_MODEL_DIR:-/nvme0n1-disk/models/omniserve-native}
llama_dir=${LLAMA_DIR:-"$code_dir/llama.cpp"}
converter_python=${ONATIVE_CONVERTER_PYTHON:-"$project_dir/.venv/bin/python"}
hf_bin=${HF_BIN:-"$project_dir/.venv/bin/hf"}
quantizer=${ONATIVE_LLAMA_QUANTIZE:-"$llama_dir/build/bin/llama-quantize"}
quant=${ONATIVE_GEMMA4_QUANT:-IQ4_XS}
threads=${ONATIVE_QUANT_THREADS:-$(nproc)}
reserve_gib=${ONATIVE_MODEL_SPACE_RESERVE_GIB:-8}
keep_intermediate=${ONATIVE_KEEP_GEMMA4_INTERMEDIATE:-0}

case "${quant^^}" in
    IQ4_XS|IQ4_NL|Q4_K_S|Q4_K_M|Q4_0|Q4_1|Q5_K_S|Q5_K_M|Q6_K|Q8_0)
        quant=${quant^^}
        ;;
    NVFP4)
        cat >&2 <<'EOF'
NVFP4 is not a normal llama-quantize output type in this checkout.
Use an NVIDIA ModelOpt/TRT-LLM NVFP4 checkpoint for TensorRT-LLM, or choose
IQ4_XS/IQ4_NL here for a native llama.cpp GGUF.
EOF
        exit 2
        ;;
    *)
        echo "unsupported Gemma 4 quantization: $quant" >&2
        exit 2
        ;;
esac

for command in "$converter_python" "$hf_bin" "$quantizer"; do
    [[ -x "$command" ]] || { echo "missing executable: $command" >&2; exit 1; }
done
[[ -f "$llama_dir/convert_hf_to_gguf.py" ]] || {
    echo "missing converter: $llama_dir/convert_hf_to_gguf.py" >&2
    exit 1
}

# This host's deployment notes document standard resumable HTTP as faster and
# more stable than Xet for large Hugging Face LFS shards. Prefer hf_transfer
# when installed, and only use Xet's high-performance mode as the fallback.
if "$converter_python" -c 'import hf_transfer' >/dev/null 2>&1; then
    export HF_HUB_ENABLE_HF_TRANSFER=1
    export HF_HUB_DISABLE_XET=1
elif "$converter_python" -c 'import hf_xet' >/dev/null 2>&1; then
    export HF_XET_HIGH_PERFORMANCE=1
    export HF_XET_NUM_CONCURRENT_RANGE_GETS=${HF_XET_NUM_CONCURRENT_RANGE_GETS:-64}
fi

mkdir -p "$model_root" "$output_dir"

available_bytes() {
    df -P -B1 "$1" | awk 'NR == 2 { print $4 }'
}

remote_bytes=$(
    "$converter_python" - "$repo_id" <<'PY'
import sys
from huggingface_hub import HfApi

repo_id = sys.argv[1]
total = 0
for item in HfApi().list_repo_tree(repo_id, recursive=True):
    if getattr(item, "path", "").endswith(".safetensors"):
        total += int(getattr(item, "size", 0) or 0)
if total <= 0:
    raise SystemExit(f"no safetensor weights found in {repo_id}")
print(total)
PY
)

# The 16-bit GGUF is normally within a few percent of the safetensor payload.
# Use a conservative estimate and leave room for filesystem/cache overhead.
converted_estimate=$((remote_bytes * 110 / 100))
quant_estimate=$((remote_bytes * 32 / 100))
reserve_bytes=$((reserve_gib * 1024 * 1024 * 1024))
checkpoint_present=0
if [[ -s "$model_root/model.safetensors.index.json" \
      && -s "$model_root/model-00001-of-00002.safetensors" \
      && -s "$model_root/model-00002-of-00002.safetensors" ]]; then
    checkpoint_present=1
fi
# If the checkpoint is already on disk, do not count its bytes again: the
# remaining overlap is the temporary BF16 GGUF, quantized GGUF, and reserve.
if (( checkpoint_present )); then
    required_bytes=$((converted_estimate + quant_estimate + reserve_bytes))
else
    required_bytes=$((remote_bytes + converted_estimate + quant_estimate + reserve_bytes))
fi
free_bytes=$(available_bytes "$model_root")

if (( free_bytes < required_bytes )); then
    printf 'insufficient disk space on %s: have %.1f GiB, need about %.1f GiB\n' \
        "$model_root" "$((free_bytes / 1024 / 1024 / 1024))" \
        "$((required_bytes / 1024 / 1024 / 1024))" >&2
    exit 3
fi

model_name=$(basename "$repo_id" | tr '[:lower:]' '[:upper:]' | tr '/ ' '--')
bf16_path="$model_root/${model_name}-BF16.gguf"
bf16_tmp="${bf16_path}.partial"
output_path="$output_dir/${model_name}-${quant}.gguf"
tmp_output="${output_path}.partial"

if (( ! checkpoint_present )); then
    echo "downloading $repo_id into $model_root"
    "$hf_bin" download "$repo_id" --local-dir "$model_root" --max-workers "${ONATIVE_HF_WORKERS:-4}"
else
    echo "checkpoint present: $model_root"
fi

if [[ ! -s "$bf16_path" ]]; then
    rm -f -- "$bf16_tmp"
    echo "converting checkpoint to BF16 GGUF: $bf16_path"
    "$converter_python" "$llama_dir/convert_hf_to_gguf.py" "$model_root" \
        --outfile "$bf16_tmp" --outtype bf16 --model-name "$model_name"
    mv -- "$bf16_tmp" "$bf16_path"
else
    echo "16-bit GGUF present: $bf16_path"
fi

if [[ ! -s "$output_path" ]]; then
    rm -f -- "$tmp_output"
    echo "quantizing $quant: $output_path"
    "$quantizer" --leave-output-tensor "$bf16_path" "$tmp_output" "$quant" "$threads"
    mv -- "$tmp_output" "$output_path"
else
    echo "quantized GGUF present: $output_path"
fi

if [[ "$keep_intermediate" != 1 ]]; then
    rm -f -- "$bf16_path"
fi

printf 'ready: %s (%s bytes)\n' "$output_path" "$(stat -c '%s' "$output_path")"
printf 'disk free: %.1f GiB\n' "$(($(available_bytes "$output_dir") / 1024 / 1024 / 1024))"
