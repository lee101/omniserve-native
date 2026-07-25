#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
code_dir=$(cd "$project_dir/.." && pwd)
llama_dir=${LLAMA_DIR:-"$code_dir/llama.cpp"}
converter_python=${ONATIVE_CONVERTER_PYTHON:-"$code_dir/.venv/bin/python"}
output_dir=${ONATIVE_MODEL_DIR:-/nvme0n1-disk/models/omniserve-native}
modernbert_source=${ONATIVE_MODERNBERT_SOURCE:-"$code_dir/text-generator.io/models/ModernBERT-base"}
gemma_source=${ONATIVE_GEMMA_SOURCE:-"$code_dir/text-generator.io/llmtraining/models/gemma-roleplay-v2-merged"}
qwen_source=${ONATIVE_QWEN_SOURCE:-"$code_dir/text-generator.io/models/Qwen3.5-4B"}

mkdir -p "$output_dir"

convert_modernbert() {
    local output="$output_dir/modernbert-base-q8_0.gguf"
    if [[ -s "$output" ]]; then
        echo "present: $output"
        return
    fi
    local staging
    staging=$(mktemp -d)
    trap 'rm -rf -- "$staging"' RETURN
    "$converter_python" "$project_dir/tools/prepare_encoder.py" "$modernbert_source" "$staging"
    "$converter_python" "$llama_dir/convert_hf_to_gguf.py" "$staging" \
        --outfile "$output" --outtype q8_0 --model-name modernbert-base
    trap - RETURN
    rm -rf -- "$staging"
}

convert_gte_modernbert() {
    local output="$output_dir/gte-modernbert-base-q8_0.gguf"
    if [[ -s "$output" ]]; then
        echo "present: $output"
        return
    fi
    local source_dir=${ONATIVE_GTE_SOURCE:-}
    if [[ -z "$source_dir" ]]; then
        source_dir=$(HF_HOME="${HF_HOME:-/nvme0n1-disk/models/huggingface}" \
            "$converter_python" - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download("Alibaba-NLP/gte-modernbert-base",
                        allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model"]))
PY
        )
    fi
    # Already an encoder trunk (ModernBertModel), so no head stripping needed.
    "$converter_python" "$llama_dir/convert_hf_to_gguf.py" "$source_dir" \
        --outfile "$output" --outtype q8_0 --model-name gte-modernbert-base
    echo "serve with OMNISERVE_NATIVE_EMBEDDING_POOLING=cls (this checkpoint is CLS-pooled)"
}

convert_gemma() {
    local output="$output_dir/gemma-roleplay-v2-q8_0.gguf"
    if [[ -s "$output" ]]; then
        echo "present: $output"
        return
    fi
    local staging
    staging=$(mktemp -d)
    trap 'rm -rf -- "$staging"' RETURN
    "$converter_python" "$project_dir/tools/prepare_hf_compat.py" "$gemma_source" "$staging"
    "$converter_python" "$llama_dir/convert_hf_to_gguf.py" "$staging" \
        --outfile "$output" --outtype q8_0 --model-name gemma-roleplay-v2
    trap - RETURN
    rm -rf -- "$staging"
}

convert_qwen() {
    local output="$output_dir/qwen3.5-4b-text-q8_0.gguf"
    if [[ ! -s "$output" ]]; then
        "$converter_python" "$llama_dir/convert_hf_to_gguf.py" "$qwen_source" \
            --outfile "$output" --outtype q8_0 --model-name qwen3.5-4b --no-mtp
    else
        echo "present: $output"
    fi

    local projector="$output_dir/mmproj-qwen3.5-4b-f16.gguf"
    local legacy_projector="$output_dir/qwen3.5-4b-f16.gguf"
    if [[ ! -s "$projector" && -s "$legacy_projector" ]]; then
        mv -- "$legacy_projector" "$projector"
    fi
    if [[ ! -s "$projector" ]]; then
        "$converter_python" "$llama_dir/convert_hf_to_gguf.py" "$qwen_source" \
            --outfile "$projector" --outtype f16 \
            --model-name qwen3.5-4b --mmproj
        [[ -s "$projector" ]] || {
            echo "multimodal projector was not created at $projector" >&2
            return 1
        }
    else
        echo "present: $projector"
    fi
}

selection=${1:-all}
case "$selection" in
    modernbert) convert_modernbert ;;
    gte) convert_gte_modernbert ;;
    gemma) convert_gemma ;;
    qwen) convert_qwen ;;
    all)
        convert_modernbert
        convert_gte_modernbert
        convert_gemma
        convert_qwen
        ;;
    *)
        echo "usage: $0 [modernbert|gte|gemma|qwen|all]" >&2
        exit 2
        ;;
esac

find "$output_dir" -maxdepth 1 -type f -name '*.gguf' -printf '%f %s bytes\n' | sort
