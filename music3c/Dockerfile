# syntax=docker/dockerfile:1.7

FROM hongccc/sglang-omni@sha256:374d0b1c30b2bff685b1716fc64a02ad3b3d0a90fe2ce73ce9861a6992c28101

ARG SGLANG_OMNI_COMMIT=e0c98529e5730f60e19251025877387b9476c8d4

COPY music3/sglang-music3-a100.patch /tmp/sglang-music3-a100.patch
COPY music3c /opt/omniserve-music3c-src

RUN git clone --filter=blob:none https://github.com/sgl-project/sglang-omni.git /opt/sglang-omni \
    && git -C /opt/sglang-omni checkout "${SGLANG_OMNI_COMMIT}" \
    && git -C /opt/sglang-omni apply /tmp/sglang-music3-a100.patch \
    && python3 -m pip install --no-deps --no-build-isolation -e /opt/sglang-omni \
    && mkdir -p /opt/omniserve-music3 \
    && cc -std=c17 -O2 -Wall -Wextra -Wpedantic -Werror -D_POSIX_C_SOURCE=200809L \
       /opt/omniserve-music3c-src/main.c /opt/omniserve-music3c-src/music3.c -lm \
       -o /opt/omniserve-music3/music3c

ENV PYTHONPATH=/opt/sglang-omni \
    SGLANG_OMNI_AUTO_CLONE=0 \
    MUSIC3_MODEL_ID=MiniMaxAI/MiniMax-Music3 \
    MUSIC3_MODEL_DIR=/runpod-volume/models/minimax-music3 \
    HF_HOME=/runpod-volume/huggingface \
    TORCHINDUCTOR_CACHE_DIR=/runpod-volume/omniserve/music3/torchinductor \
    FLASHINFER_WORKSPACE_BASE=/runpod-volume/omniserve/music3/flashinfer \
    MUSIC3_PORT=8000 \
    MUSIC3_ACOUSTIC_DTYPE=bfloat16 \
    MUSIC3_MAX_RUNNING_REQUESTS=1 \
    MUSIC3_STARTUP_TIMEOUT_SECONDS=1800 \
    MUSIC3_REQUEST_TIMEOUT_SECONDS=1800

ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
CMD ["/opt/omniserve-music3/music3c"]
