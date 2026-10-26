FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install -r /tmp/requirements.txt

COPY omniserve/ /opt/omniserve/
ENV PYTHONPATH=/opt/omniserve \
    OMNISERVE_WORKLOAD_MANIFEST=/opt/omniserve/workloads/workloads.json \
    OMNISERVE_PRELOAD=anime-pose-control \
    ANIME_CONTROL_WARMUP_SIZES=1024x1024 \
    ANIMA_COMPILE=0 \
    ANIMA_COMPILE_ARTIFACT_DIR=/opt/compile-cache \
    HF_HOME=/opt/hf \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR /opt/omniserve
CMD ["python", "-u", "-m", "runtime.handler"]
