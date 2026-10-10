FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY depth_anything_worker.py .
ENV DEPTH_HOST=0.0.0.0 DEPTH_PORT=7860 DEPTH_TORCH_COMPILE=1
CMD ["python", "app.py"]
