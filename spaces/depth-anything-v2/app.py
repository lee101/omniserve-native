import importlib.util
import pathlib


worker_path = pathlib.Path(__file__).resolve().with_name("depth_anything_worker.py")
spec = importlib.util.spec_from_file_location("depth_anything_worker", worker_path)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
app = worker.app


if __name__ == "__main__":
    import os
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")), workers=1)
