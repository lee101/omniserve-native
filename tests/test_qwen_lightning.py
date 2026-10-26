"""Lightning schedule contracts for the Python worker and native C lane."""

import base64
import io
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workloads import qwen_image as q
from tools.qwen_lightning_bench import request


def expected_sigmas(width, height):
    mu = 0.5 + ((width // 16) * (height // 16) - 256) * 0.4 / 7936
    return [math.exp(mu) / (math.exp(mu) + (1 / t - 1)) for t in (1, .75, .5, .25)] + [0]


def source_png():
    from PIL import Image
    data = io.BytesIO()
    Image.new("RGB", (64, 64), "red").save(data, "PNG")
    return base64.b64encode(data.getvalue()).decode()


class Worker(unittest.TestCase):
    def setUp(self):
        def init(pointer):
            pointer._obj.sample_params.sample_steps = 20
        self.patches = [
            patch.object(q, "_lib_instance", return_value=types.SimpleNamespace(
                lib=types.SimpleNamespace(sd_img_gen_params_init=init))),
            patch.object(q, "LIGHTNING", True), patch.object(q, "TURBO", True),
            patch.object(q, "TURBO_NODES", [1, .75, .5, .25]),
            patch.object(q, "SAMPLE_METHOD", ""),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def test_generation_and_edit_match_training_schedule(self):
        for size in (512, 1024, 1536):
            for edit in (False, True):
                with self.subTest(size=size, edit=edit):
                    values = {"prompt": "adult portrait", "width": size, "height": size,
                              "steps": 40, "guidance_scale": 6}
                    if edit:
                        values["image_base64"] = source_png()
                    params, keep = q.build_params(values, {"turbo_lora": Path("/lightning.safetensors")})
                    self.assertTrue(keep)
                    self.assertEqual(params.sample_params.sample_steps, 4)
                    self.assertEqual(params.sample_params.sample_method, 0)
                    self.assertEqual(params.sample_params.guidance.txt_cfg, 1)
                    self.assertEqual(params.cache.mode, 0)
                    self.assertEqual(params.ref_images_count, int(edit))
                    self.assertEqual(params.lora_count, 1)
                    for i, expected in enumerate(expected_sigmas(size, size)):
                        self.assertAlmostEqual(params.sample_params.custom_sigmas[i], expected, places=6)

    def test_explicit_base_opt_out(self):
        params, keep = q.build_params({"prompt": "x", "steps": 40, "turbo": False},
                                     {"turbo_lora": Path("/lightning.safetensors")})
        self.assertEqual(params.sample_params.sample_steps, 40)
        self.assertEqual(params.lora_count, 0)
        self.assertEqual(params.sample_params.custom_sigmas_count, 0)

    def test_viggle_edits_remain_base(self):
        with patch.object(q, "LIGHTNING", False):
            params, keep = q.build_params({"prompt": "x", "image_base64": source_png()},
                                         {"turbo_lora": Path("/viggle.safetensors")})
        self.assertEqual(params.lora_count, 0)
        self.assertEqual(params.sample_params.custom_sigmas_count, 0)

    def test_missing_lightning_weights_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(q, "_state", {}), \
                patch.object(q, "_writable_root", return_value=Path(directory)), \
                patch.object(q, "_download", return_value=Path(directory) / "model"), \
                patch.object(q, "TURBO_LORA_REPO", "lightx2v/Qwen-Image-2.1-Lightning"), \
                patch.object(q, "_download_url", side_effect=OSError("offline")):
            with self.assertRaisesRegex(RuntimeError, "refusing.*fallback"):
                q.ensure_models()

    def test_strip_vae_dimensions(self):
        with patch.object(q, "VAE_TILING", True), patch.object(q, "VAE_TILE_X", 64), \
                patch.object(q, "VAE_TILE_Y", 32):
            params, keep = q.build_params({"prompt": "x"}, {"turbo_lora": Path("/lightning.safetensors")})
        self.assertEqual(params.vae_tiling_params.tile_size_x, 64)
        self.assertEqual(params.vae_tiling_params.tile_size_y, 32)


@unittest.skipUnless(os.getenv("OMNISERVE_NATIVE_BIN") and os.getenv("OMNISERVE_SD_STUB"),
                     "set native binary and SD stub paths to test C integration")
class Native(unittest.TestCase):
    def test_native_generation_edit_and_opt_out(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = root / "params.jsonl"
            lora = root / "lightning.safetensors"
            lora.touch()
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            env = {k: v for k, v in os.environ.items() if not k.startswith("OMNISERVE_")}
            env.update({
                "OMNISERVE_NATIVE_SD_LIB": os.environ["OMNISERVE_SD_STUB"],
                "OMNISERVE_NATIVE_SD_MODEL": "stub.gguf",
                "OMNISERVE_NATIVE_SD_ACCEL_PROFILE": "lightning",
                "OMNISERVE_NATIVE_SD_DEFAULT_LORA": str(lora),
                "OMNISERVE_NATIVE_SD_CACHE_MODE": "easycache",
                "OMNISERVE_NATIVE_BIND": "127.0.0.1",
                "OMNISERVE_NATIVE_SD_WARMUP": "0",
                "OMNISERVE_STUB_DELAY_MS": "0", "OMNISERVE_STUB_PARAMS": str(capture),
            })
            with (root / "log").open("w") as log:
                proc = subprocess.Popen([os.environ["OMNISERVE_NATIVE_BIN"], "--port", str(port)],
                                        env=env, stdout=log, stderr=log)
                try:
                    for _ in range(100):
                        try:
                            request(port, "/status", timeout=1)
                            break
                        except OSError:
                            if proc.poll() is not None:
                                self.fail((root / "log").read_text())
                            time.sleep(.05)
                    for edit, turbo in ((False, True), (True, True), (True, False)):
                        body = {"prompt": "adult portrait", "width": 1024, "height": 1024,
                                "steps": 17, "turbo": turbo, "guidance_scale": 6,
                                "cache": False}
                        if edit:
                            body["image_base64"] = source_png()
                        request(port, "/v1/images/edits" if edit else "/v1/images/generations", body)
                    rows = [json.loads(line) for line in capture.read_text().splitlines()]
                    self.assertEqual(len(rows), 3)
                    for index, row in enumerate(rows[:2]):
                        self.assertEqual(row["steps"], 4)
                        self.assertEqual(row["cfg"], 1)
                        self.assertEqual(row["sampler"], 0)
                        self.assertEqual(row["refs"], index)
                        self.assertEqual(row["loras"], 1)
                        self.assertEqual(row["cache"], 0)
                        for got, want in zip(row["sigmas"], expected_sigmas(1024, 1024), strict=True):
                            self.assertAlmostEqual(got, want, places=6)
                    self.assertEqual(rows[2]["steps"], 17)
                    self.assertEqual(rows[2]["loras"], 0)
                    self.assertEqual(rows[2]["sigmas"], [])
                finally:
                    proc.terminate()
                    proc.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()
