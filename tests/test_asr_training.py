import importlib.util
import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


train_asr = load_module("train_asr", ROOT / "workers" / "train_asr.py")


class ManifestValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.audio = self.root / "audio"
        self.audio.mkdir()
        for name in ("train.wav", "valid.wav"):
            with wave.open(str(self.audio / name), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(b"\0\0" * 1600)

    def tearDown(self):
        self.temp.cleanup()

    def rows(self):
        base = {
            "text": "consented test speech",
            "duration": 0.1,
            "speaker_id": "speaker-a",
            "consent_scope": "public_model_weights",
            "consent_version": 1,
            "consented_at": "2026-07-25T00:00:00Z",
            "speaker_rights_confirmed": True,
        }
        return [
            {**base, "audio_filepath": "train.wav", "split": "train"},
            {**base, "audio_filepath": "valid.wav", "split": "validation"},
        ]

    def write_manifest(self, rows):
        path = self.root / "manifest.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path

    def test_explicit_public_weight_consent_is_accepted(self):
        audit = train_asr.validate_manifest(self.write_manifest(self.rows()), self.audio)
        self.assertEqual(audit["rows"], 2)
        self.assertEqual(audit["consent_scope"], "public_model_weights")
        self.assertEqual(audit["splits"], {"train": 1, "validation": 1})

    def test_legacy_product_improvement_consent_is_rejected(self):
        rows = self.rows()
        rows[0]["consent_scope"] = "product_improvement"
        with self.assertRaisesRegex(ValueError, "public-weight consent is absent"):
            train_asr.validate_manifest(self.write_manifest(rows), self.audio)

    def test_revoked_consent_is_rejected(self):
        rows = self.rows()
        rows[0]["revoked"] = True
        with self.assertRaisesRegex(ValueError, "consent was revoked"):
            train_asr.validate_manifest(self.write_manifest(rows), self.audio)

    def test_audio_must_stay_under_declared_root(self):
        rows = self.rows()
        rows[0]["audio_filepath"] = "../outside.wav"
        with self.assertRaisesRegex(ValueError, "escapes audio root"):
            train_asr.validate_manifest(self.write_manifest(rows), self.audio)


if __name__ == "__main__":
    unittest.main()
