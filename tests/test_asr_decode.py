"""Tests for the in-memory WAV fast path in the ASR worker.

The optimization is only safe if the model sees identical samples, so the
central test compares the fast path against soundfile bit for bit. Anything the
fast path cannot handle must fall back rather than guess.
"""

import io
import struct
import sys
import unittest
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workers"))


def _wav(samples, rate=16000, channels=1, width=2):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", s) for s in samples))
    return buf.getvalue()


def _load_decoder():
    """Import just the decoder without dragging in torch/transformers."""
    src = (Path(__file__).resolve().parents[1] / "workers" / "asr_worker.py").read_text()
    start = src.index("def decode_pcm16_wav")
    end = src.index("def transcribe_bytes")
    namespace = {"io": io, "wave": wave, "np": np, "TARGET_SAMPLE_RATE": 16000}
    exec(compile(src[start:end], "asr_worker.py", "exec"), namespace)
    return namespace["decode_pcm16_wav"]


decode_pcm16_wav = _load_decoder()


class TestDecodeMatchesSoundfile(unittest.TestCase):
    def test_identical_to_soundfile(self):
        """The whole optimization rests on this: same samples, so same WER."""
        try:
            import soundfile as sf
        except ImportError:
            self.skipTest("soundfile not installed")

        values = [0, 1, -1, 32767, -32768, 1234, -4321, 999]
        data = _wav(values * 64)

        fast = decode_pcm16_wav(data)
        reference, rate = sf.read(io.BytesIO(data), dtype="float32")

        self.assertEqual(rate, 16000)
        self.assertIsNotNone(fast)
        np.testing.assert_array_equal(fast, reference)

    def test_range_is_normalized(self):
        out = decode_pcm16_wav(_wav([32767, -32768, 0]))
        self.assertAlmostEqual(float(out[0]), 32767 / 32768.0)
        self.assertAlmostEqual(float(out[1]), -1.0)
        self.assertEqual(float(out[2]), 0.0)
        self.assertEqual(out.dtype, np.float32)


class TestFallbacks(unittest.TestCase):
    """Anything the fast path is not certain about must return None."""

    def test_wrong_sample_rate_falls_back(self):
        self.assertIsNone(decode_pcm16_wav(_wav([1, 2, 3], rate=44100)))

    def test_stereo_falls_back(self):
        self.assertIsNone(decode_pcm16_wav(_wav([1, 2, 3, 4], channels=2)))

    def test_eight_bit_falls_back(self):
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(1)
            w.setframerate(16000)
            w.writeframes(bytes([1, 2, 3]))
        self.assertIsNone(decode_pcm16_wav(buf.getvalue()))

    def test_compressed_container_falls_back(self):
        self.assertIsNone(decode_pcm16_wav(b"OggS\x00\x02garbage"))

    def test_garbage_falls_back(self):
        self.assertIsNone(decode_pcm16_wav(b"not audio at all"))

    def test_empty_falls_back(self):
        self.assertIsNone(decode_pcm16_wav(b""))

    def test_header_only_falls_back(self):
        self.assertIsNone(decode_pcm16_wav(_wav([])))

    def test_truncated_file_does_not_raise(self):
        truncated = _wav([1, 2, 3, 4, 5])[:20]
        self.assertIsNone(decode_pcm16_wav(truncated))

class TestDegenerateGuard(unittest.TestCase):
    """A model wired to the wrong code path returns <unk> with HTTP 200.

    That scores 100% WER while looking like a successful transcription, which is
    exactly how nvidia/parakeet-ctc-0.6b behaved as the worker's default.
    """

    @staticmethod
    def _load():
        src = (Path(__file__).resolve().parents[1] / "workers" / "asr_worker.py").read_text()
        start = src.index("_DEGENERATE_TOKENS")
        end = src.index("def transcribe_bytes")
        ns = {}
        exec(compile(src[start:end], "asr_worker.py", "exec"), ns)
        return ns["is_degenerate"]

    def test_flags_unknown_token_output(self):
        is_degenerate = self._load()
        self.assertTrue(is_degenerate("<unk>"))
        self.assertTrue(is_degenerate("<unk> <unk> <unk>"))
        self.assertTrue(is_degenerate("<pad>"))

    def test_allows_real_transcripts(self):
        is_degenerate = self._load()
        self.assertFalse(is_degenerate("hello world"))
        self.assertFalse(is_degenerate(""))

    def test_partial_unknowns_are_not_degenerate(self):
        """A real transcript with one unknown word is still a transcript."""
        is_degenerate = self._load()
        self.assertFalse(is_degenerate("the <unk> brown fox"))

if __name__ == "__main__":
    unittest.main()
