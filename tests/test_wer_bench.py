"""Tests for the WER regression harness.

The gate is only worth having if the arithmetic is right, so these pin the
alignment counts, corpus aggregation, normalization, and the regression gate.
"""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import wer_bench  # noqa: E402


class TestNormalize(unittest.TestCase):
    def test_casefold_and_punctuation(self):
        self.assertEqual(wer_bench.normalize("Hello, World!"), "hello world")

    def test_collapses_whitespace(self):
        self.assertEqual(wer_bench.normalize("  a\t\tb\n c "), "a b c")

    def test_keeps_apostrophes(self):
        # don't -> dont would silently change the word count.
        self.assertEqual(wer_bench.normalize("don't"), "don't")

    def test_unicode_is_folded(self):
        self.assertEqual(wer_bench.normalize("Ｃafé"), "café")

    def test_empty(self):
        self.assertEqual(wer_bench.tokens("   "), [])


class TestAlign(unittest.TestCase):
    def test_perfect_match(self):
        c = wer_bench.align(["a", "b", "c"], ["a", "b", "c"])
        self.assertEqual((c.hits, c.substitutions, c.deletions, c.insertions), (3, 0, 0, 0))
        self.assertEqual(c.wer(), 0.0)

    def test_single_substitution(self):
        c = wer_bench.align(["a", "b", "c"], ["a", "x", "c"])
        self.assertEqual(c.substitutions, 1)
        self.assertEqual(c.errors, 1)
        self.assertAlmostEqual(c.wer(), 1 / 3)

    def test_deletion(self):
        c = wer_bench.align(["a", "b", "c"], ["a", "c"])
        self.assertEqual(c.deletions, 1)
        self.assertAlmostEqual(c.wer(), 1 / 3)

    def test_insertion(self):
        c = wer_bench.align(["a", "c"], ["a", "b", "c"])
        self.assertEqual(c.insertions, 1)
        self.assertAlmostEqual(c.wer(), 1 / 2)

    def test_empty_hypothesis_is_all_deletions(self):
        c = wer_bench.align(["a", "b"], [])
        self.assertEqual(c.deletions, 2)
        self.assertEqual(c.wer(), 1.0)

    def test_empty_reference_with_output_is_wrong(self):
        c = wer_bench.align([], ["a"])
        self.assertEqual(c.insertions, 1)
        self.assertEqual(c.wer(), 1.0)

    def test_both_empty_is_perfect(self):
        self.assertEqual(wer_bench.align([], []).wer(), 0.0)

    def test_reference_words_excludes_insertions(self):
        # WER denominator is reference length, not hypothesis length.
        c = wer_bench.align(["a", "b"], ["a", "b", "c", "d"])
        self.assertEqual(c.reference_words, 2)
        self.assertEqual(c.wer(), 1.0)


class TestCorpusAggregation(unittest.TestCase):
    def test_corpus_wer_weights_by_length(self):
        """A long clip must dominate a short one; averaging per-clip WER would not."""
        total = wer_bench.Counts()
        # 1 error out of 1 word
        total.add(wer_bench.align(["a"], ["x"]))
        # 0 errors out of 99 words
        long_ref = ["w"] * 99
        total.add(wer_bench.align(long_ref, long_ref))
        self.assertEqual(total.reference_words, 100)
        self.assertAlmostEqual(total.wer(), 0.01)
        # The per-clip mean would be 0.5 -- fifty times worse. That gap is the
        # whole reason corpus WER is the headline number.


class TestExtractText(unittest.TestCase):
    def test_plain_key(self):
        self.assertEqual(wer_bench.extract_text({"text": "hi"}), "hi")

    def test_nested_shapes(self):
        self.assertEqual(wer_bench.extract_text({"output": {"transcript": "hi"}}), "hi")

    def test_missing(self):
        self.assertEqual(wer_bench.extract_text({"nope": 1}), "")


class _Args:
    """Minimal stand-in for the argparse namespace the gate reads."""

    def __init__(self, **kw):
        self.baseline = None
        self.max_wer = None
        self.max_wer_regression = 0.0
        self.allow_failures = False
        self.per_clip = False
        self.url = "http://x/"
        self.manifest = Path("m.jsonl")
        for k, v in kw.items():
            setattr(self, k, v)


class TestGate(unittest.TestCase):
    def _report(self, **kw):
        base = {
            "clips_scored": 5,
            "clips_failed": 0,
            "wer": 0.10,
            "rtf": 1.0,
            "normalization": "nfkc+casefold+strip-punctuation+collapse-space",
        }
        base.update(kw)
        return base

    def test_clean_run_passes(self):
        self.assertEqual(wer_bench.gate(self._report(), _Args()), [])

    def test_no_clips_fails(self):
        problems = wer_bench.gate(self._report(clips_scored=0), _Args())
        self.assertTrue(any("no clips" in p for p in problems))

    def test_transcription_failure_fails_by_default(self):
        problems = wer_bench.gate(self._report(clips_failed=2), _Args())
        self.assertTrue(any("failed to transcribe" in p for p in problems))

    def test_failures_can_be_allowed(self):
        self.assertEqual(wer_bench.gate(self._report(clips_failed=2), _Args(allow_failures=True)), [])

    def test_absolute_ceiling(self):
        problems = wer_bench.gate(self._report(wer=0.4), _Args(max_wer=0.3))
        self.assertTrue(any("exceeds" in p for p in problems))

    def _baseline_file(self, tmp: Path, wer: float, rtf: float = 1.0, norm: str | None = None):
        path = tmp / "baseline.json"
        path.write_text(
            json.dumps(
                {
                    "wer": wer,
                    "rtf": rtf,
                    "normalization": norm or "nfkc+casefold+strip-punctuation+collapse-space",
                }
            )
        )
        return path

    def test_regression_blocks(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            base = self._baseline_file(Path(td), 0.05)
            report = self._report(wer=0.09)
            problems = wer_bench.gate(report, _Args(baseline=base))
            self.assertTrue(any("regressed" in p for p in problems))
            self.assertAlmostEqual(report["wer_delta"], 0.04)

    def test_improvement_passes_and_records_speedup(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            base = self._baseline_file(Path(td), 0.10, rtf=2.0)
            report = self._report(wer=0.08, rtf=0.5)
            self.assertEqual(wer_bench.gate(report, _Args(baseline=base)), [])
            self.assertAlmostEqual(report["wer_delta"], -0.02)
            self.assertAlmostEqual(report["rtf_speedup"], 4.0)

    def test_tolerance_allows_small_regression(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            base = self._baseline_file(Path(td), 0.10)
            args = _Args(baseline=base, max_wer_regression=0.01)
            self.assertEqual(wer_bench.gate(self._report(wer=0.105), args), [])

    def test_mismatched_normalization_is_refused(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            base = self._baseline_file(Path(td), 0.10, norm="lowercase-only")
            problems = wer_bench.gate(self._report(), _Args(baseline=base))
            self.assertTrue(any("normalization" in p for p in problems))


if __name__ == "__main__":
    unittest.main()
