"""Tests for the ASR statistics, combination, and formatting-normalization layers."""

import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import asr_stats as st  # noqa: E402
import asr_textnorm as tn  # noqa: E402
from wer_bench import tokens  # noqa: E402


class TestAlignPath(unittest.TestCase):
    def test_identical(self):
        path = st.align_path(["a", "b"], ["a", "b"])
        self.assertEqual([op for op, _, _ in path], [st.MATCH, st.MATCH])

    def test_substitution(self):
        path = st.align_path(["a", "b"], ["a", "x"])
        self.assertEqual([op for op, _, _ in path], [st.MATCH, st.SUB])

    def test_insertion_and_deletion(self):
        self.assertIn(st.INS, [op for op, _, _ in st.align_path(["a"], ["a", "b"])])
        self.assertIn(st.DEL, [op for op, _, _ in st.align_path(["a", "b"], ["a"])])

    def test_indices_point_at_the_right_words(self):
        ref, hyp = ["the", "quick", "fox"], ["the", "slow", "fox"]
        for op, ri, hi in st.align_path(ref, hyp):
            if op == st.SUB:
                self.assertEqual(ref[ri], "quick")
                self.assertEqual(hyp[hi], "slow")


class TestBootstrap(unittest.TestCase):
    def test_point_estimate_is_corpus_wer(self):
        scores = [st.ClipScore(1, 10), st.ClipScore(3, 10)]
        ci = st.bootstrap_wer_ci(scores, resamples=500, seed=1)
        self.assertAlmostEqual(ci.point, 4 / 20)

    def test_interval_brackets_the_estimate(self):
        scores = [st.ClipScore(i % 3, 10) for i in range(20)]
        ci = st.bootstrap_wer_ci(scores, resamples=2000, seed=1)
        self.assertLessEqual(ci.low, ci.point)
        self.assertGreaterEqual(ci.high, ci.point)

    def test_perfect_system_has_zero_width(self):
        scores = [st.ClipScore(0, 10) for _ in range(5)]
        ci = st.bootstrap_wer_ci(scores, resamples=500, seed=1)
        self.assertEqual((ci.point, ci.low, ci.high), (0.0, 0.0, 0.0))

    def test_small_corpus_gives_a_wide_interval(self):
        """The whole point: 2 errors in 115 words cannot be pinned down."""
        scores = [st.ClipScore(0, 12) for _ in range(9)]
        scores[0] = st.ClipScore(1, 13)
        scores[1] = st.ClipScore(1, 12)
        ci = st.bootstrap_wer_ci(scores, resamples=4000, seed=7)
        self.assertGreater(ci.high - ci.low, 0.02)

    def test_deterministic_for_a_seed(self):
        scores = [st.ClipScore(i % 4, 9) for i in range(12)]
        a = st.bootstrap_wer_ci(scores, resamples=800, seed=3)
        b = st.bootstrap_wer_ci(scores, resamples=800, seed=3)
        self.assertEqual((a.low, a.high), (b.low, b.high))

    def test_empty_is_safe(self):
        ci = st.bootstrap_wer_ci([], resamples=10)
        self.assertEqual(ci.point, 0.0)


class TestPairedBootstrap(unittest.TestCase):
    def test_identical_systems_are_not_significant(self):
        a = [st.ClipScore(1, 10) for _ in range(10)]
        cmp = st.paired_bootstrap(a, list(a), resamples=1000, seed=2)
        self.assertAlmostEqual(cmp.delta, 0.0)
        self.assertFalse(cmp.significant)

    def test_consistent_winner_is_significant(self):
        # B beats A on every clip, so no resample can reverse it.
        a = [st.ClipScore(5, 10) for _ in range(12)]
        b = [st.ClipScore(0, 10) for _ in range(12)]
        cmp = st.paired_bootstrap(a, b, resamples=2000, seed=5)
        self.assertGreater(cmp.delta, 0)
        self.assertTrue(cmp.significant)
        self.assertLess(cmp.p_value, 0.05)

    def test_tiny_inconsistent_difference_is_not_significant(self):
        """2 errors vs 4 errors across 9 clips must not be called a win."""
        a = [st.ClipScore(0, 13) for _ in range(9)]
        b = [st.ClipScore(0, 13) for _ in range(9)]
        a[0] = st.ClipScore(1, 13)
        a[1] = st.ClipScore(1, 13)
        b[0] = st.ClipScore(2, 13)
        b[3] = st.ClipScore(2, 13)
        cmp = st.paired_bootstrap(a, b, resamples=3000, seed=11)
        self.assertFalse(cmp.significant)

    def test_mismatched_lengths_rejected(self):
        with self.assertRaises(ValueError):
            st.paired_bootstrap([st.ClipScore(1, 5)], [], resamples=10)

    def test_verdict_names_the_winner(self):
        a = [st.ClipScore(5, 10) for _ in range(12)]
        b = [st.ClipScore(0, 10) for _ in range(12)]
        cmp = st.paired_bootstrap(a, b, resamples=1000, seed=5)
        self.assertIn("beats", cmp.verdict("A", "B", 0.05))
        self.assertTrue(cmp.verdict("A", "B", 0.05).startswith("B beats A"))


class TestRover(unittest.TestCase):
    def test_unanimous_passes_through(self):
        h = [["the", "cat"], ["the", "cat"], ["the", "cat"]]
        self.assertEqual(st.rover_combine(h), ["the", "cat"])

    def test_majority_overrules_the_odd_one_out(self):
        """The payoff: two systems outvote one wrong word."""
        h = [
            ["the", "quick", "brown", "fox"],
            ["the", "quick", "brown", "fox"],
            ["the", "quick", "brown", "box"],
        ]
        self.assertEqual(st.rover_combine(h), ["the", "quick", "brown", "fox"])

    def test_voting_can_beat_every_input(self):
        # Each system makes one distinct error; the majority is right everywhere.
        h = [
            ["a", "b", "X"],
            ["a", "Y", "c"],
            ["Z", "b", "c"],
        ]
        combined = st.rover_combine(h)
        truth = ["a", "b", "c"]
        self.assertEqual(combined, truth)
        for single in h:
            self.assertNotEqual(single, truth)

    def test_majority_deletion_drops_a_word(self):
        h = [["the", "um", "cat"], ["the", "cat"], ["the", "cat"]]
        self.assertEqual(st.rover_combine(h), ["the", "cat"])

    def test_single_system_is_identity(self):
        self.assertEqual(st.rover_combine([["only", "one"]]), ["only", "one"])

    def test_empty_input(self):
        self.assertEqual(st.rover_combine([]), [])

    def test_tie_prefers_the_first_system(self):
        h = [["alpha"], ["beta"]]
        self.assertEqual(st.rover_combine(h), ["alpha"])


class TestAgreement(unittest.TestCase):
    def test_unanimous_is_one(self):
        h = [["a", "b"]] * 3
        self.assertEqual(st.agreement_rate(h), 1.0)
        self.assertEqual(st.disagreement_spans(h), [])

    def test_disagreement_is_reported(self):
        h = [["a", "b"], ["a", "c"], ["a", "b"]]
        spans = st.disagreement_spans(h)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["winner"], "b")
        self.assertLess(st.agreement_rate(h), 1.0)


class TestTextNorm(unittest.TestCase):
    def norm(self, text):
        return tn.lenient_tokens(text, tokens)

    def test_spelled_numbers_match_digits(self):
        """The real_numbers.wav case: reference is digits, model spells them."""
        self.assertEqual(self.norm("ten"), self.norm("10"))
        self.assertEqual(self.norm("testing one two three"), self.norm("testing 1 2 3"))

    def test_compound_numbers(self):
        self.assertEqual(self.norm("twenty five"), ["25"])
        self.assertEqual(self.norm("three hundred"), ["300"])
        self.assertEqual(self.norm("two thousand"), ["2000"])
        self.assertEqual(self.norm("three hundred and five"), ["305"])

    def test_letter_digit_runs_split(self):
        self.assertEqual(self.norm("mp3"), self.norm("mp 3"))

    def test_contractions_expand_on_both_sides(self):
        self.assertEqual(self.norm("don't"), self.norm("do not"))

    def test_spelling_variants(self):
        self.assertEqual(self.norm("colour"), self.norm("color"))

    def test_real_acoustic_errors_still_count(self):
        """The line this module must not cross."""
        self.assertNotEqual(self.norm("jumps"), self.norm("dumps"))
        self.assertNotEqual(self.norm("using"), self.norm("use"))
        self.assertNotEqual(self.norm("their"), self.norm("there"))

    def test_ordinary_words_survive(self):
        self.assertEqual(self.norm("the quick brown fox"), ["the", "quick", "brown", "fox"])

    def test_number_word_not_part_of_a_number_is_left_alone(self):
        # "one" as a pronoun still becomes 1; acceptable and symmetric, but the
        # surrounding words must be untouched.
        self.assertEqual(self.norm("no one knows")[0], "no")
        self.assertEqual(self.norm("no one knows")[-1], "knows")


if __name__ == "__main__":
    unittest.main()
