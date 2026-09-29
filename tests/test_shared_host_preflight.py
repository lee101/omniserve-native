import unittest
from unittest.mock import patch

from tools.shared_host_preflight import capacity, main, parse_gpus, parse_meminfo


class SharedHostPreflightTests(unittest.TestCase):
    def test_meminfo_units(self):
        self.assertEqual(parse_meminfo("MemTotal: 1024 kB\nHugePages_Total: 0\n"),
                         {"MemTotal": 1048576})

    def test_gpu_csv(self):
        gpu = parse_gpus("0, NVIDIA RTX 5090, 32768, 14336, 18432, 0\n")[0]
        self.assertEqual(gpu["free_gib"], 14)
        self.assertEqual(gpu["total_gib"], 32)

    def test_shared_host_does_not_use_total_vram(self):
        result = capacity({"free_gib": 14}, 15.7, 2, 2, 24)
        self.assertFalse(result["fits_estimate"])
        self.assertAlmostEqual(result["shortfall_gib"], 5.7)

    def test_target_cap_and_reserve(self):
        self.assertFalse(capacity({"free_gib": 80}, 23, 2, 2, 24)["fits_estimate"])
        self.assertTrue(capacity({"free_gib": 24}, 19, 3, 2, 24)["fits_estimate"])
        self.assertEqual(capacity({"free_gib": 1}, 1, 1, 2, 24)["candidate_budget_gib"], 0)

    def test_invalid_sizes(self):
        for value in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                capacity({"free_gib": 24}, value, 2, 2, 24)

    def test_missing_gpu_fails_closed(self):
        with patch("sys.argv", ["preflight", "--model", __file__, "--runtime-gib", "2"]), \
                patch("tools.shared_host_preflight.snapshot", return_value={"gpus": []}), \
                patch("builtins.print"):
            self.assertEqual(main(), 2)


if __name__ == "__main__":
    unittest.main()
