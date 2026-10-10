#!/usr/bin/env python3
"""The art planner must discover both legacy and current VN project IDs."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import prepare_vn_library_art


class ProjectDiscoveryTest(unittest.TestCase):
    def test_two_and_three_digit_projects_are_discovered(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = [root / "09-legacy-title", root / "237-current-title"]
            for path in expected:
                path.mkdir()
            (root / "not-a-project").mkdir()
            (root / "238-incomplete-file").write_text("ignored")

            self.assertEqual(prepare_vn_library_art.discover_projects(root), expected)

    def test_missing_library_is_empty(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(
                prepare_vn_library_art.discover_projects(Path(temporary) / "missing"),
                [],
            )


if __name__ == "__main__":
    unittest.main()
