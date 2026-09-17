"""Checks for correctness-only mode; no elegant subprocesses are launched."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import elegant_test_regression as regression


class CorrectnessOnlyTests(unittest.TestCase):
    metadata = {"kind": "local", "suite": {
        "suite_type": "gpu-performance", "recommended_jobs": 1}}

    def test_parallel_timing_still_rejected(self):
        with self.assertRaises(regression.RegressionError):
            regression.validate_suite_run_settings(self.metadata, 8)
        regression.validate_suite_run_settings(self.metadata, 1)

    def test_parallel_correctness_requires_explicit_option(self):
        parser = regression.build_parser()
        base = ["baseline", "--elegant", "elegant", "--output", "artifact"]
        self.assertFalse(parser.parse_args(base).correctness_only)
        args = parser.parse_args(base + ["--correctness-only", "--jobs", "8"])
        regression.validate_suite_run_settings(
            self.metadata, args.jobs, correctness_only=args.correctness_only)

    def test_correctness_artifact_does_not_claim_performance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {"test_set": self.metadata, "tests": [],
                        "run_options": {"correctness_only": True}}
            with patch.object(regression, "hardware_metadata") as hardware:
                regression.write_performance_baseline_summary(root, manifest)
                hardware.assert_not_called()
            self.assertFalse((root / "baseline-summary.json").exists())

    def test_default_artifact_keeps_performance_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {"test_set": self.metadata, "tests": [], "run_options": {}}
            with patch.object(regression, "hardware_metadata", return_value={}):
                regression.write_performance_baseline_summary(root, manifest)
            summary = json.loads((root / "baseline-summary.json").read_text())
            self.assertEqual(summary["summary"]["tests"], 0)


if __name__ == "__main__":
    unittest.main()
