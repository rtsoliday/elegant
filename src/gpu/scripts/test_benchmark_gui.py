"""Fast benchmark-GUI checks; no elegant process or generated runner is used."""

import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import benchmark_gui as gui
import elegant_test_regression as regression


class BenchmarkPlannerTests(unittest.TestCase):
    def inputs(self, root: Path, **changes):
        values = dict(
            suite=Path("/repo/src/gpu/test-set"),
            output=root / "cycle",
            cases=("case-a", "case-b"),
            mode="quick",
            cpu_source="run",
            cpu_value="/repo/cpu",
            prior_source="none",
            prior_value="",
            candidate="/repo/gpu",
            targets=(),
            overrides={},
        )
        values.update(changes)
        return gui.CycleInputs(**values)

    def test_quick_commands_use_single_correctness_run(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs = self.inputs(Path(directory))
            with patch.object(regression, "require_commands"), patch.object(
                regression,
                "test_set_metadata",
                return_value={
                    "kind": "local",
                    "suite": {"suite_type": "gpu-performance"},
                },
            ), patch.object(
                regression, "select_tests", return_value=(list(inputs.cases), [])
            ), patch.object(
                regression, "resolve_executable", side_effect=lambda path: Path(path)
            ), patch.object(
                regression, "sha256_file", return_value="hash"
            ):
                prepared = gui.prepare_cycle(inputs, Path("/repo/harness.py"))
            self.assertEqual(
                [stage.name for stage in prepared.stages],
                ["cpu", "candidate-gpu", "cpu-comparison"],
            )
            self.assertIn("--correctness-only", prepared.stages[0].command)
            self.assertIn("--cancel-file", prepared.stages[0].command)
            self.assertNotIn("--pre-change-gpu", prepared.stages[-1].command)
            self.assertEqual(
                prepared.workflow["inputs"]["candidate_gpu"]["sha256"], "hash"
            )

    def test_timed_prior_defaults_to_guard_only(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs = self.inputs(
                Path(directory),
                mode="timed",
                prior_source="run",
                prior_value="/repo/prior",
            )
            with patch.object(regression, "require_commands"), patch.object(
                regression,
                "test_set_metadata",
                return_value={
                    "kind": "local",
                    "suite": {"suite_type": "gpu-performance"},
                },
            ), patch.object(
                regression, "select_tests", return_value=(list(inputs.cases), [])
            ), patch.object(
                regression, "resolve_executable", side_effect=lambda path: Path(path)
            ), patch.object(
                regression, "sha256_file", return_value="hash"
            ), patch.object(
                regression, "hardware_metadata", return_value={}
            ):
                prepared = gui.prepare_cycle(inputs, Path("/repo/harness.py"))
            names = [stage.name for stage in prepared.stages]
            self.assertEqual(
                names,
                [
                    "cpu",
                    "prior-gpu",
                    "candidate-gpu",
                    "cpu-comparison",
                    "prior-gpu-comparison",
                ],
            )
            cpu_compare = prepared.stages[3].command
            self.assertIn("--guard-only", cpu_compare)
            self.assertIn("--pre-change-gpu", cpu_compare)
            self.assertNotIn("--target-test", cpu_compare)
            self.assertEqual(prepared.stages[0].command[-2:], ("case-a", "case-b"))

    def test_override_validation_rejects_nonfinite_values(self):
        for value in ("nan", "inf", "-1"):
            with self.subTest(value=value):
                with self.assertRaises(regression.RegressionError):
                    gui.parse_overrides({"gpu_noise_absolute_tolerance": value})
        with self.assertRaises(regression.RegressionError):
            gui.parse_overrides({"suite_regression_limit": "1.1"})


class GuardOnlyTests(unittest.TestCase):
    @staticmethod
    def sample_run(seconds, *, gpu=False):
        return {
            "tests": [
                {
                    "name": "case-a",
                    "execution_seconds": seconds,
                    "gpu_usage": {"total_elements": 1} if gpu else None,
                }
            ]
        }

    def test_empty_target_set_enforces_prior_gpu_slowdown(self):
        result = regression.assess_runtime_performance(
            self.sample_run(10),
            self.sample_run(10, gpu=True),
            minimum_speedup=2,
            require_gpu_activity=True,
            preceding_gpu_manifest=self.sample_run(9, gpu=True),
            target_tests=set(),
        )
        self.assertEqual(result["gate_mode"], "guard_only")
        self.assertEqual(result["target_tests"], [])
        self.assertEqual(result["tests"][0]["status"], "needs_review")
        self.assertFalse(result["complete"])

    def test_no_prior_is_informational(self):
        result = regression.assess_runtime_performance(
            self.sample_run(10),
            self.sample_run(9, gpu=True),
            minimum_speedup=2,
            require_gpu_activity=True,
            target_tests=set(),
        )
        self.assertEqual(result["gate_mode"], "informational")
        self.assertEqual(result["tests"][0]["status"], "informational")
        self.assertEqual(result["summary"]["tests_passing_gates"], 0)
        self.assertTrue(result["complete"])

    def test_parser_keeps_original_target_default(self):
        parser = regression.build_parser()
        args = parser.parse_args(
            [
                "compare-existing",
                "--baseline",
                "cpu",
                "--candidate",
                "gpu",
                "--output",
                "comparison",
            ]
        )
        self.assertFalse(args.guard_only)
        self.assertEqual(args.target_test, [])

    def test_guard_only_rejects_speedup_target_before_output_creation(self):
        parser = regression.build_parser()
        args = parser.parse_args(
            [
                "compare-existing",
                "--baseline",
                "cpu",
                "--candidate",
                "gpu",
                "--output",
                "comparison",
                "--guard-only",
                "--target-test",
                "case-a",
            ]
        )
        with patch.object(regression, "require_commands") as require:
            with self.assertRaisesRegex(
                regression.RegressionError, "cannot be combined"
            ):
                regression.compare_existing_command(args)
            require.assert_not_called()


class ReuseAndCancellationTests(unittest.TestCase):
    def test_reuse_rejects_wrong_cases(self):
        manifest = {
            "test_set": {"kind": "local", "fingerprint": "same"},
            "tests": [{"name": "other", "gpu_usage": None}],
        }
        with patch.object(regression, "load_baseline", return_value=manifest):
            with self.assertRaisesRegex(regression.RegressionError, "case list"):
                gui.validate_reused_artifact(
                    Path("/artifact"),
                    role="CPU",
                    metadata={"kind": "local", "fingerprint": "same"},
                    cases=("case-a",),
                    timed=False,
                    current_hardware=None,
                )

    def test_reuse_rejects_hardware_mismatch_for_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            hardware = {
                "hostname": "other",
                "cpu_model": "cpu",
                "gpus": [{"uuid": "gpu-1", "driver": "driver-1"}],
            }
            (path / "baseline-summary.json").write_text(
                json.dumps({"hardware": hardware})
            )
            manifest = {
                "test_set": {"kind": "local", "fingerprint": "same"},
                "run_options": {
                    "jobs": 1,
                    "warmup_runs": 1,
                    "repetitions": 5,
                    "additional_repetitions_if_mad_exceeds_percent": 5,
                    "maximum_timing_dispersion_percent": 5.0,
                    "timing_metric": "execution_seconds",
                    "executable_arguments": [],
                },
                "tests": [{"name": "case-a", "timing_repetitions": 5}],
            }
            with patch.object(regression, "load_baseline", return_value=manifest):
                with self.assertRaisesRegex(
                    regression.RegressionError, "hardware differs"
                ):
                    gui.validate_reused_artifact(
                        path,
                        role="CPU",
                        metadata={"kind": "local", "fingerprint": "same"},
                        cases=("case-a",),
                        timed=True,
                        current_hardware={
                            "hostname": "here",
                            "cpu_model": "cpu",
                            "gpus": [{"uuid": "gpu-1", "driver": "driver-1"}],
                        },
                    )

    def test_reuse_rejects_missing_noisy_sample_policy(self):
        manifest = {
            "test_set": {"kind": "local", "fingerprint": "same"},
            "run_options": {
                "jobs": 1,
                "warmup_runs": 1,
                "repetitions": 5,
                "timing_metric": "execution_seconds",
                "executable_arguments": [],
            },
            "tests": [{"name": "case-a", "timing_repetitions": 5}],
        }
        with patch.object(regression, "load_baseline", return_value=manifest):
            with self.assertRaisesRegex(regression.RegressionError, "protocol"):
                gui.validate_reused_artifact(
                    Path("/artifact"),
                    role="CPU",
                    metadata={"kind": "local", "fingerprint": "same"},
                    cases=("case-a",),
                    timed=True,
                    current_hardware={},
                )

    def test_cancel_file_stops_registered_child(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "stop"
            child = Mock()
            child.poll.return_value = None
            with regression.CancellationController(marker) as controller:
                controller.register(child, False)
                marker.write_text("stop")
                for _ in range(30):
                    if child.terminate.called:
                        break
                    time.sleep(0.02)
                self.assertTrue(child.terminate.called)
                with self.assertRaises(regression.CancellationRequested):
                    controller.check()

    def test_result_reader_combines_both_comparisons(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("cpu-comparison", "prior-gpu-comparison", "candidate-gpu"):
                (root / name).mkdir()
            (root / "cpu-comparison" / "manifest.json").write_text(
                json.dumps(
                    {
                        "complete": True,
                        "comparisons": [{"name": "case-a", "status": "unchanged"}],
                        "gpu_significance_assessment": {"summary": {}},
                    }
                )
            )
            (root / "prior-gpu-comparison" / "manifest.json").write_text(
                json.dumps(
                    {
                        "complete": True,
                        "comparisons": [
                            {
                                "name": "case-a",
                                "status": "probably_expected_gpu_roundoff",
                                "changes": [
                                    {
                                        "path": "beam.sdds",
                                        "detail": "screened difference",
                                    }
                                ],
                            }
                        ],
                    }
                )
            )
            (root / "candidate-gpu" / "manifest.json").write_text(
                json.dumps(
                    {
                        "tests": [
                            {
                                "name": "case-a",
                                "execution_seconds": 1.25,
                                "gpu_usage": {"total_elements": 3},
                            }
                        ],
                    }
                )
            )
            result = gui.read_results(root)
            self.assertTrue(result["complete"])
            self.assertEqual(result["rows"][0]["gpu_elements"], 3)
            self.assertEqual(
                result["rows"][0]["prior_status"], "probably_expected_gpu_roundoff"
            )
            self.assertIn(
                "Prior GPU: beam.sdds — screened difference",
                result["rows"][0]["findings"],
            )


if __name__ == "__main__":
    unittest.main()
