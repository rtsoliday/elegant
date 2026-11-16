"""Checks for suite options; no elegant subprocesses are launched."""
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


class ExtraDeckTests(unittest.TestCase):
    def make_suite(self, root, configuration=None):
        case = root / "alpha-heavy"
        case.mkdir()
        (root / "suite.json").write_text(json.dumps(configuration or {}))
        (root / "data").mkdir()
        (root / "data" / "beam.ele").write_text("not a test\n")
        (case / "lattice.lte").write_text("D: DRIF\n")
        (case / "run.ele").write_text("main\n")
        (case / "run-small-fallback.ele").write_text("small\n")
        (case / "mode2.ele").write_text("mode2\n")

    def test_discovery_skips_exclusions_and_folders_without_run_ele(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_suite(root)
            self.assertEqual(
                regression.discover_extra_decks(root, ["alpha-heavy/mode2.ele"]),
                ["alpha-heavy__run-small-fallback"],
            )

    def test_named_deck_must_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_suite(root)
            names, _ = regression.select_tests(root, ["alpha-heavy__mode2"])
            self.assertEqual(names, ["alpha-heavy__mode2"])
            with self.assertRaises(regression.RegressionError):
                regression.select_tests(root, ["alpha-heavy__absent"])

    def test_export_runs_the_extra_deck_as_run_ele(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_suite(root)
            destination = root / "work"
            regression.export_named_test(
                root, "alpha-heavy__run-small-fallback", destination, "local"
            )
            self.assertEqual((destination / "run.ele").read_text(), "small\n")
            self.assertFalse((destination / "run-small-fallback.ele").exists())
            self.assertTrue((destination / "lattice.lte").is_file())
            self.assertEqual((root / "alpha-heavy" / "run.ele").read_text(), "main\n")

    def test_plain_name_exports_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_suite(root)
            destination = root / "work"
            regression.export_named_test(root, "alpha-heavy", destination, "local")
            self.assertEqual((destination / "run.ele").read_text(), "main\n")
            self.assertTrue((destination / "mode2.ele").is_file())

    def test_baseline_rejects_extra_decks_in_timed_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_suite(root)
            args = regression.build_parser().parse_args(
                ["baseline", "--test-set", str(root), "--elegant", "elegant",
                 "--output", str(root / "artifact"), "--include-extra-decks"]
            )
            with patch.object(regression, "require_commands"), patch.object(
                regression, "resolve_executable"
            ):
                with self.assertRaises(regression.RegressionError):
                    regression.baseline_command(args)
            self.assertFalse((root / "artifact").exists())


class EnvironmentOverrideTests(unittest.TestCase):
    metadata = {"suite": {
        "gpu_environment": {"ELEGANT_GPU_MODE": "required"},
        "gpu_environment_by_test": {"alpha-*": {"ELEGANT_GPU_MODE": "auto"}},
    }}

    def test_suite_settings_can_differ_by_test(self):
        environments = regression.suite_test_environments(self.metadata)
        base = regression.suite_environment(self.metadata)
        self.assertEqual(
            regression.environment_for_test(base, environments, "alpha-one"),
            {"ELEGANT_GPU_MODE": "auto"},
        )
        self.assertEqual(
            regression.environment_for_test(base, environments, "beta"),
            {"ELEGANT_GPU_MODE": "required"},
        )

    def test_command_line_overrides_per_test_settings(self):
        settings = ["ELEGANT_GPU_MODE=off"]
        self.assertEqual(
            regression.environment_for_test(
                regression.suite_environment(self.metadata, settings),
                regression.suite_test_environments(self.metadata, settings),
                "alpha-one",
            ),
            {"ELEGANT_GPU_MODE": "off"},
        )

    def test_per_test_settings_must_be_gpu_strings(self):
        for value in ("auto", {"PATH": "x"}, {"ELEGANT_GPU_MODE": 1}):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "suite.json").write_text(
                    json.dumps({"gpu_environment_by_test": {"alpha": value}})
                )
                with self.assertRaises(regression.RegressionError):
                    regression.local_suite_configuration(root)

    def test_command_line_overrides_suite(self):
        self.assertEqual(
            regression.suite_environment(
                self.metadata, ["ELEGANT_GPU_MODE=off", "CUDA_VISIBLE_DEVICES="]),
            {"ELEGANT_GPU_MODE": "off", "CUDA_VISIBLE_DEVICES": ""},
        )
        self.assertEqual(
            regression.suite_environment(self.metadata),
            {"ELEGANT_GPU_MODE": "required"},
        )

    def test_setting_needs_a_name_and_equals_sign(self):
        for setting in ("ELEGANT_GPU_MODE", "=off"):
            with self.assertRaises(regression.RegressionError):
                regression.suite_environment(self.metadata, [setting])


class GpuActivityExemptionTests(unittest.TestCase):
    manifest = {"test_set": {"suite": {"gpu_activity_exempt_tests": [
        "frequency-map-batched-heavy", "*__*fallback"]}}}

    def test_patterns_match_only_listed_tests(self):
        for name in ("frequency-map-batched-heavy", "alpha__run-small-fallback"):
            self.assertTrue(regression.gpu_activity_exempt(self.manifest, name))
        for name in ("alpha-heavy", "alpha__run-loss", "alpha-fallback"):
            self.assertFalse(regression.gpu_activity_exempt(self.manifest, name))

    def test_manifests_without_a_suite_exempt_nothing(self):
        self.assertFalse(regression.gpu_activity_exempt({}, "alpha-heavy"))

    def test_suite_lists_must_hold_strings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "suite.json").write_text(
                json.dumps({"gpu_activity_exempt_tests": "alpha-heavy"})
            )
            with self.assertRaises(regression.RegressionError):
                regression.local_suite_configuration(root)



class PelegantRunTests(unittest.TestCase):
    def test_preference_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, reference in (("serial", "reference"), ("both", "reference"),
                                    ("parallel", "preference")):
                (root / name / reference).mkdir(parents=True)
                (root / name / "run.ele").write_text("&stop &end\n")
            (root / "both" / "preference").mkdir()
            names, _ = regression.select_tests(root, [], reference_directory="preference")
            self.assertEqual(names, ["both", "parallel"])

    def test_launcher_runs_each_input_file(self):
        with tempfile.TemporaryDirectory() as directory:
            test_dir = Path(directory)
            for name in ("run2.ele", "run1.ele"):
                (test_dir / name).write_text("")
            commands, require_done, mode = regression.prepare_commands(
                test_dir, Path("/bin/Pelegant"), ["-x"], ["mpiexec", "-np", "4"])
            self.assertEqual(commands, [
                ["mpiexec", "-np", "4", "/bin/Pelegant", "run1.ele", "-x"],
                ["mpiexec", "-np", "4", "/bin/Pelegant", "run2.ele", "-x"]])
            self.assertFalse(require_done)
            self.assertEqual(mode, "mpi")

    def test_launcher_is_the_template_command(self):
        with tempfile.TemporaryDirectory() as directory:
            test_dir = Path(directory)
            (test_dir / "run.ele").write_text("")
            (test_dir / "runTemplate").write_text("#!/bin/sh\n<executable> run.ele\n")
            regression.prepare_commands(
                test_dir, Path("/bin/Pelegant"), [], ["mpiexec", "-np", "4"])
            self.assertEqual((test_dir / "runJob").read_text(),
                             "#!/bin/sh\nmpiexec -np 4 /bin/Pelegant run.ele\n")


if __name__ == "__main__":
    unittest.main()
