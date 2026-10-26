"""Preflight control-flow fixtures; these are not GPU or profiler execution results."""
import importlib.util
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

path = Path(__file__).resolve().parents[2] / "scripts/gpu_preflight.py"
spec = importlib.util.spec_from_file_location("preflight", path)
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


def fixture(stdout="", status="exited", returncode=0):
    return dict(stdout=stdout, stderr="", status=status, returncode=returncode)


class PreflightTest(unittest.TestCase):
    def test_overall_deadline_starts_no_more_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "deadline"
            clock = itertools.chain([0], itertools.repeat(2))
            with mock.patch.object(sys, "argv", ["preflight", "--output", str(output), "--run-gpu", "--cuda-arch", "86", "--max-seconds", "1"]), mock.patch.object(preflight.time, "monotonic", side_effect=clock), mock.patch.object(preflight, "run_command") as run:
                self.assertEqual(preflight.main(), 2)
                run.assert_not_called()
            result = json.loads((output / "preflight.json").read_text())
            self.assertTrue(all(c["status"] == "timed_out" for c in result["commands"].values()))
            self.assertFalse(result["execution_checks_passed"])

    def test_only_specific_success_counts_and_denial_is_preserved(self):
        self.assertFalse(preflight.probe_passed(fixture("version 1"), "KVLAB_NATIVE_CUDA_OK"))
        self.assertFalse(preflight.probe_passed(fixture("KVLAB_NATIVE_CUDA_OK", returncode=1), "KVLAB_NATIVE_CUDA_OK"))
        self.assertFalse(preflight.probe_passed(fixture("KVLAB_NATIVE_CUDA_OK", status="timed_out", returncode=None), "KVLAB_NATIVE_CUDA_OK"))
        self.assertEqual(preflight.counter_status(fixture("ERR_NVGPUCTRPERM", returncode=1)), "permission_denied")
        self.assertEqual(preflight.counter_status(fixture("kvlab_smoke KVLAB_NATIVE_CUDA_OK No kernels were profiled")), "failed")
        self.assertEqual(preflight.counter_status(fixture("kvlab_smoke KVLAB_NATIVE_CUDA_OK")), "verified")

    def test_missing_and_timed_out_commands_do_not_pass(self):
        self.assertEqual(preflight.run_command(["/nonexistent-kvlab-program"])["status"], "not_installed")
        result = preflight.run_command([sys.executable, "-c", "import time; time.sleep(10)"], timeout=0.05)
        self.assertEqual(result["status"], "timed_out")
        self.assertIsNone(result["returncode"])

    def test_inventory_never_claims_execution_and_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "inventory"
            args = ["gpu_preflight.py", "--output", str(output)]
            with mock.patch.object(sys, "argv", args), mock.patch.object(preflight, "run_command", return_value=fixture("version only")):
                self.assertEqual(preflight.main(), 0)
                result = json.loads((output / "preflight.json").read_text())
                self.assertFalse(result["execution_checks_passed"])
                self.assertEqual(result["performance_counters"], "not_tested")
                with self.assertRaises(FileExistsError):
                    preflight.main()

    def test_failed_compile_stops_before_launching_native_or_frameworks(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failed"
            def fake(argv, timeout=30):
                return fixture("compiler unavailable", returncode=1) if "-arch=sm_86" in argv else fixture("inventory")
            with mock.patch.object(sys, "argv", ["preflight", "--output", str(output), "--run-gpu", "--cuda-arch", "86"]), mock.patch.object(preflight, "run_command", side_effect=fake):
                self.assertEqual(preflight.main(), 2)
            result = json.loads((output / "preflight.json").read_text())
            self.assertEqual(result["status"], "failed_native_build")
            self.assertNotIn("torch", result["commands"])
            self.assertNotIn("native", result["commands"])

    def test_missing_framework_and_counter_denial_do_not_become_success(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "execute"
            def fake(argv, timeout=30):
                values = list(map(str, argv))
                if values[0] == "ncu" and "--set" in values:
                    return fixture("ERR_NVGPUCTRPERM", returncode=1)
                if "--mode" in values:
                    mode = values[-1]
                    if mode == "triton":
                        return fixture("missing triton", returncode=1)
                    return fixture("KVLAB_" + mode.upper() + "_CUDA_OK")
                if values[0] == "compute-sanitizer" and "--tool" in values:
                    return fixture("KVLAB_NATIVE_CUDA_OK ERROR SUMMARY: 0 errors")
                return fixture("KVLAB_NATIVE_CUDA_OK")
            with mock.patch.object(sys, "argv", ["preflight", "--output", str(output), "--run-gpu", "--cuda-arch", "89"]), mock.patch.object(preflight, "run_command", side_effect=fake):
                self.assertEqual(preflight.main(), 2)
            result = json.loads((output / "preflight.json").read_text())
            self.assertEqual(result["performance_counters"], "permission_denied")
            self.assertEqual(result["timeline_capture"], "failed")  # No report file was generated.
            self.assertFalse(result["execution_checks_passed"])
            self.assertFalse(result["execution_checks"]["triton"])


if __name__ == "__main__":
    unittest.main()
