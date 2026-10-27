"""Runner evidence/cleanup tests on CPU. Synthetic reports are not GPU results."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import check_rmsnorm_cuda as runner
import gpu_preflight as preflight


def record(stdout="", status="exited", returncode=0):
    return dict(stdout=stdout, stderr="", status=status, returncode=returncode)


def records(mode):
    result = [runner.HOST_RESULT.copy()]
    if mode == "host-only":
        return result
    for dtype, rows, width, pattern in sorted(runner.expected_cases(mode)):
        result.append(dict(dtype=dtype, rows=rows, width=width, pattern=pattern,
                           residual_bitwise_equal=True, max_abs_error=0., normalized_rms_error=0.,
                           worst_flat_index=0 if rows else None))
    result.append(dict(status="passed", device_execution=True, cases=len(result) - 1))
    return result


def encoded(rows):
    return record("\n".join(json.dumps(row) for row in rows) + "\n")


class EvidenceTest(unittest.TestCase):
    def test_complete_cases_and_host_are_distinct(self):
        self.assertEqual(len(runner.expected_cases("smoke")), 34)
        self.assertEqual(len(runner.expected_cases("full")), 218)
        for mode in ("host-only", "smoke", "full"):
            self.assertTrue(runner.checker_passed(encoded(records(mode)), mode))
        self.assertFalse(runner.checker_passed(encoded(records("host-only")), "smoke"))
        self.assertFalse(runner.checker_passed(encoded(records("smoke")), "full"))

    def test_missing_duplicate_and_wrong_cases_fail_even_with_success_summary(self):
        original = records("smoke")
        for rows in (original[:1] + original[2:], original + [original[-1]],
                     original[:2] + [original[1]] + original[3:], original[-1:]):
            self.assertFalse(runner.checker_passed(encoded(rows), "smoke"))
        original[1]["width"] = 8192  # Valid in the full matrix, not this smoke case.
        self.assertFalse(runner.checker_passed(encoded(original), "smoke"))

    def test_malformed_metrics_indices_and_residual_fail(self):
        for field, value in (("max_abs_error", float("nan")), ("normalized_rms_error", float("inf")),
                             ("max_abs_error", -1), ("max_abs_error", True),
                             ("worst_flat_index", 0), ("rows", False),
                             ("residual_bitwise_equal", False)):
            with self.subTest(field=field, value=value):
                rows = records("smoke")  # First case is empty; index must be null.
                rows[1][field] = value
                self.assertFalse(runner.checker_passed(encoded(rows), "smoke"))
        rows = records("smoke")
        next(row for row in rows if row.get("rows", 0) > 0)["worst_flat_index"] = 10**9
        self.assertFalse(runner.checker_passed(encoded(rows), "smoke"))

    def test_malformed_json_and_unsuccessful_process_fail(self):
        good = encoded(records("smoke"))
        for text in (good["stdout"][:-10], "[]", "null", "not JSON", '{"host_checks":"passed","device_execution":false,"device_execution":false}'):
            self.assertFalse(runner.checker_passed(record(text), "host-only"))
        for status, code in (("timed_out", None), ("exited", 99), ("interrupted", None)):
            failed = dict(good, status=status, returncode=code)
            self.assertFalse(runner.checker_passed(failed, "smoke"))

    def test_sanitizer_requires_exact_zero_summary_including_warnings(self):
        for tool in ("memcheck", "racecheck", "synccheck"):
            good = ("RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)"
                    if tool == "racecheck" else "ERROR SUMMARY: 0 errors")
            self.assertTrue(runner.sanitizer_clean("========= " + good + "\n", tool))
            for bad in ("", "version only", good + "\nERROR SUMMARY: 1 error", good.replace("0 errors", "10 errors"),
                        "Internal Sanitizer Error: instrumentation failed\n" + good,
                        good + "\n" + good):
                self.assertFalse(runner.sanitizer_clean(bad, tool))
        self.assertFalse(runner.sanitizer_clean("RACECHECK SUMMARY: 1 hazard displayed (0 errors, 1 warning)", "racecheck"))


class RunnerTest(unittest.TestCase):
    def exercise(self, directory, mode="--run-gpu", fail=None):
        output = Path(directory) / "evidence % literal"
        calls = []
        def fake(command, timeout=30):
            values = list(map(str, command))
            calls.append(values)
            if "--build" in values:
                binary = output / "build/rmsnorm_cuda_check"
                binary.parent.mkdir()
                binary.write_bytes(b"fixture binary, never executable")
            if values[0] == "compute-sanitizer" and "--tool" in values:
                tool = values[values.index("--tool") + 1]
                if fail == tool:
                    return encoded(records("smoke"))  # A successful app but missing sanitizer report.
                log = values[values.index("--log-file") + 1].replace("%%", "%")
                summary = ("RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)"
                           if tool == "racecheck" else "ERROR SUMMARY: 0 errors")
                Path(log).write_text("========= " + summary + "\n")
            if fail == "configure" and "-S" in values:
                return record("configure failed", returncode=1)
            if fail == "interrupt" and "--smoke" in values:
                raise preflight.CommandInterrupted(record("partial case", status="interrupted", returncode=None))
            if fail == "smoke" and "--smoke" in values:
                return encoded(records("smoke")[-1:])
            for check in ("host-only", "smoke", "full"):
                if "--" + check in values:
                    return encoded(records(check))
            return record("fixture inventory/build")
        with mock.patch.object(runner, "run_command", side_effect=fake), redirect_stdout(io.StringIO()):
            code = runner.main(["--output", str(output), mode, "--cuda-arch", "86"])
        return code, json.loads((output / "validation.json").read_text()), calls

    def test_host_only_never_runs_device_or_sanitizer_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            code, report, calls = self.exercise(directory, "--host-only")
            self.assertEqual(code, 0)
            self.assertEqual(report["status"], "host_verified")
            self.assertFalse(report["hardware_checks_passed"])
            self.assertTrue(report["checks"]["sources_unchanged"])
            self.assertFalse(any("compute-sanitizer" == c[0] or "--smoke" in c for c in calls))
            with self.assertRaises(FileExistsError):
                self.exercise(directory, "--host-only")

    def test_complete_fixture_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            code, report, calls = self.exercise(directory)
            self.assertEqual(code, 0)
            self.assertEqual(report["status"], "hardware_verified")  # Mocked control flow only.
            self.assertTrue(report["hardware_checks_passed"])
            self.assertEqual(list(report["checks"]), ["configure", "build", "host", "smoke", "memcheck", "racecheck", "synccheck", "full", "sources_unchanged", "binary_unchanged"])
            self.assertTrue(all(report["checks"].values()))
            self.assertEqual(len(report["binary_sha256"]), 64)

    def test_failures_stop_dependent_commands_and_preserve_evidence(self):
        for fail in ("configure", "smoke", "memcheck", "racecheck", "synccheck", "interrupt"):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as directory:
                code, report, calls = self.exercise(directory, fail=fail)
                self.assertEqual(code, 130 if fail == "interrupt" else 2)
                self.assertFalse(report["hardware_checks_passed"])
                self.assertNotIn("full", report["commands"])
                if fail == "configure":
                    self.assertNotIn("build", report["commands"])
                if fail == "interrupt":
                    self.assertEqual(report["commands"]["smoke"]["stdout"], "partial case")
                    self.assertEqual(report["status"], "interrupted")

    def test_deadline_prevents_starting_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(runner.time, "monotonic", side_effect=[0] + [2] * 20), mock.patch.object(runner, "run_command") as run, redirect_stdout(io.StringIO()):
                code = runner.main(["--output", str(Path(directory) / "run"), "--run-gpu", "--cuda-arch", "86", "--max-seconds", "1"])
            run.assert_not_called()
            self.assertEqual(code, 2)

    def test_changed_source_or_binary_cannot_pass(self):
        original = runner.sha256
        for name, check in (("CMakeLists.txt", "sources"), ("rmsnorm_cuda_check", "binary")):
            counts = {}
            def changed(path):
                counts[path] = counts.get(path, 0) + 1
                return "changed" if path.name == name and counts[path] > 1 else original(path)
            with self.subTest(file=name), tempfile.TemporaryDirectory() as directory, mock.patch.object(runner, "sha256", side_effect=changed):
                code, report, _ = self.exercise(directory, "--host-only")
                self.assertEqual(code, 2)
                self.assertEqual(report["status"], "failed_" + check + "_unchanged")


@unittest.skipUnless(os.name == "posix", "POSIX process groups required")
class ProcessCleanupTest(unittest.TestCase):
    def test_timeout_kills_descendants_holding_output_pipes(self):
        child = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); print('child ready',flush=True); time.sleep(60)"
        result = preflight.run_command([sys.executable, "-c", child], timeout=0.3)
        self.assertEqual(result["status"], "timed_out")
        # communicate() could not finish if a descendant still held either output pipe.
        self.assertLess(result["elapsed_seconds"], 5)

    def test_sigint_and_sigterm_kill_children_and_retain_partial_output(self):
        for termination in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(signal=termination), tempfile.TemporaryDirectory() as directory:
                ready = Path(directory) / "ready"
                child = ("import subprocess,sys,time; "
                         "subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
                         "print('partial output',flush=True); "
                         "from pathlib import Path; Path(sys.argv[1]).touch(); time.sleep(60)")
                wrapper = """import json,sys
sys.path.insert(0, sys.argv[1])
from gpu_preflight import run_command, CommandInterrupted, termination_as_interrupt
with termination_as_interrupt():
    try:
        run_command([sys.executable, '-c', sys.argv[2], sys.argv[3]], timeout=60)
    except CommandInterrupted as error:
        print(json.dumps(error.record))
        raise SystemExit(130)
"""
                process = subprocess.Popen([sys.executable, "-c", wrapper, str(SCRIPTS), child, str(ready)],
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
                try:
                    deadline = time.monotonic() + 5
                    while not ready.exists() and time.monotonic() < deadline and process.poll() is None:
                        time.sleep(0.01)
                    self.assertTrue(ready.exists(), "child did not start")
                    process.send_signal(termination)
                    stdout, stderr = process.communicate(timeout=5)
                    self.assertEqual(process.returncode, 130, stderr)
                    result = json.loads(stdout)
                    self.assertEqual(result["status"], "interrupted")
                    self.assertIn("partial output", result["stdout"])
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.communicate(timeout=5)


if __name__ == "__main__":
    unittest.main()
