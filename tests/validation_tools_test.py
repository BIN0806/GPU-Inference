"""Regression cases for false-green validation, without executing undefined behavior."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


def load(relative):
    path = Path(__file__).resolve().parents[1] / relative
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


flags = load("scripts/check_sanitizer_flags.py")
runtime = load("tests/sanitizer_runtime_test.py")


class ValidationToolsTest(unittest.TestCase):
    def test_runtime_rejects_generic_crash_timeout_and_recovered_fault(self):
        marker = "AddressSanitizer: heap-buffer-overflow"
        record = dict(returncode=1, stdout="", stderr=marker, timed_out=False)
        self.assertTrue(runtime.expected_result(record, "heap-overflow", marker))
        for changes in (dict(stderr="segmentation fault", returncode=-11),
                        dict(returncode=0), dict(timed_out=True, returncode=None)):
            with self.subTest(changes=changes):
                self.assertFalse(runtime.expected_result(record | changes, "heap-overflow", marker))

    def test_clean_probe_requires_success_and_marker(self):
        record = dict(returncode=0, stdout="clean marker", stderr="", timed_out=False)
        self.assertTrue(runtime.expected_result(record, "clean", "clean marker"))
        self.assertFalse(runtime.expected_result(record | dict(returncode=1), "clean", "clean marker"))
        self.assertFalse(runtime.expected_result(record | dict(stdout=""), "clean", "clean marker"))

    def test_compile_audit_rejects_missing_and_overridden_instrumentation(self):
        with tempfile.TemporaryDirectory(prefix="kvlab audit ") as directory:
            root = Path(directory)
            (root / "src").mkdir()
            source = root / "src/example.cpp"
            source.touch()
            database = root / "compile_commands.json"
            entry = dict(directory=str(root), file="src/example.cpp",
                         arguments=["c++", "-fsanitize=address", "-fno-omit-frame-pointer"])
            database.write_text(json.dumps([entry]))
            flags.audit(root, root, {"address"})
            for extra in ("-fno-sanitize=address", "-fno-sanitize=alignment", "-fomit-frame-pointer"):
                database.write_text(json.dumps([entry | dict(arguments=entry["arguments"] + [extra])]))
                with self.subTest(extra=extra), self.assertRaises(ValueError):
                    flags.audit(root, root, {"address"})
            for entries in ([], [entry | dict(arguments=["c++"])],
                            [entry | dict(arguments=["c++", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"]) ]):
                database.write_text(json.dumps(entries))
                with self.subTest(entries=entries), self.assertRaises(ValueError):
                    flags.audit(root, root, {"address"})
            database.write_text(json.dumps([entry]))
            with self.assertRaises(ValueError):
                flags.audit(root, root, set())
            source.unlink()
            with self.assertRaises(ValueError):
                flags.audit(root, root, set())


if __name__ == "__main__":
    unittest.main()
