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
comparison = load("scripts/compare_allocator_variants.py")


class ValidationToolsTest(unittest.TestCase):
    def test_variant_comparison_rejects_changed_work_and_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = [Path(directory) / name for name in ("baseline", "candidate")]
            summary = dict(schema_version=2, measurement="cpu_metadata_event", config={},
                           kv_model={}, warmup_events=0, outcomes={"admitted": 1},
                           live_pages_before_drain=1, free_pages_after_drain=1,
                           build_type="Release")
            events = ("index,kind,slot,tokens,share_draw,outcome,elapsed_ns,metadata_allocations\n"
                      "0,admit,0,1,0,admitted,42,1\n")
            for path in (first, second):
                path.mkdir()
                (path / "summary.json").write_text(json.dumps(summary))
                (path / "events.csv").write_text(events)
                (path / "memory.csv").write_text("same physical accounting\n")
            # Timing/allocation differences are allowed, but not different accepted work.
            (second / "events.csv").write_text(events.replace(",42,1\n", ",17,0\n"))
            comparison.compare_outputs(first, second)
            (second / "events.csv").write_text(events.replace(",admitted,", ",skipped,"))
            with self.assertRaises(ValueError):
                comparison.compare_outputs(first, second)
            (second / "events.csv").write_text(events)
            (second / "memory.csv").write_text("different accounting\n")
            with self.assertRaises(ValueError):
                comparison.compare_outputs(first, second)
            (second / "memory.csv").write_text("same physical accounting\n")
            (second / "summary.json").write_text(json.dumps(summary | dict(config={"events": 2})))
            with self.assertRaises(ValueError):
                comparison.compare_outputs(first, second)

    def test_variant_comparison_rejects_instrumented_release_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "allocator_bench"
            binary.write_text("fixture only")
            (root / "CMakeCache.txt").write_text("CMAKE_BUILD_TYPE:STRING=Release\n")
            database = root / "compile_commands.json"
            database.write_text(json.dumps([dict(arguments=["c++", "-O3"])]))
            comparison.inspect_build(binary)
            database.write_text(json.dumps([dict(arguments=["c++", "-O3", "-fsanitize=address"])]))
            with self.assertRaises(ValueError):
                comparison.inspect_build(binary)

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
