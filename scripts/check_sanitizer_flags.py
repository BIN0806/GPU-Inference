"""Audit every first-party C++ translation unit in CMake's compilation database."""
import argparse
import json
from pathlib import Path
import shlex
import sys


def sanitizer_set(arguments):
    enabled = set()
    for argument in arguments:
        if argument.startswith("-fsanitize="):
            enabled.update(argument.split("=", 1)[1].split(","))
        elif argument.startswith("-fno-sanitize="):
            removed = argument.split("=", 1)[1].split(",")
            if "all" in removed:
                enabled.clear()
            else:
                enabled.difference_update(removed)
    return enabled


def audit(source_dir, build_dir, expected):
    source_dir = source_dir.resolve()
    # Enumerate sources independently of the database so omissions fail closed.
    required = {p.resolve() for folder in ("src", "bench", "examples", "tests")
                for p in (source_dir / folder).rglob("*.cpp")}
    if not expected:
        required.discard(source_dir / "tests/sanitizer_probe.cpp")
    if not required:
        raise ValueError("no first-party sources found; check --source-dir")
    entries = json.loads((build_dir / "compile_commands.json").read_text())
    seen = set()
    errors = []
    for entry in entries:
        path = Path(entry["file"])
        if not path.is_absolute():
            path = Path(entry["directory"]) / path
        path = path.resolve()
        if path not in required:
            continue  # GoogleTest is deliberately not a first-party target.
        seen.add(path)
        arguments = entry.get("arguments") or shlex.split(entry["command"])
        actual = sanitizer_set(arguments)
        if actual != expected:
            errors.append(f"{path.relative_to(source_dir)}: expected {sorted(expected)}, got {sorted(actual)}")
        # Group exclusions such as undefined minus alignment are compiler-specific.
        # Reject all disabling overrides instead of claiming to emulate that expansion.
        if expected and any(arg.startswith("-fno-sanitize=") for arg in arguments):
            errors.append(f"{path.relative_to(source_dir)}: sanitizer disabling override is unsupported")
        frame_flags = [arg for arg in arguments if arg in ("-fno-omit-frame-pointer", "-fomit-frame-pointer")]
        if expected and (not frame_flags or frame_flags[-1] != "-fno-omit-frame-pointer"):
            errors.append(f"{path.relative_to(source_dir)}: missing frame-pointer flag")
    errors.extend(f"missing compilation entry: {p.relative_to(source_dir)}" for p in sorted(required - seen))
    if errors:
        raise ValueError("\n".join(errors))
    print(f"Verified {len(seen)} first-party translation units: {','.join(sorted(expected)) or 'no sanitizers'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--expected", choices=("none", "address", "undefined", "address,undefined"), required=True)
    args = parser.parse_args()
    expected = set() if args.expected == "none" else set(args.expected.split(","))
    try:
        audit(args.source_dir, args.build_dir, expected)
    except (OSError, ValueError, KeyError) as error:
        print(f"Sanitizer flag audit failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
