"""Run the published native RMSNorm checker on one Modal L4."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import modal


# The first hardware result should be attributable to the host-validated control.
REVISION = "48ed5301e1a9344f6326b6169e936c24add184a0"
BASE = "nvidia/cuda@sha256:4b9ed5fa8361736996499f64ecebf25d4ec37ff56e4d11323ccde10aa36e0c43"
ROOT = Path("/opt/kvlab")
image = (
    modal.Image.from_registry(BASE, add_python="3.12")
    .entrypoint([])
    .apt_install("ca-certificates", "git", "g++", "cmake", "ninja-build")
    .run_commands(
        f"git init -q {ROOT}",
        f"git -C {ROOT} fetch --depth 1 https://github.com/BIN0806/GPU-Inference.git {REVISION}",
        f"git -C {ROOT} checkout --detach FETCH_HEAD",
        "nvcc --version && compute-sanitizer --version && cmake --version",
    )
)
app = modal.App("kvlab-native-validation")


@app.function(image=image, gpu="L4", cpu=1.0, memory=4096, max_containers=1,
              retries=0, timeout=720, scaledown_window=5)
def validate() -> tuple[int, bytes]:
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "session"
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/check_rmsnorm_cuda.py"),
             "--run-gpu", "--cuda-arch", "89", "--max-seconds", "600",
             "--output", str(output)],
            cwd=ROOT, check=False,
        )
        # A failed gate can produce the most useful report of the session.
        return result.returncode, (output / "validation.json").read_bytes()


@app.local_entrypoint()
def main(output: str):
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    code, contents = validate.remote()
    report_file = destination / "validation.json"
    report_file.write_bytes(contents)
    report = json.loads(contents)
    if (code != 0 or report.get("status") != "hardware_verified"
            or report.get("hardware_checks_passed") is not True
            or report["commands"]["git"]["stdout"].strip() != REVISION):
        raise RuntimeError(f"Native L4 validation failed for {REVISION} (native exit {code}). See {report_file}")
    print(f"Native L4 validation passed for {REVISION}. Evidence: {report_file}")
