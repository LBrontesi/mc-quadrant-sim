import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest


def test_native_random_distributions_and_streams(tmp_path):
    compiler = shlex.split(os.environ.get("MC_CXX", "c++"))
    if not compiler or shutil.which(compiler[0]) is None:
        pytest.skip("C++ compiler is unavailable")
    root = Path(__file__).resolve().parents[1]
    executable = tmp_path / "native-random-check"
    subprocess.run(
        [*compiler, "-O3", "-std=c++17", "-pthread",
         f"-I{root / 'src/mc_quadrants'}",
         str(root / "tests/native_random_check.cpp"), "-o", str(executable)],
        check=True, capture_output=True, text=True,
    )
    subprocess.run([str(executable)], check=True, capture_output=True, text=True, timeout=30)
