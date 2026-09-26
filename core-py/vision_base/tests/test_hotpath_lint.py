"""Tests for tools/vb_hotpath_lint.py (BASE-1 §10.5 ①, M1.14).

Positive/negative cases per mode: core (byte loop flagged), hook (byte loop
allowed, forbidden import flagged), dev (numpy allowed under dev/, flagged
outside).
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile

_TOOLS = os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools")
_spec = importlib.util.spec_from_file_location(
    "vb_hotpath_lint", os.path.join(_TOOLS, "vb_hotpath_lint.py"))
lint = importlib.util.module_from_spec(_spec)
sys.modules["vb_hotpath_lint"] = lint
_spec.loader.exec_module(lint)


def _write(tmpdir: str, rel: str, source: str) -> str:
    path = os.path.join(tmpdir, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(source)
    return path


BYTE_LOOP = (
    "import socket\n"
    "s = socket.socket()\n"
    "data = s.recv(1024)\n"
    "for b in data:\n"
    "    pass\n"
    "for b in memoryview(data):\n"
    "    pass\n"
)
CLEAN_STRUCT = (
    "import struct\n"
    "def parse(buf):\n"
    "    return struct.unpack_from('<I', buf, 0)\n"
)
NUMPY_IMPORT = "import numpy as np\n"
CTYPES_CDLL = "import ctypes\nlib = ctypes.CDLL('librknnrt.so')\n"
DYN_IMPORT = "mod = __import__('cv2')\n"


def test_core_flags_byte_loop():
    assert lint.lint_source(BYTE_LOOP, "x.py", "core"), "core must flag byte loops"


def test_core_clean_struct_passes():
    assert lint.lint_source(CLEAN_STRUCT, "x.py", "core") == []


def test_hook_allows_byte_loop():
    assert lint.lint_source(BYTE_LOOP, "app.py", "hook") == []


def test_hook_flags_forbidden_import():
    v = lint.lint_source(NUMPY_IMPORT, "app.py", "hook")
    assert len(v) == 1 and "numpy" in v[0] and ":1:" in v[0]


def test_hook_flags_ctypes():
    assert any("ctypes" in x for x in lint.lint_source(CTYPES_CDLL, "app.py", "hook"))


def test_hook_flags_dynamic_import():
    assert any("__import__" in x for x in lint.lint_source(DYN_IMPORT, "app.py", "hook"))


def test_dev_allows_numpy_under_dev():
    assert lint.lint_source(NUMPY_IMPORT, "dev/on_tensors.py", "dev") == []


def test_dev_flags_numpy_outside_dev():
    # mode dev over a module NOT under dev/ must error even without imports
    v = lint.lint_source("x = 1\n", "app/hooks.py", "dev")
    assert len(v) == 1 and "dev/" in v[0]


def test_dev_still_flags_ctypes():
    assert any("ctypes" in x for x in lint.lint_source(CTYPES_CDLL, "dev/t.py", "dev"))


def test_path_lint_skips_tests(tmp_path):
    with tempfile.TemporaryDirectory() as td:
        _write(td, "pkg/mod.py", CLEAN_STRUCT)
        _write(td, "pkg/tests/test_mod.py", NUMPY_IMPORT)
        _write(td, "pkg/test_skipme.py", NUMPY_IMPORT)
        assert lint.lint_paths([td], "core") == []


def test_path_lint_dev_directory_ok(tmp_path):
    with tempfile.TemporaryDirectory() as td:
        _write(td, "app/dev/on_tensors.py", NUMPY_IMPORT + CTYPES_CDLL.replace(
            "lib = ctypes.CDLL('librknnrt.so')\n", ""))
        assert lint.lint_paths([os.path.join(td, "app", "dev")], "dev") == []


def test_main_exit_codes(tmp_path):
    with tempfile.TemporaryDirectory() as td:
        good = _write(td, "pkg/mod.py", CLEAN_STRUCT)
        assert lint.main(["--mode", "core", good]) == 0
        bad = _write(td, "pkg/bad.py", BYTE_LOOP)
        assert lint.main(["--mode", "core", bad]) == 1
