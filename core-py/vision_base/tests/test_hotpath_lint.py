"""Tests for tools/vb_hotpath_lint.py (BASE-1 §10.5 ①, M1.14).

Positive/negative cases per mode: core (byte loop flagged), hook (byte loop
allowed, forbidden import flagged), dev (numpy allowed under dev/, flagged
outside).

Plus the review follow-up: the lint is a *screen*, so the obvious rewordings
of a violation must not slip through — a variable module name, a ``ctypes``
import alias, byte taint carried through assignment / slicing / a ``bytes``
annotation, comprehensions over a byte buffer, and index loops over a byte
buffer or a hand-rolled array.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile

import pytest

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


# --------------------------------------------- aliases and dynamic imports

VAR_MODULE_IMPORT = "mod = 'numpy'\n__import__(mod)\n"
VAR_IMPORTLIB = (
    "import importlib\n"
    "pkg = 'cv2'\n"
    "importlib.import_module(pkg)\n"
)
VAR_MODULE_IMPORT_CHAIN = "a = 'torch'\nb = a\n__import__(b)\n"
VAR_MODULE_CLEAN = "mod = 'json'\n__import__(mod)\n"


def test_variable_module_name_is_resolved():
    assert any("numpy" in v for v in lint.lint_source(VAR_MODULE_IMPORT, "x.py", "core"))


def test_variable_module_name_importlib_alias():
    assert any("cv2" in v for v in lint.lint_source(VAR_IMPORTLIB, "x.py", "core"))


def test_variable_module_name_through_assignment_chain():
    assert any("torch" in v
               for v in lint.lint_source(VAR_MODULE_IMPORT_CHAIN, "x.py", "core"))


def test_variable_module_name_allowed_package_passes():
    assert lint.lint_source(VAR_MODULE_CLEAN, "x.py", "core") == []


def test_import_module_imported_directly_is_checked():
    src = "from importlib import import_module\nimport_module('onnxruntime')\n"
    assert any("onnxruntime" in v for v in lint.lint_source(src, "x.py", "core"))


CTYPES_ALIAS = "import ctypes as C\nlib = C.CDLL('librknnrt.so')\n"
CTYPES_FROM_IMPORT = "from ctypes import CDLL\nlib = CDLL('librknnrt.so')\n"
CTYPES_CDLL_LOADLIB = "import ctypes\nlib = ctypes.cdll.LoadLibrary('x.so')\n"
CTYPES_ALIAS_CLEAN = "import ctypes.util as u\nu.find_library('z')\n"


def test_ctypes_import_alias_is_flagged():
    assert any("ctypes" in v for v in lint.lint_source(CTYPES_ALIAS, "x.py", "core"))


def test_ctypes_from_import_is_flagged():
    assert any("ctypes" in v
               for v in lint.lint_source(CTYPES_FROM_IMPORT, "x.py", "core"))


def test_ctypes_cdll_loadlibrary_is_flagged():
    assert any("ctypes" in v
               for v in lint.lint_source(CTYPES_CDLL_LOADLIB, "x.py", "core"))


def test_ctypes_non_loader_call_is_clean():
    assert lint.lint_source(CTYPES_ALIAS_CLEAN, "x.py", "core") == []


def test_ctypes_alias_still_flagged_in_dev_mode():
    assert any("ctypes" in v for v in lint.lint_source(CTYPES_ALIAS, "dev/t.py", "dev"))


# --------------------------------------------------- byte taint propagation

BYTE_ALIAS = (
    "import socket\n"
    "s = socket.socket()\n"
    "buf = s.recv(1024)\n"
    "copy = buf\n"
    "for b in copy:\n"
    "    pass\n"
)
BYTE_SLICE = (
    "import socket\n"
    "s = socket.socket()\n"
    "data = s.recv(1024)\n"
    "head = data[:4]\n"
    "for b in head:\n"
    "    pass\n"
)
BYTE_VIEW = (
    "import socket\n"
    "s = socket.socket()\n"
    "data = s.recv(1024)\n"
    "view = memoryview(data)\n"
    "for b in view:\n"
    "    pass\n"
)
BYTE_PARAM = "def parse(body: bytes):\n    for b in body:\n        pass\n"
BYTE_PARAM_CLEAN = "def parse(rows: list):\n    for r in rows:\n        pass\n"


def test_byte_taint_follows_assignment():
    assert lint.lint_source(BYTE_ALIAS, "x.py", "core"), "alias not tainted"


def test_byte_taint_follows_slice():
    assert lint.lint_source(BYTE_SLICE, "x.py", "core"), "slice not tainted"


def test_byte_taint_follows_memoryview():
    assert lint.lint_source(BYTE_VIEW, "x.py", "core"), "memoryview not tainted"


def test_bytes_annotated_parameter_is_tainted():
    assert lint.lint_source(BYTE_PARAM, "x.py", "core")


def test_non_bytes_annotated_parameter_is_clean():
    assert lint.lint_source(BYTE_PARAM_CLEAN, "x.py", "core") == []


def test_hook_mode_allows_byte_loops_but_keeps_ctypes():
    assert lint.lint_source(BYTE_VIEW, "app.py", "hook") == []
    assert lint.lint_source(CTYPES_ALIAS, "app.py", "hook")


# ------------------------------------------------------- comprehensions

BYTE_COMP = (
    "import socket\n"
    "s = socket.socket()\n"
    "data = s.recv(64)\n"
    "total = [b for b in data]\n"
)
BYTE_COMP_CLEAN = "rows = []\ntotal = [r * 2 for r in rows]\n"


def test_comprehension_over_bytes_is_flagged():
    assert any("per-byte" in v for v in lint.lint_source(BYTE_COMP, "x.py", "core"))


def test_comprehension_over_list_is_clean():
    assert lint.lint_source(BYTE_COMP_CLEAN, "x.py", "core") == []


# ------------------------------------------- pure-Python numeric loops

INDEX_LOOP = (
    "import socket\n"
    "s = socket.socket()\n"
    "data = s.recv(1024)\n"
    "acc = 0\n"
    "for i in range(len(data)):\n"
    "    acc += data[i]\n"
)
BIG_ARRAY = (
    "arr = [0.0] * 100000\n"
    "for i in range(len(arr)):\n"
    "    arr[i] = 1.0\n"
)
BIG_ARRAY_ITER = "frame = bytearray(4096)\nfor v in frame:\n    pass\n"
SMALL_HEADER = "hdr = [0] * 4\nfor i in range(len(hdr)):\n    pass\n"
DET_LIST_LOOP = (
    "dets = []\n"
    "for i in range(len(dets)):\n"
    "    pass\n"
)


def test_index_loop_over_byte_buffer_is_flagged():
    assert any("index loop" in v for v in lint.lint_source(INDEX_LOOP, "x.py", "core"))


def test_index_loop_over_hand_rolled_array_is_flagged():
    assert any("index loop" in v for v in lint.lint_source(BIG_ARRAY, "x.py", "core"))


def test_iteration_over_bytearray_is_flagged():
    assert any("per-byte" in v for v in lint.lint_source(BIG_ARRAY_ITER, "x.py", "core"))


def test_small_fixed_size_header_array_is_clean():
    assert lint.lint_source(SMALL_HEADER, "x.py", "core") == []


def test_index_loop_over_structured_list_is_clean():
    """§10.5 allows once-per-frame scalar conversion of detection lists."""
    assert lint.lint_source(DET_LIST_LOOP, "x.py", "core") == []


# --------------------------------------------------------- help wording

def test_help_states_that_it_is_a_screen(capsys):
    with pytest.raises(SystemExit) as exc:
        lint.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "first-pass static screen" in out
    assert "not a proof" in out
