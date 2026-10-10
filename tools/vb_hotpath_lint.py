#!/usr/bin/env python3
"""vb_hotpath_lint — §10.5 ① static hot-path lint.

Checks Python modules for violations of the BASE-1 layering rules that keep
per-frame computation out of the Python layer. Three modes:

  core  (default)  hot-path modules: ``core-py/vision_base`` and anything that
                   runs per-frame in the base — all four rules apply.
  hook              app ``on_frame`` hooks — imports and ctypes checked, byte
                   loops allowed (hooks only see structured lists).
  dev               ``dev/`` modules only (raw tensors, debug builds) — imports
                   allowed, but every checked path must contain ``dev/``;
                   ctypes is still forbidden.

Exit 1 on any violation, printing ``file:line: message``; exit 0 otherwise.
``tests/`` directories and ``test_*.py`` files are always skipped.

This is a **first-pass static screen, not a proof**. It resolves import
aliases and follows simple intra-module dataflow (literal module names,
``ctypes`` aliases, byte-buffer taint through assignment / slicing /
``memoryview()`` / parameters annotated ``bytes``) so that the obvious
rewordings of a violation are still caught. It cannot see:

  - names crossing function or module boundaries (a buffer passed in from a
    caller without a ``bytes`` annotation is not tainted);
  - module names built at runtime (f-strings, ``os.environ``, computed
    concatenation);
  - numeric loops over a pure-Python list that never touched a byte buffer.

A clean run therefore means "nothing recognisable", not "nothing per-frame".
§10.5 ② (process CPU share) and ③ (py-spy sampling) are the checks that
bound the un-recognised remainder.
"""
from __future__ import annotations

import argparse
import ast
import os
import sys

FORBIDDEN_PACKAGES = frozenset({
    "numpy", "cv2", "PIL", "onnxruntime", "tensorrt", "pycuda",
    "hailo_platform", "rknnlite", "rknn", "torch", "scipy", "gi",
})

# Rules per mode (§10.5 ① table).
RULES = {
    "core": {"imports": True, "dynamic": True, "ctypes": True, "byte_loop": True},
    "hook": {"imports": True, "dynamic": True, "ctypes": True, "byte_loop": False},
    "dev":  {"imports": False, "dynamic": False, "ctypes": True, "byte_loop": False},
}

# Calls that hand back a bytes-like object (socket reads, views, constructors).
# Iterating one of these element by element is the §10.5 ① byte-loop rule.
BYTE_CALLS = frozenset({
    "recv", "recvfrom", "recv_into", "read", "readinto", "readinto1",
    "readline", "readlines", "read_all", "memoryview", "bytes", "bytearray",
})

# Names that load a shared library through ctypes (the member we forbid).
_CTYPES_LOADERS = frozenset({"CDLL", "WinDLL", "OleDLL", "LoadLibrary"})

# A repeated literal at least this large is a hand-rolled array rather than a
# small fixed-size header, so looping over it per element is a hot-path shape.
_MIN_ARRAY_REPEAT = 64


def _root_package(name: str) -> str:
    return name.split(".", 1)[0]


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


class _Visitor(ast.NodeVisitor):
    def __init__(self, filename: str, mode: str, rules: dict):
        self.filename = filename
        self.mode = mode
        self.rules = rules
        self.violations: list[str] = []
        # Names holding a bytes-like object or a hand-rolled large array.
        self._buffers: set[str] = set()
        # Literal module names (``mod = "numpy"``) for dynamic-import checks.
        self._const_str: dict[str, str] = {}
        # Names bound to the ctypes / importlib modules and to members
        # imported out of them (``import ctypes as C``, ``from ctypes import
        # CDLL``, ``importlib.import_module``).
        self._ctypes_mods: set[str] = set()
        self._from_ctypes: set[str] = set()
        self._importlib_mods: set[str] = set()
        self._import_module_fns: set[str] = set()

    def _err(self, node: ast.AST, msg: str) -> None:
        self.violations.append(f"{self.filename}:{node.lineno}: {msg}")

    # ------------------------------------------------------------ taint

    def _is_buffer(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id in self._buffers
        if isinstance(node, ast.Call):
            return _call_name(node) in BYTE_CALLS
        if isinstance(node, ast.Subscript):
            return self._is_buffer(node.value)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            return self._is_repeated_literal(node)
        return False

    @staticmethod
    def _is_repeated_literal(node: ast.BinOp) -> bool:
        """``[0.0] * 100000`` / ``b"\\0" * n`` — a hand-rolled big array."""
        for literal, count in ((node.left, node.right), (node.right, node.left)):
            if not isinstance(literal, (ast.List, ast.Tuple)):
                if not (isinstance(literal, ast.Constant)
                        and isinstance(literal.value, (bytes, bytearray))):
                    continue
            if isinstance(count, ast.Constant) and isinstance(count.value, int) \
                    and not isinstance(count.value, bool):
                if count.value >= _MIN_ARRAY_REPEAT:
                    return True
            else:
                return True  # size unknown at lint time -> treat as an array
        return False

    def _mentions_buffer(self, node: ast.AST) -> bool:
        return any(self._is_buffer(sub) for sub in ast.walk(node))

    def _bind(self, target: ast.AST, value: ast.AST | None) -> None:
        if not isinstance(target, ast.Name) or value is None:
            return
        name = target.id
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            self._const_str[name] = value.value
        elif isinstance(value, ast.Name):
            if value.id in self._const_str:
                self._const_str[name] = self._const_str[value.id]
            if value.id in self._ctypes_mods:
                self._ctypes_mods.add(name)
            if value.id in self._importlib_mods:
                self._importlib_mods.add(name)
        if self._is_buffer(value):
            self._buffers.add(name)

    # ---------------------------------------------------------- imports

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = _root_package(alias.name)
            if self.rules["imports"] and root in FORBIDDEN_PACKAGES:
                self._err(node, f"import of forbidden package '{root}' "
                                f"(mode {self.mode})")
            local = alias.asname or root
            if root == "ctypes":
                self._ctypes_mods.add(local)
            if alias.name == "importlib":
                self._importlib_mods.add(local)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level == 0 and node.module:
            root = _root_package(node.module)
            if self.rules["imports"] and root in FORBIDDEN_PACKAGES:
                self._err(node, f"import of forbidden package '{root}' "
                                f"(mode {self.mode})")
            if node.module == "ctypes":
                self._from_ctypes.update(a.asname or a.name for a in node.names)
            if node.module == "importlib":
                for a in node.names:
                    if a.name == "import_module":
                        self._import_module_fns.add(a.asname or a.name)
        self.generic_visit(node)

    # -------------------------------------------------- dynamic imports

    def _resolved_module(self, arg: ast.AST | None) -> str | None:
        """Literal module name, following ``name = "pkg"`` assignments."""
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value
        if isinstance(arg, ast.Name):
            return self._const_str.get(arg.id)
        return None

    def _check_dynamic(self, node: ast.Call, fname: str,
                       arg: ast.AST | None) -> None:
        if not self.rules["dynamic"]:
            return
        module = self._resolved_module(arg)
        if module is None:
            return
        root = _root_package(module)
        if root in FORBIDDEN_PACKAGES:
            self._err(node, f"{fname}() of forbidden package '{root}' "
                            f"(mode {self.mode})")

    # ------------------------------------------------------------ calls

    def _is_ctypes_call(self, node: ast.Call) -> bool:
        func = node.func
        if isinstance(func, ast.Name):
            return func.id in _CTYPES_LOADERS and func.id in self._from_ctypes
        if isinstance(func, ast.Attribute):
            if func.attr not in _CTYPES_LOADERS:
                return False
            base = func.value
            if isinstance(base, ast.Name):
                return base.id in self._ctypes_mods
            # ctypes.cdll.LoadLibrary
            return (isinstance(base, ast.Attribute) and base.attr == "cdll"
                    and isinstance(base.value, ast.Name)
                    and base.value.id in self._ctypes_mods)
        return False

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        arg = node.args[0] if node.args else None
        if isinstance(func, ast.Name):
            if func.id == "__import__":
                self._check_dynamic(node, func.id, arg)
            elif func.id in self._import_module_fns:
                self._check_dynamic(node, "import_module", arg)
        elif isinstance(func, ast.Attribute) and func.attr == "import_module":
            if isinstance(func.value, ast.Name) \
                    and func.value.id in self._importlib_mods:
                self._check_dynamic(node, "import_module", arg)
        if self._is_ctypes_call(node):
            self._err(node, "ctypes.CDLL/LoadLibrary forbidden in all modes")
        self.generic_visit(node)

    # --------------------------------------------------- assignments

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._bind(target, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        # ``body: bytes`` in a signature: the annotation itself taints.
        if isinstance(node.target, ast.Name) and node.annotation is not None \
                and self._annotation_is_bytes(node.annotation) \
                and not node.target.id.startswith("_"):
            self._buffers.add(node.target.id)
        self._bind(node.target, node.value)
        self.generic_visit(node)

    @staticmethod
    def _annotation_is_bytes(annotation: ast.AST) -> bool:
        names = {n.id for n in ast.walk(annotation) if isinstance(n, ast.Name)}
        names |= {n.attr for n in ast.walk(annotation)
                  if isinstance(n, ast.Attribute)}
        return bool(names & {"bytes", "bytearray", "memoryview"})

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._bind_params(node)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def _bind_params(self, node: ast.FunctionDef) -> None:
        args = node.args
        params = list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
        params += [a for a in (args.vararg, args.kwarg) if a is not None]
        for a in params:
            if a.annotation is not None and self._annotation_is_bytes(a.annotation):
                self._buffers.add(a.arg)

    # ---------------------------------------------------- byte loops

    def _check_iteration(self, node: ast.AST, iterables: list[ast.AST]) -> None:
        if not self.rules["byte_loop"]:
            return
        for it in iterables:
            if self._is_buffer(it):
                self._err(node, "per-byte loop over bytes/memoryview "
                                "(use struct.unpack_from) (mode core)")
                return
            # ``for i in range(len(buf))`` — a per-element index loop over a
            # byte buffer or hand-rolled array: pure-Python numeric work on
            # frame data.
            if isinstance(it, ast.Call) and _call_name(it) == "range" \
                    and any(self._mentions_buffer(a) for a in it.args):
                self._err(node, "per-element index loop over a byte/array "
                                "buffer in Python (move it to the native "
                                "layer) (mode core)")
                return

    def visit_For(self, node: ast.For) -> None:
        self._check_iteration(node, [node.iter])
        self.generic_visit(node)

    visit_AsyncFor = visit_For  # type: ignore[assignment]

    def _visit_comp(self, node: ast.AST) -> None:
        self._check_iteration(node, [g.iter for g in node.generators])  # type: ignore[attr-defined]
        self.generic_visit(node)

    visit_ListComp = _visit_comp
    visit_SetComp = _visit_comp
    visit_GeneratorExp = _visit_comp
    visit_DictComp = _visit_comp


def lint_source(source: str, filename: str, mode: str) -> list[str]:
    """Return a list of ``file:line: message`` violations for one module."""
    violations: list[str] = []
    rel = filename.replace(os.sep, "/")
    if mode == "dev" and "/dev/" not in f"/{rel}/":
        violations.append(f"{filename}:1: dev mode requires modules under a "
                          f"dev/ directory (mode dev)")
        return violations
    rules = RULES[mode]
    tree = ast.parse(source, filename=filename)
    visitor = _Visitor(filename, mode, rules)
    visitor.visit(tree)
    return violations + visitor.violations


def _iter_py_files(paths: list[str]):
    for p in paths:
        if os.path.isfile(p):
            yield p
        elif os.path.isdir(p):
            for root, dirs, files in os.walk(p):
                dirs[:] = [d for d in dirs if d != "tests" and not d.startswith(".")]
                for f in sorted(files):
                    if f.endswith(".py") and not f.startswith("test_"):
                        yield os.path.join(root, f)


def lint_paths(paths: list[str], mode: str) -> list[str]:
    """Lint files/directories; returns all violations (dev-path rule included)."""
    violations: list[str] = []
    checked = 0
    for path in _iter_py_files(paths):
        checked += 1
        rel = path.replace(os.sep, "/")
        if mode == "dev" and "/dev/" not in f"/{rel}/":
            violations.append(f"{path}:1: dev mode requires modules under a dev/ "
                              f"directory (mode dev)")
            continue
        with open(path, "r", encoding="utf-8") as fh:
            source = fh.read()
        violations.extend(lint_source(source, path, mode))
    if not checked:
        violations.append("(no Python files found under: " + ", ".join(paths) + ")")
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vb_hotpath_lint",
        description="BASE-1 §10.5 ① hot-path lint (first-pass static screen, "
                    "not a proof: see the module docstring for what it cannot "
                    "see; §10.5 ② and ③ bound the remainder)",
        epilog="Exit 0 means no *recognised* violation, not that the module "
               "is free of per-frame computation.")
    parser.add_argument("--mode", choices=["core", "hook", "dev"], default="core",
                        help="lint tier (default: core)")
    parser.add_argument("paths", nargs="+", help="Python files or directories")
    args = parser.parse_args(argv)

    violations = lint_paths(args.paths, args.mode)
    for v in violations:
        print(v, file=sys.stderr)
    if violations:
        print(f"vb_hotpath_lint: FAILED ({len(violations)} violation(s), "
              f"mode={args.mode})", file=sys.stderr)
        return 1
    print(f"vb_hotpath_lint: ok (mode={args.mode}; first-pass screen only — "
          f"§10.5 ②③ still apply)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
