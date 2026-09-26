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


def _root_package(name: str) -> str:
    return name.split(".", 1)[0]


class _Visitor(ast.NodeVisitor):
    def __init__(self, filename: str, mode: str, rules: dict):
        self.filename = filename
        self.mode = mode
        self.rules = rules
        self.violations: list[str] = []
        # names assigned from recv()/read()/memoryview(...) — potential bytes
        self._tainted: set[str] = set()

    def _err(self, node: ast.AST, msg: str) -> None:
        self.violations.append(f"{self.filename}:{node.lineno}: {msg}")

    def visit_Import(self, node: ast.Import) -> None:
        if self.rules["imports"]:
            for alias in node.names:
                root = _root_package(alias.name)
                if root in FORBIDDEN_PACKAGES:
                    self._err(node, f"import of forbidden package '{root}' (mode {self.mode})")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if self.rules["imports"] and node.level == 0 and node.module:
            root = _root_package(node.module)
            if root in FORBIDDEN_PACKAGES:
                self._err(node, f"import of forbidden package '{root}' (mode {self.mode})")
        self.generic_visit(node)

    def _check_dynamic(self, node: ast.Call, fname: str, arg: ast.AST | None) -> None:
        if not self.rules["dynamic"]:
            return
        if fname in ("__import__", "import_module") and isinstance(arg, ast.Constant) \
                and isinstance(arg.value, str):
            root = _root_package(arg.value)
            if root in FORBIDDEN_PACKAGES:
                self._err(node, f"{fname}() of forbidden package '{root}' (mode {self.mode})")

    def _is_ctypes_call(self, node: ast.Call) -> bool:
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in ("CDLL", "LoadLibrary"):
            value = func.value
            if isinstance(value, ast.Name) and value.id == "ctypes":
                return True
            if isinstance(value, ast.Attribute) and value.attr == "cdll":
                return True
        return False

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name):
            self._check_dynamic(node, func.id, node.args[0] if node.args else None)
        elif isinstance(func, ast.Attribute):
            if func.attr == "import_module":
                base = func.value
                base_name = base.id if isinstance(base, ast.Name) else ""
                if base_name == "importlib" or (isinstance(base, ast.Attribute) and base.attr == "importlib"):
                    self._check_dynamic(node, "import_module", node.args[0] if node.args else None)
        if self._is_ctypes_call(node):
            self._err(node, "ctypes.CDLL/LoadLibrary forbidden in all modes")
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if isinstance(node.value, ast.Call):
            func = node.value.func
            name = ""
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name in ("recv", "read", "recvfrom", "recv_into", "readinto", "memoryview"):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self._tainted.add(target.id)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        if self.rules["byte_loop"]:
            it = node.iter
            hit = False
            if isinstance(it, ast.Name) and it.id in self._tainted:
                hit = True
            elif isinstance(it, ast.Call):
                func = it.func
                name = func.id if isinstance(func, ast.Name) else (
                    func.attr if isinstance(func, ast.Attribute) else "")
                if name in ("recv", "read", "recvfrom", "memoryview"):
                    hit = True
            if hit:
                self._err(node, "per-byte loop over bytes/memoryview "
                                "(use struct.unpack_from) (mode core)")
        self.generic_visit(node)


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
        prog="vb_hotpath_lint", description="BASE-1 §10.5 hot-path lint")
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
    print(f"vb_hotpath_lint: ok (mode={args.mode})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
