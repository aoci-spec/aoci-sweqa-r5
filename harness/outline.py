#!/usr/bin/env python3
"""Deterministic file outline for AOCI authoring.

Gives an index author what a careful engineer skims first, at bounded cost:
line count, module docstring, in-repo imports, top-level classes/functions
with their first docstring line, and __all__. Full reads stay the author's
call for high-importance files. Non-Python files get a bounded head.

Usage: outline.py <repo_root> <rel_path> [<rel_path> ...]
"""
import ast
import os
import sys

HEAD_LINES = int(os.environ.get("OUTLINE_HEAD", "40"))


def py_outline(src, path):
    out = []
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f"  (syntax error: {e}) — falling back to head"] + head_lines(src)
    doc = ast.get_docstring(tree)
    if doc:
        out.append("  doc: " + doc.strip().splitlines()[0][:200])
    imports = []
    for n in tree.body:
        if isinstance(n, ast.ImportFrom) and n.module and (n.module.startswith("xarray") or n.level):
            mod = ("." * n.level) + (n.module or "")
            imports.append(f"{mod}:{','.join(a.name for a in n.names)[:80]}")
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name.startswith("xarray"):
                    imports.append(a.name)
    if imports:
        out.append("  in-repo imports: " + "; ".join(imports)[:600])
    for n in tree.body:
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id == "__all__":
                    try:
                        names = [e.value for e in n.value.elts][:30]
                        out.append("  __all__: " + ", ".join(map(str, names)))
                    except Exception:
                        pass
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            kind = "class" if isinstance(n, ast.ClassDef) else "def"
            d = ast.get_docstring(n)
            d1 = (" — " + d.strip().splitlines()[0][:110]) if d else ""
            bases = ""
            if isinstance(n, ast.ClassDef) and n.bases:
                bases = "(" + ",".join(getattr(b, "id", getattr(b, "attr", "?")) for b in n.bases)[:60] + ")"
            out.append(f"  {kind} {n.name}{bases}{d1}")
            if isinstance(n, ast.ClassDef):
                meths = [m.name for m in n.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and not m.name.startswith("_")]
                if meths:
                    out.append("      methods: " + ", ".join(meths[:25]) + (" …" if len(meths) > 25 else ""))
    return out


def head_lines(src, n=HEAD_LINES):
    lines = src.splitlines()
    out = ["  " + l[:160] for l in lines[:n]]
    if len(lines) > n:
        out.append(f"  … ({len(lines) - n} more lines)")
    return out


def line_count(root, rel):
    try:
        with open(os.path.join(root, rel), "rb") as f:
            raw = f.read()
    except OSError:
        return 0
    src = raw.decode("utf-8", "replace")
    return src.count("\n") + (0 if src.endswith("\n") or not src else 1)


def outline_for(root, rel, max_lines=None):
    """Outline text for one file (what the index author sees)."""
    p = os.path.join(root, rel)
    try:
        with open(p, "rb") as f:
            raw = f.read()
    except OSError as e:
        return f"== {rel}: {e}"
    if b"\x00" in raw[:8000]:
        return f"== {rel}  [binary, {len(raw)} bytes]"
    src = raw.decode("utf-8", "replace")
    n = src.count("\n") + (0 if src.endswith("\n") or not src else 1)
    body = py_outline(src, rel) if rel.endswith(".py") else head_lines(src)
    if max_lines and len(body) > max_lines:
        body = body[:max_lines] + [f"  … ({len(body) - max_lines} more outline lines)"]
    return f"== {rel}  [{n} lines]\n" + "\n".join(body)


def main():
    root = sys.argv[1]
    for rel in sys.argv[2:]:
        print(outline_for(root, rel))
        print()


if __name__ == "__main__":
    main()
