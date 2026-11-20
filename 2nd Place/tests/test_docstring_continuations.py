"""Command examples in docstrings keep their shell line continuations.

For every .py under src/ and scripts/ (except the modules frozen in inference/assets/ace/):
a docstring line starting with "python " or "bash " (optionally after VAR=value prefixes) that
is followed by an indented "--" line must end with "\\", and no backslash-newline may have been
swallowed by a non-raw docstring (which joins the lines into one with a run of spaces).
"""
import ast
import re

import pytest

from conftest import ASSETS, ROOT

CMD = re.compile(r"^\s*(?:[A-Z_][A-Z0-9_]*=\S+\s+)*(?:python|bash)\s")
JOINED = re.compile(r"(?:python|bash)\s.*\S {3,}--")
FROZEN = {f"src/{p.relative_to(ASSETS).as_posix()}" for p in (ASSETS / "ace").rglob("*.py")}
FROZEN.add("src/ace/submission/main.py")
FILES = sorted(p.relative_to(ROOT).as_posix() for d in ("src", "scripts")
               for p in (ROOT / d).rglob("*.py")
               if p.relative_to(ROOT).as_posix() not in FROZEN and "__pycache__" not in p.parts)


def docstrings(path):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(n, clean=False)
            if doc:
                yield getattr(n, "name", "module"), doc


def test_scope():
    assert len(FILES) > 20 and "src/ace/llm/train_lora.py" in FILES
    assert not FROZEN & set(FILES)


@pytest.mark.parametrize("path", FILES)
def test_command_examples_keep_continuations(path):
    problems = []
    for name, doc in docstrings(path):
        lines = doc.splitlines()
        for i, line in enumerate(lines[:-1]):
            follows = lines[i + 1].lstrip().startswith("--")
            in_cmd = CMD.match(line) or (line.lstrip().startswith("--") and i and lines[i - 1].rstrip().endswith("\\"))
            if in_cmd and follows and not line.rstrip().endswith("\\"):
                problems.append(f"{name}: no '\\' at the end of: {line.strip()[:70]}")
        for line in lines:
            if JOINED.search(line):
                problems.append(f"{name}: continuation swallowed (make the docstring raw): {line.strip()[:70]}")
    assert not problems, "\n".join(problems)
