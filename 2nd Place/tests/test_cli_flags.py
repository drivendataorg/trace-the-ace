"""Every --flag the shell scripts and README pass exists in the target's argparse."""
import ast
import re

import pytest

from conftest import ROOT

SOURCES = (["scripts/run_training.sh", "scripts/run_inference.sh", "scripts/stage_llm.sh",
            "README.md"]
           + sorted(str(p.relative_to(ROOT)) for p in (ROOT / "scripts/gen_synth").glob("*")
                    if p.suffix in (".sh", ".py")))


def target_path(t):
    return ROOT / t if t.endswith(".py") else ROOT / "src" / (t.replace(".", "/") + ".py")


def defined_flags(path):
    flags = set()
    for n in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "add_argument":
            flags |= {a.value for a in n.args
                      if isinstance(a, ast.Constant) and str(a.value).startswith("-")}
    return flags


def invocations():
    out = []
    for src in SOURCES:
        text = re.sub(r"\\\n\s*", " ", (ROOT / src).read_text(encoding="utf-8"))
        m = re.search(r"LORA_ARGS=\((.*?)\)", text, re.S)
        lora = re.findall(r"--[\w-]+", m.group(1)) if m else []
        for line in text.splitlines():
            for m in re.finditer(
                    r"python\s+(?:-m\s+([\w.]+)|\"?(?:\$(?:REPO|ROOT)/)?((?:src|scripts)/[\w/]+\.py)\"?)(.*)",
                    line):
                rest = re.split(r"\|\||&&|\|(?!\|)|;|`", m.group(3))[0]
                flags = set(re.findall(r"(?<![\w-])(--[a-z][\w-]*)", rest))
                if "LORA_ARGS" in rest:
                    flags |= set(lora)
                out.append((src, m.group(1) or m.group(2), tuple(sorted(flags))))
    return sorted(set(out))


CALLS = invocations()


def test_found_the_pipeline():
    targets = {t for _, t, _ in CALLS}
    assert {"ace.llm.train_lora", "ace.llm.score_vllm", "ace.submission.fit_final_v2",
            "ace.models.train_tabular_v2", "scripts/fit_stack_sub9.py",
            "scripts/compare_synth_holdout.py", "scripts/gen_synth/generate.py",
            "scripts/gen_synth/decontaminate.py"} <= targets


@pytest.mark.parametrize("src,target,flags", CALLS, ids=lambda x: str(x)[:40])
def test_flags_exist(src, target, flags):
    path = target_path(target)
    assert path.is_file(), f"{src}: {target} does not exist"
    missing = set(flags) - defined_flags(path)
    assert not missing, f"{src}: {target} has no {sorted(missing)}"


# ---- shell-level commands in the README (Quick start, Setup, Run *): scripts, env names,
# training stages and VAR=... prefixes must all exist in the files they target
README_SHELL = [l for l in (ROOT / "README.md").read_text(encoding="utf-8").splitlines()
                if re.search(r"\bbash (scripts|envs)/\S+\.sh|(^\s*|`)([A-Z]\w*=\S+\s+)+(pytest|python)\b", l)]


def test_readme_shell_commands_found():
    joined = "\n".join(README_SHELL)
    assert "bash envs/setup_envs.sh inference" in joined
    assert "bash scripts/run_inference.sh" in joined and "bash scripts/run_training.sh" in joined
    assert "ACE_VERIFY_WEIGHTS=1 pytest" in joined and "ACE_OOF_DIR=" in joined


@pytest.mark.parametrize("line", README_SHELL, ids=lambda l: l.strip()[:50])
def test_readme_shell_command_targets_exist(line):
    for part in re.split(r"&&|;", line.split("#")[0] if "`" not in line else line):
        part = part.strip().strip("`")
        m = re.search(r"bash ((?:scripts|envs)/[\w/]+\.sh)\s*(.*)", part)
        if m:
            script, rest = ROOT / m.group(1), m.group(2).split("`")[0]
            assert script.is_file(), f"{m.group(1)} does not exist"
            text = script.read_text(encoding="utf-8")
            args = [a for a in rest.split() if not a.startswith("#")]
            if m.group(1) == "envs/setup_envs.sh":
                for env in args:
                    if env.startswith("<"):
                        continue
                    assert (ROOT / "envs" / f"requirements-{env}.txt").is_file(), f"no env '{env}'"
            if m.group(1) == "scripts/run_training.sh":
                for st in (a for a in args if a.isdigit()):
                    assert re.search(rf"^stage{st}\(\)", text, re.M), f"no training stage {st}"
            pre = re.match(r"((?:[A-Z][A-Z0-9_]*=\S+\s+)+)bash", part)
            for var in re.findall(r"([A-Z][A-Z0-9_]*)=", pre.group(1)) if pre else []:
                assert re.search(rf"\${{?{var}\b", text), f"{m.group(1)} does not read {var}"
        p = re.search(r"((?:[A-Z][A-Z0-9_]*=\S+\s+)+)pytest\b", part)
        if p:
            tests_text = "\n".join(t.read_text(encoding="utf-8") for t in (ROOT / "tests").glob("*.py"))
            for var in re.findall(r"([A-Z][A-Z0-9_]*)=", p.group(1)):
                assert var in tests_text, f"no test reads {var}"
