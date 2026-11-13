"""Code checks of the general items (README "Evals").

`python_tests`: the code of an answer is run against the item's tests in a child process (a temp folder, a clean
environment, time, memory and file-size limits, no network). The convention is the item files': run the answer's
code, then `check.tests` (plain asserts) in the same namespace; `check.entry_point` is the name the tests call;
with `check.language: sql` the tests expect the answer's SQL statement in a string variable `sql`.

`exact` and `numeric`: quick signals. The reference value is looked for anywhere in the free text of the answer;
no answer format is asked for or read, and the signal says "found", never "right": the review decides.

Every rule here is the same for every arm, so a paired difference stays fair even where a rule misreads an answer.
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any

from haisenberg.verifier import read_numbers

from evals.items import check_values

PYTHON_TAGS = ("python", "py", "python3")
SQL_START = re.compile(r"\s*(?:SELECT|WITH|INSERT|UPDATE|DELETE|CREATE)\b", re.I)
SQL_CANDIDATES = 4  # at most this many SQL blocks are tried, from the last to the first
FILE_MB = 32  # largest file the checked code may write
DETAIL_CHARS = 300
# A fenced block: an opening fence of three or more ` or ~ with an optional language tag, up to the same fence on a
# line of its own, or to the end of the text when it is never closed.
_FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})[ \t]*\{?\.?([\w+#-]*)[^\n]*\n(.*?)(?:^[ \t]*\1[ \t]*$|\Z)", re.S | re.M)

# Runs in the child process, inside the temp folder. It writes result.json: the checked code's own output can
# then never be mistaken for the result.
DRIVER = r'''
import json, os, sys, traceback

payload = json.load(open("payload.json", encoding="utf-8"))
result = {"status": "error", "detail": "", "network": "guard", "skipped": 0}

try:  # limits, where the system has them
    import resource
    limits = payload["limits"]
    resource.setrlimit(resource.RLIMIT_CPU, (limits["cpu_s"], limits["cpu_s"] + 1))
    resource.setrlimit(resource.RLIMIT_FSIZE, (limits["file_bytes"], limits["file_bytes"]))
    if limits["memory_bytes"]:
        resource.setrlimit(resource.RLIMIT_AS, (limits["memory_bytes"], limits["memory_bytes"]))
except Exception:
    pass
try:  # no network: a network namespace of its own, where the system allows one
    os.unshare(os.CLONE_NEWUSER | os.CLONE_NEWNET)
    result["network"] = "namespace"
except Exception:
    import socket

    def _blocked(*args, **kwargs):
        raise OSError("network disabled for this check")

    class _NoSocket(socket.socket):
        def __init__(self, *args, **kwargs):
            _blocked()

    socket.socket, socket.create_connection, socket.getaddrinfo = _NoSocket, _blocked, _blocked

# Files and processes: while the answer's code and the tests run, an audit hook refuses any write, delete, rename
# or permission change outside this temp folder (the eval machine's own files stay as they are, e.g. /mnt/c in
# WSL) and any new or signalled process. A guard against accidents, not a jail: code that means to escape can.
sys.dont_write_bytecode = True
ROOT = os.path.realpath(os.getcwd())
GUARD = {"on": False}
WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
PATH_EVENTS = {"os.remove": (0,), "os.rmdir": (0,), "os.mkdir": (0,), "os.chmod": (0,), "os.chown": (0,),
               "os.truncate": (0,), "os.utime": (0,), "os.rename": (0, 1), "os.link": (1,), "os.symlink": (1,),
               "shutil.rmtree": (0,), "shutil.move": (0, 1), "shutil.copyfile": (1,), "shutil.copytree": (1,),
               "sqlite3.connect": (0,)}
PROCESS_EVENTS = {"subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork", "os.forkpty",
                  "os.kill", "os.killpg", "os.startfile"}


def _inside(path):
    if isinstance(path, int) or path in (os.devnull, ":memory:", ""):
        return True  # an open file descriptor (checked when it was opened), the null device, an in-memory database
    try:
        full = os.path.realpath(os.fsdecode(path))
    except Exception:
        return False
    return full == ROOT or full.startswith(ROOT + os.sep)


def _guard(event, args):
    if not GUARD["on"]:
        return
    if event == "open":
        path, mode, flags = (tuple(args) + (None, None, None))[:3]
        writes = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
            isinstance(flags, int) and flags & WRITE_FLAGS)
        if writes and not _inside(path):
            raise PermissionError(f"the checked code may write only in its own temp folder, not {path!r}")
    elif event in PATH_EVENTS:
        if any(i < len(args) and not _inside(args[i]) for i in PATH_EVENTS[event]):
            raise PermissionError(f"the checked code may change files only in its own temp folder ({event})")
    elif event in PROCESS_EVENTS:
        raise PermissionError(f"the checked code may not start or signal processes ({event})")


sys.addaudithook(_guard)


def tests_pass(namespace):
    try:
        exec(compile(payload["tests"], "tests.py", "exec"), namespace)
        return True
    except BaseException as e:
        frames = [f for f in traceback.extract_tb(e.__traceback__) if f.filename == "tests.py"]
        line = payload["tests"].splitlines()[frames[-1].lineno - 1].strip() if frames else ""
        where = f"tests line {frames[-1].lineno} ({line}): " if frames else ""
        result["detail"] = f"{where}{type(e).__name__}: {e}"
        return False


GUARD["on"] = True
if payload["language"] == "sql":
    passed = any(tests_pass({"__name__": "tests", "sql": block}) for block in payload["blocks"])
else:
    namespace = {"__name__": "solution"}
    for n, block in enumerate(payload["blocks"], 1):
        try:
            exec(compile(block, f"answer_block_{n}.py", "exec"), namespace)
        except BaseException:  # an example that shows an error, a snippet that is not code, sys.exit()
            result["skipped"] += 1
    passed = tests_pass(namespace)
GUARD["on"] = False
result["status"] = "passed" if passed else "failed"
if passed:
    result["detail"] = ""
with open("result.json", "w", encoding="utf-8") as f:
    json.dump(result, f)
if hasattr(os, "killpg"):  # end whatever the checked code left running in this process group
    import signal
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    os.killpg(0, signal.SIGTERM)
'''


def fenced(text: str) -> list[tuple[str, str]]:
    """(language tag in lower case, code without the block's indentation) of every fenced block of the text."""
    return [(m.group(2).lower(), textwrap.dedent(m.group(3))) for m in _FENCE.finditer(text)]


def _defines(code: str, name: str) -> bool:
    """The code parses as Python and defines `name` at its top level (a function, a class or an assignment)."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            return True
        targets = node.targets if isinstance(node, ast.Assign) else \
            [node.target] if isinstance(node, (ast.AnnAssign, ast.AugAssign)) else []
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            return True
    return False


def code_blocks(answer: str, check: dict[str, Any]) -> list[str]:
    """The code of an answer for a python_tests check, [] when it holds none. Python: the fenced blocks tagged as
    Python; without one, the untagged fenced blocks that define the entry point; without any fence, the whole
    answer when it does. SQL: the fenced blocks whose tag contains `sql`; without one, the untagged blocks that
    start with an SQL keyword; the last SQL_CANDIDATES of them, the last first."""
    blocks = fenced(answer)
    if check.get("language", "python") == "sql":
        found = [code for tag, code in blocks if "sql" in tag] or \
                [code for tag, code in blocks if not tag and SQL_START.match(code)]
        return [code.strip() for code in found[-SQL_CANDIDATES:][::-1]]
    entry = check["entry_point"]
    return [code for tag, code in blocks if tag in PYTHON_TAGS] or \
           [code for tag, code in blocks if not tag and _defines(code, entry)] or \
           ([answer] if not blocks and _defines(answer, entry) else [])


def _kill(proc: subprocess.Popen) -> None:
    """End a child that is still running, and everything it started."""
    try:
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, signal.SIGKILL)  # the child leads its own process group (start_new_session)
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError):
        pass
    proc.wait()


def python_tests(answer: str, check: dict[str, Any], timeout_s: float, memory_mb: int) -> dict[str, Any]:
    """Run the item's tests against the code of an answer: `status` passed | failed | no_code | timeout | error,
    `passed`, `detail` (one line), `seconds` and `network` (namespace | guard: how the child was kept offline;
    None when nothing ran)."""
    blocks = code_blocks(answer, check)
    if not blocks:
        return {"status": "no_code", "passed": False, "detail": "", "seconds": 0.0, "network": None}
    payload = {"language": check.get("language", "python"), "blocks": blocks, "tests": check["tests"],
               "limits": {"cpu_s": math.ceil(timeout_s) + 1, "file_bytes": FILE_MB * 2**20,
                          "memory_bytes": memory_mb * 2**20}}
    with tempfile.TemporaryDirectory(prefix="hai-check-") as tmp:
        folder = Path(tmp)
        (folder / "payload.json").write_text(json.dumps(payload), encoding="utf-8")
        (folder / "driver.py").write_text(DRIVER, encoding="utf-8")
        env = {k: os.environ[k] for k in ("PATH", "SYSTEMROOT") if k in os.environ}
        env |= {"HOME": tmp, "TMPDIR": tmp, "TEMP": tmp, "TMP": tmp}
        started, timed_out = time.monotonic(), False
        with (folder / "stderr.txt").open("wb") as err:
            proc = subprocess.Popen([sys.executable, "-I", "-X", "utf8", "driver.py"], cwd=tmp, env=env,
                                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=err,
                                    start_new_session=True)
            try:
                proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                _kill(proc)
        seconds = round(time.monotonic() - started, 2)
        if timed_out:
            return {"status": "timeout", "passed": False, "detail": f"no result within {timeout_s:g} s",
                    "seconds": seconds, "network": None}
        try:
            result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            tail = (folder / "stderr.txt").read_text(encoding="utf-8", errors="replace").strip().splitlines()
            why = f"the check ended without a result (exit code {proc.returncode})" + (f": {tail[-1]}" if tail else "")
            return {"status": "error", "passed": False, "detail": why[:DETAIL_CHARS], "seconds": seconds,
                    "network": None}
    return {"status": result["status"], "passed": result["status"] == "passed",
            "detail": " ".join(str(result["detail"]).split())[:DETAIL_CHARS], "seconds": seconds,
            "network": result["network"]}


# --- reference value signals ----------------------------------------------------------------------------

_FRACTION = re.compile(r"(?<![\w.,/])(-?)[ \t]*(\d+)[ \t]*/[ \t]*(\d+)(?![\w/]|[.,]\d)")
_LATEX_FRACTION = re.compile(r"(-?)[ \t]*\\[dt]?frac[ \t]*\{[ \t]*(-?\d+(?:\.\d+)?)[ \t]*\}"
                             r"[ \t]*\{[ \t]*(\d+(?:\.\d+)?)[ \t]*\}")
_PERCENT = re.compile(r"[ \t\xa0]*\\?%")
# A scale after a number (language data, English and Russian): a word after at most one space, or a suffix letter
# right after the digits. `80 тыс.`, `80k`, `1.2 million`, `1.3B`.
_SCALES = tuple((re.compile(rf"[ \t\xa0]?(?:{words})(?![^\W\d_])", re.I), factor) for words, factor in (
    (r"thousands?|тыс[^\W\d_]*", 1e3), (r"millions?|mln|mn|млн|миллион[^\W\d_]*", 1e6),
    (r"billions?|bn|bln|млрд|миллиард[^\W\d_]*", 1e9), (r"trillions?|трлн|триллион[^\W\d_]*", 1e12)))
_SUFFIXED = re.compile(r"(?<![\w.,])(-?\d+(?:\.\d+)?)([kKMB])(?!\w)")  # the reader stops at a letter: read apart
_SUFFIXES = {"k": 1e3, "K": 1e3, "M": 1e6, "B": 1e9}
_DIGITS = str.maketrans("₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹", "01234567890123456789")
_GROUP = "[,\u00a0\u2009\u202f ]"  # what may separate digit groups: a comma or a (thin, no-break) space


def _body(answer: str) -> str:
    """The answer without the source list of the annotated style."""
    return answer.split("\n\nSources:\n", 1)[0]


def numbers_in(text: str) -> list[float]:
    """Every value a number of the text may stand for: as written with digit groups, decimal points and powers of
    ten (the verifier's reader), read once with English commas and once with decimal commas; fractions `a/b` and
    `\\frac{a}{b}`; a number followed by % also as its hundredth; and a number followed by a scale (`80 тыс.`,
    `80k`, `1.2 million`) also as that multiple. Liberal on purpose: every reading counts."""
    text = _body(text).replace("−", "-").replace("_", " ")
    readings = (re.sub(r"(\d),(\d{1,2}(?!\d)|\d{4,})", r"\1.\2", text), re.sub(r"(\d),(\d)", r"\1.\2", text))
    values: list[float] = []
    for reading in readings:
        for n in read_numbers(reading):
            values.append(n.value)
            if _PERCENT.match(reading, n.end):
                values.append(n.value / 100)
            values += [n.value * factor for scale, factor in _SCALES if scale.match(reading, n.end)]
        values += [float(number) * _SUFFIXES[suffix] for number, suffix in _SUFFIXED.findall(reading)]
    for sign, a, b in _FRACTION.findall(text) + _LATEX_FRACTION.findall(text):
        if float(b):
            values.append((-1 if sign else 1) * float(a) / float(b))
    return values


def _normal(text: str) -> str:
    """Text as an exact value is compared: case, Markdown emphasis, subscript digits, digit-group separators and
    runs of whitespace do not count."""
    text = re.sub(r"[*_`]", "", text.translate(_DIGITS).casefold())
    text = re.sub(rf"(?<=\d){_GROUP}(?=\d{{3}}(?!\d))", "", text)
    return re.sub(r"\s+", " ", text).strip()


def text_in(answer: str, value: str) -> bool:
    """The value occurs in the answer as a whole token (`11` not in `2011`, `11.5` or `1:11`); a clock time H:MM
    also written 0H:MM; a value with whitespace and punctuation (`[2, 2, 2]`) also with all whitespace removed."""
    text, want = _normal(_body(answer)), _normal(value)
    if not want:
        return False
    pattern = ("0?" if re.fullmatch(r"\d:\d\d", want) else "") + re.escape(want)
    before = r"(?<!\w)(?<!\d[.,:])" if want[0].isdigit() else r"(?<!\w)"
    after = r"(?!\w)(?![.,:]\d)" if want[-1].isdigit() else r"(?!\w)"
    if re.search(before + pattern + after, text):
        return True
    spaced = " " in want and re.search(r"[^\w\s]", want) is not None
    return spaced and re.sub(r"\s", "", want) in re.sub(r"\s", "", text)


def reference_found(answer: str, check: dict[str, Any]) -> dict[str, Any]:
    """The quick signal of an exact or numeric check: `status` found | missing, `passed`, `detail` (the values
    not found)."""
    values = check_values(check)
    if check["type"] == "numeric":
        have, tolerance = numbers_in(answer), float(check["tolerance"])
        missing = [v for v in values if not any(abs(n - v) <= tolerance + 1e-9 * abs(v) for n in have)]
    else:
        missing = [v for v in values if not text_in(answer, v)]
    detail = "not found: " + ", ".join(f"{v:g}" if isinstance(v, float) else v for v in missing) if missing else ""
    return {"status": "missing" if missing else "found", "passed": not missing, "detail": detail[:DETAIL_CHARS]}


def run_check(item: dict[str, Any], answer: str, draft: str, python: Any = None) -> dict[str, Any] | None:
    """The `check` of an item log: the item's code check on the final answer and on the draft (the same object
    when the texts are equal), or None when the item has none. `python`: the `evals.python_tests` settings;
    without them, or with `enabled` false, no code is run and a python_tests item has no check."""
    check = (item.get("reference") or {}).get("check") or {}
    kind = check.get("type")
    if kind == "python_tests":
        if python is None or not python.enabled:
            return None

        def one(text: str) -> dict[str, Any]:
            return python_tests(text, check, python.timeout_s, python.memory_mb)
    elif kind in ("exact", "numeric"):
        def one(text: str) -> dict[str, Any]:
            return reference_found(text, check)
    else:
        return None
    final = one(answer)
    return {"type": kind, "final": final, "draft": final if draft == answer else one(draft)}
