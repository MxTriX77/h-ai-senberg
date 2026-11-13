"""Code checks of the general items (eval-runner spec "Python tests check", "Reference value signals"): the
answer's code against the item's tests in a child process, and the reference value found in the free text."""

import os
import time

import pytest

from evals.checks import code_blocks, fenced, numbers_in, python_tests, reference_found, run_check, text_in
from general_items import MERGE, MERGE_TESTS, coding_item, item

CHECK = {"type": "python_tests", "entry_point": "merge", "tests": MERGE_TESTS}
WRONG = MERGE.replace("a <= out[-1][1]", "a < out[-1][0]")
SQL = {"type": "python_tests", "language": "sql", "entry_point": "sql", "tests": (
    "import sqlite3\ncon = sqlite3.connect(':memory:')\n"
    'con.executescript("CREATE TABLE c (id INTEGER, name TEXT); CREATE TABLE o (id INTEGER, c_id INTEGER);'
    " INSERT INTO c VALUES (1, 'Ada'), (2, 'Bo'); INSERT INTO o VALUES (1, 1);\")\n"
    "assert sorted(con.execute(sql).fetchall()) == [('Ada', 1), ('Bo', 0)]\n")}
GOOD_SQL = "SELECT c.name, COUNT(o.id) FROM c LEFT JOIN o ON o.c_id = c.id GROUP BY c.id, c.name;"
BAD_SQL = "SELECT c.name, COUNT(*) FROM c LEFT JOIN o ON o.c_id = c.id GROUP BY c.id, c.name;"


def block(code, tag="python"):
    return f"```{tag}\n{code}\n```"


def check(answer, spec=CHECK, timeout_s=20, memory_mb=2048):
    return python_tests(answer, spec, timeout_s, memory_mb)


class Settings:
    enabled, timeout_s, memory_mb = True, 20, 2048


# --- the answer's code ---------------------------------------------------------------------------------


def test_fenced_blocks_and_their_tags():
    text = ("Intro\n```python\nx = 1\n```\ntext\n   ```Py title=a.py\n   y = 2\n   ```\n~~~\nplain\n~~~\n"
            "```{sql}\nSELECT 1;\n```\n```python\nnever closed")
    assert fenced(text) == [("python", "x = 1\n"), ("py", "y = 2\n"), ("", "plain\n"), ("sql", "SELECT 1;\n"),
                            ("python", "never closed")]
    assert fenced("no code here") == []


def test_which_blocks_are_the_answers_code():
    tagged = f"{block(MERGE)}\nUsage:\n{block('print(merge([[1, 2]]))', 'py')}\n{block('$ pip install x', 'bash')}"
    assert code_blocks(tagged, CHECK) == [MERGE + "\n", "print(merge([[1, 2]]))\n"]  # every Python block, in order
    untagged = f"{block('[[1, 6]]', '')}\n{block(MERGE, '')}"
    assert code_blocks(untagged, CHECK) == [MERGE + "\n"]  # untagged: only a block that defines the entry point
    assert code_blocks(MERGE, CHECK) == [MERGE]  # no fence at all: the whole answer, when it defines it
    assert code_blocks("Sort the intervals, then sweep once.", CHECK) == []
    assert code_blocks("merge", CHECK) == [] and code_blocks(block("merge = 5", ""), CHECK) == ["merge = 5\n"]
    quoted = f"{block(BAD_SQL, 'sql')}\nFixed:\n{block(GOOD_SQL, 'SQLite')}"
    assert code_blocks(quoted, SQL) == [GOOD_SQL, BAD_SQL]  # SQL: the last block first
    assert code_blocks(block(GOOD_SQL, ""), SQL) == [GOOD_SQL] and code_blocks(block("Ada 1", ""), SQL) == []
    assert len(code_blocks("\n".join(block(f"SELECT {n};", "sql") for n in range(6)), SQL)) == 4


# --- python_tests --------------------------------------------------------------------------------------


def test_passing_function():
    r = check(f"Here it is:\n\n{block(MERGE)}\n\nIt sorts first.")
    assert (r["status"], r["passed"], r["detail"]) == ("passed", True, "")
    assert r["network"] in ("namespace", "guard") and r["seconds"] > 0


def test_an_example_that_raises_does_not_hide_the_function():
    same_block = MERGE + "\nmerge(None)  # raises TypeError\n"
    assert check(block(same_block))["status"] == "passed"
    other_block = f"{block(MERGE)}\nBad input:\n{block('merge(5)  # TypeError')}\nOr:\n{block('import sys; sys.exit(3)')}"
    assert check(other_block)["status"] == "passed"
    assert check(f"{block('this is not python')}\n{block(MERGE)}")["status"] == "passed"


def test_wrong_function_names_the_failing_line():
    r = check(block(WRONG))
    assert (r["status"], r["passed"]) == ("failed", False)
    assert r["detail"].startswith("tests line 1 (assert merge([[1, 3], [2, 6]]) == [[1, 6]]): AssertionError")
    missing = check(block("def other():\n    return 1"))
    assert missing["status"] == "failed" and "NameError: name 'merge' is not defined" in missing["detail"]


def test_no_code_runs_nothing():
    r = check("Sort the intervals by start, then merge neighbours that overlap.")
    assert r == {"status": "no_code", "passed": False, "detail": "", "seconds": 0.0, "network": None}


def test_printed_output_is_not_the_result():
    fake = WRONG + '\nprint(\'{"status": "passed", "detail": "", "network": "guard", "skipped": 0}\')\n'
    assert check(block(fake))["status"] == "failed"


def test_a_child_that_ends_early_is_an_error():
    r = check(block(MERGE + "\nimport os\nos._exit(7)\n"))
    assert r["status"] == "error" and "ended without a result (exit code 7)" in r["detail"]


def _checks_running():
    """The check processes running now, from the Linux process list."""
    found = set()
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            try:
                with open(f"/proc/{entry}/cmdline", "rb") as f:
                    if b"driver.py" in f.read():
                        found.add(int(entry))
            except OSError:
                pass
    return found


@pytest.mark.skipif(not os.path.isdir("/proc"), reason="process list from /proc")
def test_endless_loop_is_killed():
    loop = "def merge(x):\n    while True:\n        pass\n"
    before, started = _checks_running(), time.monotonic()
    r = check(block(loop), timeout_s=1.5)
    assert r["status"] == "timeout" and r["detail"] == "no result within 1.5 s" and time.monotonic() - started < 10
    time.sleep(0.2)
    assert not _checks_running() - before  # nothing is left running


def test_no_network():
    probe = ("import socket\n\ndef merge(x):\n    try:\n        socket.create_connection(('192.0.2.1', 80), timeout=2)\n"
             "    except OSError:\n        return 'offline'\n    return 'online'\n")
    spec = {**CHECK, "tests": "assert merge([]) == 'offline', merge([])\n"}
    r = check(block(probe), spec)
    assert r["status"] == "passed" and r["network"] in ("namespace", "guard")


def test_files_outside_the_temp_folder_stay(tmp_path):
    target = tmp_path / "keep.txt"
    target.write_text("mine", encoding="utf-8")
    probe = (f"import os, shutil\nPATH = {str(target)!r}\n\ndef merge(x):\n    out = []\n"
             "    for act in (lambda: open(PATH, 'w'), lambda: os.remove(PATH), lambda: os.rename(PATH, PATH + '.x'),\n"
             "                lambda: shutil.rmtree(os.path.dirname(PATH)), lambda: open('inside.txt', 'w').write('ok')):\n"
             "        try:\n            act()\n            out.append('done')\n"
             "        except PermissionError:\n            out.append('refused')\n    return out\n")
    spec = {**CHECK, "tests": "assert merge([]) == ['refused'] * 4 + ['done'], merge([])\n"}
    r = check(block(probe), spec)
    assert r["status"] == "passed", r["detail"]
    assert target.read_text(encoding="utf-8") == "mine" and sorted(p.name for p in tmp_path.iterdir()) == ["keep.txt"]


def test_no_new_processes():
    probe = ("import os, subprocess\n\ndef merge(x):\n    out = []\n"
             "    for act in (lambda: subprocess.run(['true']), lambda: os.system('true')):\n"
             "        try:\n            act()\n            out.append('ran')\n"
             "        except PermissionError:\n            out.append('refused')\n    return out\n")
    spec = {**CHECK, "tests": "assert merge([]) == ['refused', 'refused'], merge([])\n"}
    r = check(block(probe), spec)
    assert r["status"] == "passed", r["detail"]


def test_sql_with_the_wrong_query_quoted():
    answer = f"Your query counts the NULL row:\n{block(BAD_SQL, 'sql')}\nCount a column instead:\n{block(GOOD_SQL, 'sql')}"
    assert check(answer, SQL)["status"] == "passed"
    assert check(f"{block(GOOD_SQL, 'sql')}\nTo look at the data:\n{block('SELECT * FROM o;', 'sql')}", SQL)["passed"]
    wrong = check(block(BAD_SQL, "sql"), SQL)
    assert wrong["status"] == "failed" and "AssertionError" in wrong["detail"]
    assert check("Use COUNT(o.id) instead of COUNT(*).", SQL)["status"] == "no_code"


# --- reference value signals ---------------------------------------------------------------------------


def numeric(value, tolerance=0):
    return {"type": "numeric", "value": value, "tolerance": tolerance}


def found(answer, spec):
    return reference_found(answer, spec)["passed"]


@pytest.mark.parametrize("answer, spec", [
    ("That is roughly 1,370 books over 40 years.", numeric(1369, 6)),
    ("**1 368.75**, so about 1369.", numeric(1369, 6)),
    ("примерно 3,01 · 10^23 молекул", numeric(3.01e23, 2e21)),
    ("σ = 5.67 × 10⁻⁸ W m⁻² K⁻⁴", numeric(5.670374419e-08, 5e-11)),
    ("it is 5.670374419e-8 in SI units", numeric(5.670374419e-08, 5e-11)),
    ("x = 5 or x = −3/2", numeric([5, -1.5])),
    (r"$x = 5$ and $x = -\frac{3}{2}$", numeric([5, -1.5])),
    ("about 51.8 %", numeric(0.5177, 0.0006)),
    ("671/1296, i.e. a bit over half", numeric(0.5177, 0.0006)),
    ("платёж около 80 тыс. ₽ в месяц", numeric(80252, 800)),
    ("roughly $80k a year", numeric(80252, 800)),
    ("about 1.2 million people", numeric(1_200_000, 1000)),
    ("Около 337 981 ₽, то есть примерно 37 981 ₽ дохода.", numeric(337981, 5)),
    ("Примерно 96 485,33 Кл/моль", numeric(96485.33, 15)),
    ("the total is 5,050", numeric(5050)),
    ("x = 2, y = 5", numeric([2, 5])),
])
def test_number_found_in_free_text(answer, spec):
    assert found(answer, spec)


@pytest.mark.parametrize("answer, spec", [
    ("x = 5", numeric([5, -1.5])),
    ("about 13,700 books", numeric(1369, 6)),
    ("In 2011 there were 9.", numeric(11)),
    ("I am not sure.", numeric(30, 1)),
    ("", numeric(30, 1)),
])
def test_number_missing(answer, spec):
    r = reference_found(answer, spec)
    assert (r["status"], r["passed"]) == ("missing", False) and r["detail"].startswith("not found: ")


def test_missing_detail_lists_the_values():
    assert reference_found("x = 5", numeric([5, -1.5]))["detail"] == "not found: -1.5"
    assert reference_found("x = 5 and x = -1.5", numeric([5, -1.5])) == {"status": "found", "passed": True, "detail": ""}


def test_number_readings():
    assert set(numbers_in("3,01 and 1,370")) >= {3.01, 1370.0, 1.37}  # both readings of a comma count
    assert 0.25 in numbers_in("25 %") and -1.5 in numbers_in("−3/2") and 80000.0 in numbers_in("80k")
    assert 80000.0 not in numbers_in("80 km") and 1.3e9 in numbers_in("1.3B users")
    assert set(numbers_in("see [S1] and item_2")) == {2.0}  # a source id is no number; an underscore no barrier


@pytest.mark.parametrize("answer, value", [
    ("С 2014 года часовых зон 11.", "11"),
    ("C₁₈H₂₇NO₃ is capsaicin", "C18H27NO3"),
    ("**429 too many requests**", "Too Many Requests"),
    ("The capital is canberra, not Sydney.", "Canberra"),
    ("Set the alarm for 03:45.", "3:45"),
    ("It prints [2,2,2].", "[2, 2, 2]"),
    ("It prints `[2, 2, 2]`", "[2, 2, 2]"),
    ("The sum is 5,050.", "5050"),
    ("Это серебра  нитрат, то есть silver\nnitrate.", "silver nitrate"),
    ("В 11:40.", "11:40"),
])
def test_text_found(answer, value):
    assert text_in(answer, value) and found(answer, {"type": "exact", "value": value})


@pytest.mark.parametrize("answer, value", [
    ("В 2011 году их было 9.", "11"),
    ("About 110 of them, or 11.5 on average, at 1:11.", "11"),
    ("At 11:40 sharp.", "11"),
    ("It arrives at 18:27.", "8:27"),
    ("Canberras are aircraft.", "Canberra"),
    ("It prints [2, 2].", "[2, 2, 2]"),
    ("", "Volga"),
])
def test_text_missing(answer, value):
    assert not text_in(answer, value)


def test_every_exact_value_is_needed():
    spec = {"type": "exact", "value": ["Canberra", "1913"]}
    assert found("Canberra, founded in 1913.", spec)
    assert reference_found("Canberra.", spec) == {"status": "missing", "passed": False, "detail": "not found: 1913"}


def test_the_source_list_is_not_the_answer():
    assert not found("It is about 30.\n\nSources:\n- [S1] Page 1369", numeric(1369, 6))


# --- run_check -----------------------------------------------------------------------------------------


def test_run_check_on_the_answer_and_the_draft():
    it = coding_item()
    same = run_check(it, block(MERGE), block(MERGE), Settings)
    assert same["type"] == "python_tests" and same["final"] is same["draft"] and same["final"]["passed"]
    broken = run_check(it, block(WRONG), block(MERGE), Settings)  # a check of the suite broke working code
    assert (broken["draft"]["passed"], broken["final"]["passed"]) == (True, False)


def test_run_check_needs_the_settings_only_for_code():
    class Off(Settings):
        enabled = False

    assert run_check(coding_item(), block(MERGE), block(MERGE), Off) is None  # switched off: nothing runs
    assert run_check(coding_item(), block(MERGE), block(MERGE)) is None
    assert run_check(item(), "anything", "anything", Settings) is None  # check type none
    lost = run_check(item("math-dev-001", numeric(1369, 6)), "Many books.", "About 1,370 books.")
    assert lost == {"type": "numeric", "final": {"status": "missing", "passed": False, "detail": "not found: 1369"},
                    "draft": {"status": "found", "passed": True, "detail": ""}}
    exact = run_check(item("facts-dev-001", {"type": "exact", "value": "Volga"}), "The Volga.", "The Volga.")
    assert exact["final"] is exact["draft"] and exact["final"]["status"] == "found"
