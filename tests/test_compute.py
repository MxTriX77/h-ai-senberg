"""The compute tool's operations: exact procedures read by a fixed parser (agent-tools spec, Compute tool)."""

import pytest

from haisenberg.compute import parse_args, run


def result(entry: str) -> str:
    line = run(entry)
    assert line.startswith(f"{entry} = "), line
    return line[len(entry) + 3:]


def error(entry: str) -> str:
    line = run(entry)
    assert line.startswith(f"{entry}: error: "), line
    return line[len(entry) + 9:]


# --- the parser -----------------------------------------------------------------------------------------------

def test_arguments():
    assert parse_args("a, [b, c], 'd, e', \"[f]\"") == ["a", ["b", "c"], "d, e", "[f]"]
    assert parse_args("2, [[Ann, 31], [Bob, 25]]") == ["2", [["Ann", "31"], ["Bob", "25"]]]
    assert parse_args("") == [] and parse_args("[]") == [[]] and parse_args(" O'Brien , x ") == ["O'Brien", "x"]


@pytest.mark.parametrize("entry, why", [
    ("sort(a,,b)", "an empty item"),
    ("sort(a, b,)", "an empty item"),
    ("sort([a, b)", "a list is not closed"),
    ("sort(a, b])", "unexpected ']'"),
    ("sort('a, b)", "a quote is not closed"),
    ("sort(\"a\" b)", "unexpected 'b'"),
    ("sort(a[1])", "unexpected '['"),
    ("sort([[[a]]])", "no deeper"),
    ("weekday 2026-10-29", "write an operation as name(arguments)"),
    ("frobnicate(1)", "unknown operation 'frobnicate'; available: weekday, add_days"),
    ("weekday(2026-10-29, 2026-10-30)", "wrong number of arguments; use weekday(date)"),
    ("weekday([2026-10-29])", "a date must be one value, not a list"),
])
def test_unreadable_entries(entry, why):
    assert why in error(entry)


def test_not_code():
    assert error("__import__('os').system('x')").startswith("unknown operation '__import__'")
    assert error("sort(" + "a, " * 2000 + "b)") == "operation longer than 4000 characters"
    assert result("sort(1e999999999, 2)") == "1e999999999, 2 (as text, ignoring case)"  # not a number: no huge integer


def test_not_a_string():
    assert run(None).startswith("error: an operation must be a non-empty string")
    assert run("  ").startswith("error:")


def test_operation_names_ignore_case_and_spaces():
    assert run("  Weekday (2026-10-29) ") == "Weekday (2026-10-29) = Thursday"


# --- dates ----------------------------------------------------------------------------------------------------

def test_weekday():
    assert result("weekday(2026-10-29)") == "Thursday"
    assert result("weekday(2024-02-29)") == "Thursday"  # leap day
    assert result("weekday(2000-02-29)") == "Tuesday"  # divisible by 400: a leap year
    assert result("weekday(\"2026-1-5\")") == "Monday"


@pytest.mark.parametrize("entry, why", [
    ("weekday(2026-02-30)", "unknown date 2026-02-30 (February 2026 has 28 days)"),
    ("weekday(2100-02-29)", "unknown date 2100-02-29 (February 2100 has 28 days)"),  # not a leap year
    ("weekday(2026-13-01)", "unknown date 2026-13-01 (there is no month 13)"),
    ("weekday(0000-01-01)", "unknown date 0000-01-01 (years start at 1)"),
    ("weekday(29.10.2026)", "not a date: '29.10.2026' (write YYYY-MM-DD)"),
    ("add_days(9999-12-31, 1)", "the result is outside the years 1 to 9999"),
    ("add_days(2026-01-01, 99999999999999)", "the result is outside the years 1 to 9999"),
    ("add_months(9999-12-01, 1)", "the result is outside the years 1 to 9999"),
    ("add_days(2026-01-01, 1.5)", "n must be a whole number: 1.5"),
])
def test_bad_dates(entry, why):
    assert error(entry) == why


def test_add_days_and_weeks():
    assert result("add_days(2026-10-29, 30)") == "2026-11-28 (Saturday)"
    assert result("add_days(2026-12-30, 5)") == "2027-01-04 (Monday)"  # across a year boundary
    assert result("add_days(2026-03-01, -1)") == "2026-02-28 (Saturday)"
    assert result("add_days(2024-03-01, -1)") == "2024-02-29 (Thursday)"  # leap year
    assert result("add_days(2026-10-29, +0)") == "2026-10-29 (Thursday)"
    assert result("add_weeks(2026-10-29, -2)") == "2026-10-15 (Thursday)"
    assert result("add_weeks(2026-12-24, 2)") == "2027-01-07 (Thursday)"


def test_add_months_clamps_month_ends():
    assert result("add_months(2024-01-31, 1)") == "2024-02-29 (Thursday; clamped to the month end, 2024-02-31 does not exist)"
    assert result("add_months(2026-01-31, 1)") == "2026-02-28 (Saturday; clamped to the month end, 2026-02-31 does not exist)"
    assert result("add_months(2026-03-31, -1)") == "2026-02-28 (Saturday; clamped to the month end, 2026-02-31 does not exist)"
    assert result("add_months(2026-10-31, 1)") == "2026-11-30 (Monday; clamped to the month end, 2026-11-31 does not exist)"
    assert result("add_months(2026-11-15, 3)") == "2027-02-15 (Monday)"  # across a year boundary
    assert result("add_months(2026-01-15, -2)") == "2025-11-15 (Saturday)"
    assert result("add_months(2024-02-29, 12)") == "2025-02-28 (Friday; clamped to the month end, 2025-02-29 does not exist)"


def test_days_between():
    assert result("days_between(2026-12-30, 2027-01-04)") == "5 days (exclusive; 6 counting both dates)"
    assert result("days_between(2024-02-01, 2024-03-01)") == "29 days (exclusive; 30 counting both dates)"
    assert result("days_between(2026-02-01, 2026-03-01)") == "28 days (exclusive; 29 counting both dates)"
    assert result("days_between(2026-10-29, 2026-10-30)") == "1 day (exclusive; 2 counting both dates)"
    assert result("days_between(2026-10-29, 2026-10-29)") == "0 days (exclusive; 1 counting both dates)"
    assert result("days_between(2027-01-04, 2026-12-30)") == (
        "-5 days (the second date is earlier; exclusive; 6 counting both dates)")


def test_count_weekdays():
    assert result("count_weekdays(2026-10-01, 2026-10-31, Sat, Sun)") == "9 (Saturday 5, Sunday 4; both dates included)"
    assert result("count_weekdays(2024-02-01, 2024-02-29, thursday)") == "5 (Thursday 5; both dates included)"
    assert result("count_weekdays(2026-10-29, 2026-10-29, Thu, Fri)") == "1 (Thursday 1, Friday 0; both dates included)"
    assert result("count_weekdays(2026-12-28, 2027-01-08, [Mon, Tue, Wed, Thu, Fri], Mondays)") == (
        "10 (Monday 2, Tuesday 2, Wednesday 2, Thursday 2, Friday 2; both dates included)")  # across a year boundary
    assert error("count_weekdays(2026-10-31, 2026-10-01, Sat)") == "the first date is after the second"
    assert error("count_weekdays(2026-10-01, 2026-10-31, Funday)") == "not a weekday: 'Funday' (write Monday or Mon, ...)"
    assert "wrong number of arguments" in error("count_weekdays(2026-10-01, 2026-10-31)")


# --- clock times and durations --------------------------------------------------------------------------------

def test_clock():
    assert result("clock(09:45 + 2:30)") == "12:15"
    assert result("clock(9:45 + 2:30 - 0:15)") == "12:00"
    assert result("clock(23:30 + 1:45)") == "01:15 (next day)"
    assert result("clock(00:30 - 0:45)") == "23:45 (previous day)"
    assert result("clock(10:00 + 50 h)") == "12:00 (2 days later)"
    assert result("clock(10:00 - 3 days)") == "10:00 (3 days earlier)"
    assert result("clock(08:00 + 90 min + 0:00:30)") == "09:30:30"
    assert error("clock(24:30 + 1:00)") == "not a clock time: '24:30' (00:00 to 23:59)"
    assert "start with a clock time" in error("clock(-1:00 + 2:00)")
    assert "a sign without a time or duration" in error("clock(10:00 +)")


def test_duration():
    assert result("duration(1:30 + 2:45 - 0:50)") == "3:25 (205 minutes)"
    assert result("duration(-0:15)") == "-0:15 (-15 minutes)"
    assert result("duration(0:01)") == "0:01 (1 minute)"
    assert result("duration(1:00:30 + 15 s)") == "1:00:45 (3645 seconds)"
    assert result("duration(50:00)") == "50:00 (3000 minutes; 2 days 2:00)"
    assert result("duration(1.5 h + 2 hours + 1 week)") == "171:30 (10290 minutes; 7 days 3:30)"
    assert error("duration(1:75)").startswith("not a time or duration: '1:75'")
    assert error("duration(0.5 s)") == "0.5 s is not a whole number of seconds"
    assert error("duration(3 parsecs)") == "unknown time unit 'parsecs' (use s, min, h, d or week)"


def test_time_between():
    assert result("time_between(22:30, 06:15)") == "7:45 (465 minutes; past midnight)"
    assert result("time_between(08:15, 17:00)") == "8:45 (525 minutes)"
    assert result("time_between(08:15, 08:15)") == "0:00 (0 minutes)"
    assert result("time_between(08:15:10, 08:16)") == "0:00:50 (50 seconds)"


def test_convert():
    assert result("convert(90, min, h)") == "1.5 h"
    assert result("convert(100, min, h)") == "5/3 h (≈ 1.666666667 h)"  # exact, with its value
    assert result("convert(90 min, h)") == "1.5 h"
    assert result("convert(3, days, hours)") == "72 hours"
    assert result("convert(1, week, s)") == "604800 s"
    assert result("convert(1, s, d)") == "1/86400 d (≈ 1.157407407e-05 d)"
    assert result("convert(36, h, d)") == "1.5 d"
    assert error("convert(x, h, d)") == "not a number: 'x'"
    assert error("convert(3, kg, g)") == "unknown time unit 'kg' (use s, min, h, d or week)"


# --- lists and sets -------------------------------------------------------------------------------------------

def test_sort():
    assert result("sort(10, 9, 100)") == "9, 10, 100 (as numbers)"
    assert result("sort([3.5, -2, 1e1, 0])") == "-2, 0, 3.5, 1e1 (as numbers)"
    assert result("sort_desc(9:05, 10:00, 8:59)") == "10:00, 9:05, 8:59 (as times, descending)"
    assert result("sort(b, A, a, C)") == "A, a, b, C (as text, ignoring case)"  # ties keep their order
    assert result("sort(10, 9, x)") == "10, 9, x (as text, ignoring case)"
    assert result("sort(3, 1, 3)") == "1, 3, 3 (as numbers)"  # duplicates kept
    assert result("sort(\"b, c\", a)") == 'a, "b, c" (as text, ignoring case)'
    assert result("sort([])") == "(empty list)" and result("sort()") == "(empty list)"
    assert error("sort([a], [b])") == "give the items, or one list of them; use sort(a, b, c) or sort([a, b, c])"


def test_unique_and_count():
    assert result("unique(b, a, b, 3, 3.0)") == "b, a, 3 (3 distinct of 5 items)"
    assert result("unique([Apple, apple])") == "Apple, apple (2 distinct of 2 items)"  # text is case-sensitive
    assert result("unique([])") == "(empty list)"
    assert result("count(a, b, a)") == "3 items (2 distinct)"
    assert result("count([])") == "0 items (0 distinct)"
    assert result("count(x)") == "1 item (1 distinct)"


def test_position():
    assert result("position(b, [a, b, c, b])") == "2, 4 (1-based; 2 matches in 4 items)"
    assert result("position(2, [1, 2.0, 3])") == "2 (1-based; 1 match in 3 items)"
    assert result("position(z, [a, b])") == "not found (0 matches in 2 items)"
    assert result("position(z, [])") == "not found (0 matches in 0 items)"
    assert "the second argument must be a list" in error("position(a, b)")


def test_set_operations():
    assert result("union([a, b], [b, c], [c, d])") == "a, b, c, d (4 items)"
    assert result("union([1, 2], [2.0, 3])") == "1, 2, 3 (3 items)"
    assert result("union([], [])") == "(empty list)"
    assert result("intersection([a, b, c, b], [c, b, d])") == "b, c (2 items)"  # first list's order, no duplicates
    assert result("intersection([a, b], [c])") == "(empty list)"
    assert result("intersection([a, b, c], [b, c], [c])") == "c (1 item)"
    assert result("difference([a, b, c, a], [b])") == "a, c (2 items)"
    assert result("difference([a, b, c], [b], [c])") == "a (1 item)"
    assert result("difference([], [a])") == "(empty list)"
    assert result("difference([a], [])") == "a (1 item)"
    assert error("union([a, b])") == "give two or more lists; use union([a, b], [b, c])"
    assert error("difference(a, b)").startswith("give two or more lists")


def test_range():
    assert result("range(3, 17)") == "15 integers: 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 (3 to 17, both included)"
    assert result("range(5, 5)") == "1 integer: 5 (5 to 5, both included)"
    assert result("range(3, 1)") == "3 integers: 3, 2, 1 (3 to 1, both included)"
    assert result("range(-2, 2)") == "5 integers: -2, -1, 0, 1, 2 (-2 to 2, both included)"
    assert result("range(3, 16, 2)") == "7 integers: 3, 5, 7, 9, 11, 13, 15 (3 to 16 in steps of 2; first 3, last 15)"
    assert result("range(10, 1, -3)") == "4 integers: 10, 7, 4, 1 (10 to 1 in steps of -3; first 10, last 1)"
    assert result("range(1, 1000)") == "1000 integers (1 to 1000, both included; list not shown, more than 50)"
    assert result("range(1, 50)").startswith("50 integers: 1, 2,")
    assert result("range(1, 10000000000000000000000)").startswith("10000000000000000000000 integers (")
    assert error("range(1, 10, 0)") == "the step must not be 0"
    assert error("range(1, 10, -1)") == "a step of -1 never reaches 10 from 1"
    assert error("range(1.5, 3)") == "a must be a whole number: 1.5"


# --- schedules ------------------------------------------------------------------------------------------------

def tasks(line: str) -> dict[str, str]:
    """task -> its fields, from a schedule result."""
    return dict(part.split(": ", 1) for part in line.split("; ")[2:])


def test_schedule_from_the_brief():
    line = result("schedule(A:3; B:2 after A; C:4 after A; D:1 after B,C)")
    assert line.startswith("total duration 8; critical path A → C → D; ")
    assert tasks(line) == {
        "A": "earliest start 0, earliest finish 3, latest start 0, latest finish 3, slack 0",
        "B": "earliest start 3, earliest finish 5, latest start 5, latest finish 7, slack 2",  # from A's finish
        "C": "earliest start 3, earliest finish 7, latest start 3, latest finish 7, slack 0",
        "D": "earliest start 7, earliest finish 8, latest start 7, latest finish 8, slack 0",  # after the later of B, C
    }


def test_schedule_with_two_critical_paths():
    line = result("schedule(A: 3; B: 4 after A; C: 4 after A; D: 1 after B, C)")
    assert line.startswith("total duration 8; critical paths A → B → D, A → C → D; ")
    assert all(f.endswith("slack 0") for f in tasks(line).values())


def test_schedule_backward_pass_uses_latest_starts():
    # E's latest finish is the earlier of its successors' latest starts (F: 7 - 1 = 6, G: 7 - 4 = 3), not a finish
    line = result("schedule(Start: 2; E: 1 after Start; F: 1 after E; G: 4 after E; H: 3 after Start;"
                  " End: 0 after F, G, H)")
    t = tasks(line)
    assert line.startswith("total duration 7; critical path Start → E → G → End; ")
    assert t["E"] == "earliest start 2, earliest finish 3, latest start 2, latest finish 3, slack 0"
    assert t["F"] == "earliest start 3, earliest finish 4, latest start 6, latest finish 7, slack 3"
    assert t["H"] == "earliest start 2, earliest finish 5, latest start 4, latest finish 7, slack 2"


def test_schedule_parallel_starts_and_decimals():
    line = result("schedule(Design review: 1.5; Build: 2.25\nTest: 0.5 AFTER Design review, Build)")
    assert line.startswith("total duration 2.75; critical path Build → Test; ")
    assert tasks(line)["Design review"].endswith("latest start 0.75, latest finish 2.25, slack 0.75")


def test_schedule_single_task_and_zero_durations():
    assert result("schedule(A: 5)") == ("total duration 5; critical path A; "
                                        "A: earliest start 0, earliest finish 5, latest start 0, latest finish 5, slack 0")
    line = result("schedule(Go: 0; A: 2 after Go; B: 2 after Go; Done: 0 after A, B)")
    assert line.startswith("total duration 2; critical paths Go → A → Done, Go → B → Done; ")


def test_many_critical_paths_capped():
    names = [f"T{i}" for i in range(12)]
    line = result("schedule(" + "; ".join(f"{n}: 1" for n in names) + ")")
    assert line.startswith("total duration 1; critical paths (first 10 shown) T0, T1, T2,")


@pytest.mark.parametrize("entry, why", [
    ("schedule(A: 1 after B; B: 2 after A)", "dependency cycle: A → B → A"),
    ("schedule(A: 1 after C; B: 1 after A; C: 1 after B; D: 2)", "dependency cycle: A → B → C → A"),
    ("schedule(A: 1 after A)", "dependency cycle: A → A"),
    ("schedule(A: 1; B: 2 after X)", "unknown task 'X' (B is after it)"),
    ("schedule(A: 1; A: 2)", "task A is defined twice"),
    ("schedule(A: -1)", "the duration of A is not a number of at least 0: -1"),
    ("schedule(A: 3 days)", "cannot read task 'A: 3 days'"),
    ("schedule(A 3)", "cannot read task 'A 3'"),
    ("schedule(A: 1; B: 1 after A,)", "an empty task name after B's 'after'"),
    ("schedule( ; )", "no tasks"),
])
def test_schedule_errors(entry, why):
    assert why in error(entry)


# --- tables ---------------------------------------------------------------------------------------------------

def test_sort_rows():
    assert result("sort_rows(2, [Ann, 31], [Bob, 25], [Cy, 40])") == "[Bob, 25], [Ann, 31], [Cy, 40] (by column 2, as numbers)"
    assert result("sort_rows_desc(2, [[Ann, 31], [Bob, 25], [Cy, 31]])") == (
        "[Ann, 31], [Cy, 31], [Bob, 25] (by column 2, as numbers, descending)")  # ties keep their order
    assert result("sort_rows(1, [b, 1], [A, 2])") == "[A, 2], [b, 1] (by column 1, as text, ignoring case)"
    assert error("sort_rows(3, [Ann, 31])") == "row 1 has no column 3"
    assert error("sort_rows(0, [Ann, 31])") == "columns are numbered from 1"
    assert "row 2 must be a list of cells" in error("sort_rows(1, [Ann, 31], Bob)")
    assert error("sort_rows(1)").startswith("no rows")


def test_column_stats():
    assert result("column_stats(2, [Ann, 31], [Bob, 25], [Cy, 40])") == "sum 96, min 25 (row 2), max 40 (row 3); 3 rows"
    assert result("column_stats(2, [[Ann, 0.1], [Bob, 0.2], [Cy, 0.2]])") == (
        "sum 0.5, min 0.1 (row 1), max 0.2 (rows 2, 3); 3 rows")  # exact, no float error
    assert error("column_stats(2, [name, age], [Ann, 31])") == "row 1, column 2 is not a number: 'age'"
