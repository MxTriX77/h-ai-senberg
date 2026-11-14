"""Items of the general schema (eval-runner spec "Items of the general schema"): loading, validation, filters and
the proportional limit. Synthetic items; set EVAL_ITEMS_DIR to also load a real folder of items."""

import os
from collections import Counter
from pathlib import Path

import pytest
import yaml

from evals.items import check_values, is_general, keep, load_general, load_paths, pick_proportional
from evals.run import QUESTIONS, load_file, pick
from general_items import coding_item, item, write_items

TURNS = [{"role": "user", "content": "Aconcagua?"}, {"role": "assistant", "content": "About 6,961 m."}]


def test_valid_file_loads(tmp_path):
    path = write_items(tmp_path / "mixed.yaml",
                       item("reasoning-dev-001", turns=TURNS, probes=["multi_turn"]),
                       item("chat-dev-001", lang="ru", request_type="chat", prompt="Привет, как дела?"),
                       coding_item("coding-dev-001", reference={
                           "answer": "Sort, then sweep.", "key_points": [], "sources": [], "solution": "def merge(x): ...",
                           "check": {"type": "python_tests", "entry_point": "merge", "tests": "assert merge([]) == []"}}))
    first, chat, coding = load_general(path)
    assert [i["set"] for i in (first, chat, coding)] == ["reasoning", "chat", "coding"]  # the set is the category
    assert first["question"] == first["prompt"] and first["turns"] == TURNS and first["file"] == path
    assert chat["lang"] == "ru" and "parts" not in chat and "answer" not in chat  # nothing an answer line could use
    assert coding["reference"]["solution"] == "def merge(x): ..."  # a field the schema does not name is kept
    assert is_general(yaml.safe_load(path.read_text(encoding="utf-8")))


def test_any_category_and_the_monitor_split(tmp_path):
    path = write_items(tmp_path / "x.yaml", item("cooking-dev-001"), item("u-01", category="travel", split="monitor"))
    assert [i["category"] for i in load_general(path)] == ["cooking", "travel"]


def test_a_family_is_optional_and_a_text(tmp_path):
    path = write_items(tmp_path / "p.yaml", item("procedure-dev-001", family="synthesis"), item("chat-dev-001"))
    first, second = load_general(path)
    assert first["family"] == "synthesis" and "family" not in second
    for bad in (3, "", " "):
        path = write_items(tmp_path / "bad.yaml", item("procedure-dev-002", family=bad))
        with pytest.raises(ValueError) as e:
            load_general(path)
        assert str(e.value) == f"{path}: item procedure-dev-002: family must be a non-empty text"


def test_unquoted_yes_and_no(tmp_path):
    text = yaml.safe_dump({"items": [item("science-dev-001"), item("science-dev-002")]}, sort_keys=False)
    assert text.count("offline_base: n/a") == 2
    path = tmp_path / "s.yaml"
    path.write_text(text.replace("offline_base: n/a", "offline_base: yes", 1).replace("offline_base: n/a", "offline_base: no"),
                    encoding="utf-8")
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["items"][0]["offline_base"] is True  # what YAML makes of it
    assert [i["offline_base"] for i in load_general(path)] == ["yes", "no"]


@pytest.mark.parametrize("change, error", [
    ({"request_type": "essay"}, "request_type must be one of chat|factual|analytic|audit"),
    ({"split": "later"}, "split must be one of dev|holdout|monitor"),
    ({"lang": "de"}, "lang must be one of en|ru"),
    ({"category": " "}, "category must be a non-empty text"),
    ({"prompt": ""}, "prompt must be a non-empty text"),
    ({"turns": [{"role": "system", "content": "c"}]}, "turns must be a list"),
    ({"turns": None}, "turns must be a list"),
    ({"reference": {"key_points": [], "sources": [], "check": {"type": "none"}}}, "reference needs an answer"),
    ({"reference": {"answer": "a", "key_points": "one", "sources": [], "check": {"type": "none"}}},
     "reference.key_points must be a list of texts"),
    ({"must_not": "x"}, "must_not must be a list of texts"),
    ({"probes": [1]}, "probes must be a list of texts"),
    ({"offline_base": "maybe"}, "offline_base must be one of yes|likely|partial|no|n/a"),
    ({"notes": None}, "notes must be a text"),
])
def test_malformed_item_names_the_file_item_and_field(tmp_path, change, error):
    path = write_items(tmp_path / "bad.yaml", item("ok-dev-001"), item("x-dev-001", **change))
    with pytest.raises(ValueError) as e:
        load_general(path)
    assert str(e.value) == f"{path}: item x-dev-001: {error}" or str(e.value).startswith(f"{path}: item x-dev-001: {error}")


@pytest.mark.parametrize("check, error", [
    ({"type": "regex"}, "reference.check: type must be one of none|exact|numeric|python_tests"),
    ({"type": "exact"}, "reference.check: an exact check needs a value"),
    ({"type": "numeric", "value": 5}, "reference.check: a numeric check needs a tolerance of 0 or more"),
    ({"type": "numeric", "value": 5, "tolerance": -1}, "reference.check: a numeric check needs a tolerance of 0 or more"),
    ({"type": "numeric", "value": "five", "tolerance": 0}, "reference.check: a numeric check needs a value"),
    ({"type": "numeric", "value": [], "tolerance": 0}, "reference.check: a numeric check needs a value"),
    ({"type": "python_tests", "entry_point": "f"}, "reference.check: a python_tests check needs tests"),
    ({"type": "python_tests", "tests": "assert f()"}, "reference.check: a python_tests check needs an entry_point"),
    ({"type": "python_tests", "tests": "assert f()", "entry_point": "f", "language": "rust"},
     "reference.check: language must be one of python|sql"),
])
def test_malformed_check(tmp_path, check, error):
    with pytest.raises(ValueError, match="item x-dev-001: ") as e:
        load_general(write_items(tmp_path / "bad.yaml", item("x-dev-001", check)))
    assert error in str(e.value)


def test_check_values():
    assert check_values({"type": "numeric", "value": [5, -1.5], "tolerance": 0}) == [5.0, -1.5]
    assert check_values({"type": "numeric", "value": 1369, "tolerance": 6}) == [1369.0]
    assert check_values({"type": "exact", "value": 1989}) == ["1989"]  # an unquoted year
    assert check_values({"type": "exact", "value": ["Canberra", "1913"]}) == ["Canberra", "1913"]


def test_duplicate_ids(tmp_path):
    with pytest.raises(ValueError, match="missing or duplicate id 'a-dev-001'"):
        load_general(write_items(tmp_path / "a.yaml", item("a-dev-001"), item("a-dev-001")))
    write_items(tmp_path / "set" / "a.yaml", item("a-dev-001"))
    write_items(tmp_path / "set" / "b.yaml", item("a-dev-001"), item("b-dev-001"))
    with pytest.raises(ValueError, match=r"duplicate ids \['a-dev-001'\]"):
        load_paths([tmp_path / "set"])
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no \\*.yaml item files"):
        load_paths([tmp_path / "empty"])
    assert [i["id"] for i in load_paths([tmp_path / "set" / "b.yaml"])] == ["a-dev-001", "b-dev-001"]  # one file


def test_filters():
    items = [item("coding-dev-001", lang="ru"), item("coding-dev-002", request_type="audit", probes=["drift_bait"]),
             item("math-dev-001", lang="ru", probes=["false_premise", "multi_turn"]), item("chat-dev-001", lang="ru"),
             {"id": "f1", "set": "S1"}]  # an item of another schema passes every filter

    def ids(**filters):
        return [i["id"] for i in keep(items, **filters)]

    assert ids(categories=["coding", "math"], langs=["ru"]) == ["coding-dev-001", "math-dev-001", "f1"]
    assert ids(request_types=["audit"]) == ["coding-dev-002", "f1"]
    assert ids(probes=["drift_bait", "multi_turn"]) == ["coding-dev-002", "math-dev-001", "f1"]
    assert ids() == [i["id"] for i in items]


def general_set(sizes):
    return [item(f"{category}-dev-{n:03d}", lang="ru" if n % 5 == 0 else "en",
                 request_type="chat" if n % 2 else "analytic")
            for category, size in sizes.items() for n in range(1, size + 1)]


def test_limit_takes_a_share_of_every_category():
    items = general_set({"reasoning": 32, "chat": 16, "facts": 16})
    picked = pick_proportional(items, 8)
    assert Counter(i["category"] for i in picked) == {"reasoning": 4, "chat": 2, "facts": 2}
    assert picked == pick_proportional(items, 8) == pick(items, None, 8)  # the same items every time; pick uses it
    assert [i["id"] for i in picked] == [i["id"] for i in items if i in picked]  # in the files' order
    assert pick_proportional(items, 64) == items and pick_proportional(items, None) == items
    assert pick(items, ["chat-dev-003"], 8) == [items[32 + 2]]  # ids win over the limit


def test_limit_keeps_languages_and_small_categories():
    items = general_set({"reasoning": 40, "writing": 30, "tiny": 2})
    picked = pick_proportional(items, 12)
    assert Counter(i["category"] for i in picked) == {"reasoning": 6, "writing": 5, "tiny": 1}  # one at least
    langs = Counter(i["lang"] for i in pick_proportional(general_set({"reasoning": 100}), 20))
    assert langs == {"en": 16, "ru": 4}  # a fifth is Russian, in the set and in the sample
    assert Counter(i["category"] for i in pick_proportional(items, 2)) == {"reasoning": 1, "writing": 1}


def test_load_file_tells_the_schemas_apart(tmp_path):
    general = write_items(tmp_path / "g.yaml", item("facts-dev-001"))
    authored = tmp_path / "a.yaml"
    authored.write_text(yaml.safe_dump({"items": [{"id": "b1-u-01", "set": "U", "split": "monitor", "prompt": "Plan."}]}),
                        encoding="utf-8")
    assert load_file(general)[0]["category"] == "facts" and load_file(authored)[0]["set"] == "U"
    assert load_file(QUESTIONS)[0]["set"] == "S1"  # the older loaders are untouched
    assert not is_general({"items": [{"id": "b1-u-01", "set": "U", "prompt": "p"}]}) and not is_general({"questions": []})


@pytest.mark.skipif(not os.environ.get("EVAL_ITEMS_DIR"), reason="set EVAL_ITEMS_DIR to a folder of real items")
def test_real_items_load():
    items = load_paths([Path(os.environ["EVAL_ITEMS_DIR"])])
    assert items and all(i["set"] == i["category"] for i in items)
