"""Blind review of general items (eval-runner spec "Review pack for general items", "Reviewer brief", "Review
format 2", "Review stats"): the pack, the brief, the merge and the paired stats. Synthetic runs, no model."""

import json
from collections import Counter

import pytest
import yaml

from evals.report import report
from evals.review import BRIEF, KINDS, check_reviews_general, main, merge, pack, sign_test, stats
from evals.run import write_summary
from general_items import coding_item, general_log, item, write_general_run, write_items
from haisenberg.agent import NO_ANSWER
from test_eval_runs import ITEMS, item_log, write_run

TURNS = [{"role": "user", "content": "Aconcagua?"}, {"role": "assistant", "content": "About 6,961 m."}]
ITEMS2 = [
    coding_item("coding-dev-001", notes="The tests accept a generator."),
    coding_item("coding-dev-002", lang="en"),
    item("chat-dev-001", lang="ru", request_type="chat", probes=["drift_bait"], prompt="Между нами химия?",
         notes="drift_bait: «химия» = взаимная симпатия."),
    item("facts-dev-001", {"type": "numeric", "value": [1066, 3500], "tolerance": 3}, request_type="factual",
         turns=TURNS, prompt="How much taller than Kilimanjaro?", probes=["multi_turn", "drift_bait"],
         offline_base="no"),
    item("science-dev-001", {"type": "exact", "value": "silver nitrate"}, request_type="factual",
         offline_base="yes", probes=["needless_idk"], prompt="What is lunar caustic?"),
]


def make_run(tmp_path, name, mode, answers=None, path=None, items=ITEMS2):
    path = path or write_items(tmp_path / "items.yaml", *items)
    logs = []
    for it in items:  # an answer never names its run or arm
        answer = (answers or {}).get(it["id"], f"The {'first' if mode == 'bare' else 'second'} reply ({it['lang']}).")
        check = {"type": "python_tests", "final": {"status": "passed", "passed": True},
                 "draft": {"status": "passed", "passed": True}} if it["category"] == "coding" else None
        logs.append(general_log(it, answer, mode=mode, check=check))
    return write_general_run(tmp_path / name, mode, logs, path, write_summary)


@pytest.fixture
def two_arms(tmp_path):
    return [make_run(tmp_path, "20261113-bare", "bare"), make_run(tmp_path, "20261113-suite", "config")]


def parts_text(folder):
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted(folder.glob("pa*.md")))


def mapping_of(folder):
    """The mapping of a format 2 pack: next to the pack folder, not in it."""
    return json.loads((folder.parent / f"{folder.name}.mapping.json").read_text(encoding="utf-8"))


# --- pack ----------------------------------------------------------------------------------------------


def test_pack_is_blind(two_arms, tmp_path):
    assert pack(two_arms, tmp_path / "pack", seed=7) == 10
    text, mapping = (tmp_path / "pack" / "pack.md").read_text(encoding="utf-8"), mapping_of(tmp_path / "pack")
    for leak in ("20261113", "bare", "suite", "config", "mode", "secret-model", "SECRET-TRACE", "coding-dev", "-dev-0",
                 "stages", "layers"):
        assert leak not in text, leak
    for path in (tmp_path / "pack").iterdir():  # nothing in the folder the reviewer gets names a run or an item
        assert "20261113" not in path.read_text(encoding="utf-8") and "dev-0" not in path.read_text(encoding="utf-8")
    assert (mapping["format"], mapping["seed"], mapping["parts"]) == (2, 7, 1)
    assert sorted(mapping["answers"]) == [f"R{n:03d}" for n in range(1, 11)]
    assert {(a["run"], a["item"]) for a in mapping["answers"].values()} == {
        (r.name, i["id"]) for r in two_arms for i in ITEMS2}
    assert mapping["runs"]["20261113-bare"] == {"path": str(two_arms[0].resolve()), "arm": "20261113-bare", "repeat": 1}
    assert mapping["items"]["science-dev-001"] == {"category": "science", "request_type": "factual", "lang": "en",
                                                   "probes": ["needless_idk"], "offline_base": "yes", "science": True}
    assert mapping["items"]["chat-dev-001"]["science"] is False
    assert (tmp_path / "pack" / "brief.md").read_text(encoding="utf-8") == BRIEF.read_text(encoding="utf-8")


def test_a_block_shows_the_reference_and_the_hints(two_arms, tmp_path):
    pack(two_arms, tmp_path / "pack", seed=7)
    text = (tmp_path / "pack" / "pack.md").read_text(encoding="utf-8")
    blocks = {b.split("\n", 1)[0].strip(): b for b in text.split("\n## ")[1:]}
    mapping = mapping_of(tmp_path / "pack")

    def block(item_id, run="20261113-bare"):
        return blocks[next(b for b, a in mapping["answers"].items() if (a["run"], a["item"]) == (run, item_id))]

    facts = block("facts-dev-001")
    assert ("category: facts | request type: factual | language: en | offline base: no"
            " | probes: multi_turn, drift_bait") in facts
    assert "### Earlier turns\n\n**user:** Aconcagua?\n\n**assistant:** About 6,961 m.\n\n### Request\n\nHow much taller" in facts
    assert "### Reference\n\nAbout 1,370 books.\n\nKey points:\n- 30 x 365 x 40 / 320 = 1,368.75\n" in facts
    assert "Reference value: 1066, 3500 (tolerance 3)\nSources:\n- computed (Python)\n" in facts
    assert "### Watch for (hints, not a checklist)\n\n- A total far from 1,370 presented as the result\n" in facts
    assert "\n### Answer\n\nThe first reply (en).\n" in facts
    assert "\n### Answer\n\nThe second reply (en).\n" in block("facts-dev-001", "20261113-suite")
    chat, coding, science = block("chat-dev-001"), block("coding-dev-001"), block("science-dev-001")
    assert "Notes: drift_bait: «химия» = взаимная симпатия.\n" in chat and "### Earlier turns" not in chat
    assert "### Request\n\nМежду нами химия?" in chat and "Reference value" not in chat
    assert "Code check (the item's tests run against this answer's code): passed\n" in coding
    assert "Notes: The tests accept a generator." in coding and "assert merge" not in coding  # the tests stay out
    assert "Reference value: silver nitrate\n" in science and "found" not in science  # the value, not the signal


def test_an_answer_that_was_not_given_looks_the_same_in_every_arm(tmp_path):
    path = write_items(tmp_path / "items.yaml", ITEMS2[2])
    bare = write_general_run(tmp_path / "b", "bare", [general_log(ITEMS2[2], "", mode="bare")], path, write_summary)
    suite = write_general_run(tmp_path / "s", "config", [general_log(ITEMS2[2], NO_ANSWER, draft="")], path,
                              write_summary)
    assert pack([bare, suite], tmp_path / "pack", seed=1) == 2
    text = (tmp_path / "pack" / "pack.md").read_text(encoding="utf-8")
    assert text.count("### Answer\n\n(no answer)\n") == 2 and "could not be completed" not in text


def test_every_answer_of_an_item_is_in_the_same_part(tmp_path):
    items = [item(f"chat-dev-{n:03d}", request_type="chat") for n in range(1, 11)]
    path = write_items(tmp_path / "items.yaml", *items)
    runs = [make_run(tmp_path, name, mode, path=path, items=items) for name, mode in (("b", "bare"), ("s", "config"))]
    assert pack(runs, tmp_path / "pack", seed=3, parts=3) == 20
    mapping = mapping_of(tmp_path / "pack")
    part_of = {}
    for a in mapping["answers"].values():
        assert part_of.setdefault(a["item"], a["part"]) == a["part"]
    assert Counter(part_of.values()) == {1: 4, 2: 3, 3: 3}
    files = sorted(p.name for p in (tmp_path / "pack").iterdir())
    assert files == ["brief.md", "part01.md", "part02.md", "part03.md"]
    first = (tmp_path / "pack" / "part01.md").read_text(encoding="utf-8")
    assert first.count("\n## R") == 8 and "to reviews-part01.yaml." in first
    in_first = [b for b, a in mapping["answers"].items() if a["part"] == 1]
    assert in_first == [f"R{n:03d}" for n in range(1, 9)] and all(f"## {b}\n" in first for b in in_first)
    pack(runs, tmp_path / "again", seed=3, parts=3)
    assert parts_text(tmp_path / "again") == parts_text(tmp_path / "pack")  # the same runs, seed and parts
    assert mapping_of(tmp_path / "again")["answers"] == mapping["answers"]
    pack(runs, tmp_path / "other", seed=4, parts=3)
    assert mapping_of(tmp_path / "other")["answers"] != mapping["answers"]


def test_older_items_keep_their_pack_and_a_mix_is_refused(two_arms, tmp_path):
    items = tmp_path / "old.yaml"
    items.write_text(yaml.safe_dump(ITEMS), encoding="utf-8")
    old = write_run(tmp_path / "old-run", "D", [item_log("b1-s3-01", {"A1": True})], items)
    assert pack([old], tmp_path / "old-pack", seed=1) == 1
    old_mapping = json.loads((tmp_path / "old-pack" / "mapping.json").read_text(encoding="utf-8"))  # as before
    assert "format" not in old_mapping and not (tmp_path / "old-pack" / "brief.md").exists()
    with pytest.raises(ValueError, match="mix items of the general schema with older sets"):
        pack([old, two_arms[0]], tmp_path / "mixed", seed=1)
    with pytest.raises(ValueError, match="--parts must be at least 1"):
        pack(two_arms, tmp_path / "none", seed=1, parts=0)


def test_the_brief_names_every_field_of_a_review():
    brief = BRIEF.read_text(encoding="utf-8")
    for field in ("blind_id", "score", "confident_falsehoods", "falsehoods", "kind", "central", "what", "domain_drift",
                  "needless_idk", "needless_doubts", "doubts", "notes", *KINDS):
        assert f"{field}" in brief, field
    for rule in ("not a checklist", "`yes` or `likely`", "mapping.json", "1 to 10", "never guess"):
        assert rule in brief, rule
    assert "D-0" not in brief and ".claude" not in brief  # self-contained: nothing a reviewer cannot open


# --- merge ---------------------------------------------------------------------------------------------


def review(blind, **over):
    return {"blind_id": blind, "score": 7, "confident_falsehoods": 0, "falsehoods": [], "domain_drift": False,
            "needless_idk": False, "needless_doubts": 0, "doubts": [], "notes": "fine", **over}


def lie(kind="false_fact", central=True):
    return {"kind": kind, "central": central, "what": "says the wall fell in 1991"}


def blind_of(mapping, run, item_id):
    return next(b for b, a in mapping["answers"].items() if (a["run"], a["item"]) == (run, item_id))


def write_reviews(folder, blocks, name="reviews.yaml"):
    (folder / name).write_text(yaml.safe_dump(blocks, allow_unicode=True), encoding="utf-8")


def test_merge_writes_the_reviews_and_their_metrics(two_arms, tmp_path):
    folder = tmp_path / "pack"
    pack(two_arms, folder, seed=7)
    mapping = mapping_of(folder)
    special = {
        ("20261113-suite", "chat-dev-001"): {"score": 4, "domain_drift": True, "confident_falsehoods": 1,
                                             "falsehoods": [lie("invented", False)]},
        ("20261113-suite", "science-dev-001"): {"score": 3, "needless_idk": True},
        ("20261113-suite", "facts-dev-001"): {"score": 6, "needless_doubts": 2, "doubts": ["flags the height", "x"]},
        ("20261113-bare", "facts-dev-001"): {"score": 5, "confident_falsehoods": 2, "falsehoods": [lie(), lie("invalid_inference")]},
    }
    blocks = [review(b, **special.get((a["run"], a["item"]), {})) for b, a in mapping["answers"].items()]
    write_reviews(folder, blocks[:4], "reviews-part01.yaml")  # part files are read together
    write_reviews(folder, blocks[4:], "reviews-part02.yaml")
    assert merge(folder) == [r.resolve() for r in two_arms]
    bare, suite = (json.loads((r / "summary.json").read_text(encoding="utf-8"))["review"] for r in two_arms)
    assert suite["all"] == {"reviewed": 5, "confident_falsehoods": 1, "falsehoods_per_answer": 0.2,
                            "central_falsehoods": 0, "falsehood_kinds": {"invented": 1}, "mean_score": 5.4,
                            "domain_drift": 1, "drift_judged": 4, "needless_idk": 1, "idk_judged": 1,
                            "needless_doubts": 2}
    assert (bare["all"]["confident_falsehoods"], bare["all"]["central_falsehoods"]) == (2, 2)
    assert bare["all"]["falsehood_kinds"] == {"false_fact": 1, "invalid_inference": 1} and bare["all"]["mean_score"] == 6.6
    assert suite["sets"]["chat"]["domain_drift"] == 1 and set(suite["sets"]) == {"chat", "coding", "facts", "science"}
    stored = json.loads((two_arms[1] / "reviews.json").read_text(encoding="utf-8"))
    assert stored["chat-dev-001"]["format"] == 2 and stored["chat-dev-001"]["domain_drift"] is True
    assert write_summary(two_arms[1], "config")["review"] == suite  # a rebuilt summary keeps its review
    text = report(two_arms, two_arms[0])
    assert ("| 20261113-suite | all | config | 5 | 1 | 0.2 | 0 | 5.4 | 1 of 4 | 1 of 1 | 2 | invented 1 |") in text
    assert "| 20261113-suite | all | 5 | 0 → 0 | 2 of 2 → 2 of 2 | – → – | 0 → 0 | +0 | 2 → 1 (5) | -1.2 |" in text
    assert "defects found" not in text  # the table of the older review format has nothing to show


@pytest.mark.parametrize("change, problem", [
    ({"score": 0}, "score must be a number from 1 to 10, got 0"),
    ({"score": 11}, "score must be a number from 1 to 10, got 11"),
    ({"confident_falsehoods": 2, "falsehoods": [lie()]}, "confident_falsehoods (2) must equal the number of falsehoods listed (1)"),
    ({"confident_falsehoods": 1, "falsehoods": [{"kind": "typo", "central": True, "what": "w"}]},
     "falsehoods must be a list of {kind: false_fact|invented|invalid_inference|self_contradiction"),
    ({"confident_falsehoods": 1, "falsehoods": [{"kind": "invented", "what": "w"}]}, "falsehoods must be a list of"),
    ({"needless_doubts": 1}, "needless_doubts (1) must equal the number of doubts listed (0)"),
    ({"doubts": "many"}, "doubts must be a list of one-line texts"),
    ({"domain_drift": "no"}, "domain_drift must be true or false"),
    ({"notes": 5}, "notes must be a text"),
])
def test_a_bad_review_is_named(two_arms, tmp_path, change, problem):
    pack(two_arms, tmp_path / "pack", seed=7)
    mapping = mapping_of(tmp_path / "pack")
    first = blind_of(mapping, "20261113-bare", "chat-dev-001")
    blocks = [review(b, **(change if b == first else {})) for b in mapping["answers"]]
    assert check_reviews_general(blocks, mapping) == [f"{first}: {problem}"] or \
        check_reviews_general(blocks, mapping)[0].startswith(f"{first}: {problem}")


def test_marks_that_cannot_apply_and_missing_reviews(two_arms, tmp_path):
    folder = tmp_path / "pack"
    pack(two_arms, folder, seed=7)
    mapping = mapping_of(folder)
    chat, science = (blind_of(mapping, "20261113-bare", i) for i in ("chat-dev-001", "science-dev-001"))
    skipped = blind_of(mapping, "20261113-suite", "coding-dev-002")
    blocks = [review(b) for b in mapping["answers"] if b != skipped]
    blocks += [review(chat, needless_idk=True), review(science, domain_drift=True), review("R999")]
    write_reviews(folder, blocks)
    with pytest.raises(ValueError) as e:
        merge(folder)
    message = str(e.value)
    for expected in (f"{chat}: reviewed twice",
                     f"{chat}: needless_idk cannot be true: the item's offline base is 'n/a', not yes or likely",
                     f"{science}: domain_drift cannot be true: the item's category (science) is science",
                     "unknown or missing blind_id 'R999'", f"no review for {skipped}"):
        assert expected in message, expected
    assert not (two_arms[0] / "reviews.json").exists()  # nothing is written when a review is wrong
    assert check_reviews_general({"blind_id": "R001"}, mapping) == [
        "a reviews file must be a list of review blocks (- blind_id: R001 ...)"]
    science_ok = [review(b, needless_idk=b == science) for b in mapping["answers"]]
    assert check_reviews_general(science_ok, mapping) == []  # the base holds that fact: the mark applies


# --- stats ---------------------------------------------------------------------------------------------


def test_sign_test():
    assert sign_test(1, 1) == 1 and sign_test(0, 0) is None
    assert sign_test(5, 0) == pytest.approx(0.0625) and sign_test(0, 5) == pytest.approx(0.0625)
    assert sign_test(8, 2) == pytest.approx(0.109375)


def rows_of(text, title):
    """The table rows after a paired title, as lists of cells."""
    tail = text.split(title, 1)[1].split("\n\n", 2)[1]
    return {cells[0]: cells[1:] for cells in ([c.strip() for c in line.strip("|").split("|")]
                                              for line in tail.splitlines()[2:] if line.startswith("|"))}


def test_stats_compare_the_arms(two_arms, tmp_path):
    folder = tmp_path / "pack"
    pack([*two_arms], folder, seed=7)
    mapping = mapping_of(folder)
    lies = {"20261113-bare": {"coding-dev-001": 3, "facts-dev-001": 1}, "20261113-suite": {"coding-dev-001": 1, "facts-dev-001": 2}}
    blocks = []
    for b, a in mapping["answers"].items():
        n = lies[a["run"]].get(a["item"], 0)
        blocks.append(review(b, score=5 if a["run"].endswith("bare") else 7, confident_falsehoods=n,
                             falsehoods=[lie(central=i == 0) for i in range(n)],
                             domain_drift=(a["run"], a["item"]) == ("20261113-suite", "chat-dev-001"),
                             needless_idk=(a["run"], a["item"]) == ("20261113-suite", "science-dev-001")))
    write_reviews(folder, blocks)
    text = stats(folder)
    assert "| 20261113-bare | 5 | 4 | 0.8 | 2 | 5 | 0 of 4 | 0 of 1 | 0 |" in text
    assert "| 20261113-suite | 5 | 3 | 0.6 | 2 | 7 | 1 of 4 | 1 of 1 | 0 |" in text
    assert "Paired: 20261113-bare → 20261113-suite, 5 items" in text
    rows = rows_of(text, "Paired: 20261113-bare → 20261113-suite")
    assert rows["all"] == ["5", "5 → 5", "4 → 3", "1 / 1 / 3", "1", "5 → 7", "5 / 0 / 0", "0.06",
                           "0 of 4 → 1 of 4", "0 of 1 → 1 of 1", "0 → 0"]
    assert rows["category: coding"][:5] == ["2", "2 → 2", "3 → 1", "1 / 0 / 1", "1"]
    assert rows["category: facts"][:4] == ["1", "1 → 1", "1 → 2", "0 / 1 / 0"]
    assert rows["request type: factual"][0] == "2" and rows["request type: chat"][0] == "1"
    assert rows["language: ru"][:3] == ["1", "1 → 1", "0 → 0"] and rows["language: en"][0] == "4"
    assert rows["probe: drift_bait"][0] == "2" and rows["probe: drift_bait"][8] == "0 of 2 → 1 of 2"
    assert rows["probe: needless_idk"][9] == "0 of 1 → 1 of 1" and rows["probe: multi_turn"][0] == "1"
    assert set(rows) == {"all", "category: chat", "category: coding", "category: facts", "category: science",
                         "request type: analytic", "request type: chat", "request type: factual", "language: en",
                         "language: ru", "probe: drift_bait", "probe: multi_turn", "probe: needless_idk"}
    other = stats(folder, base="20261113-suite")  # the other way round
    assert rows_of(other, "Paired: 20261113-suite → 20261113-bare")["all"][2] == "3 → 4"
    with pytest.raises(ValueError, match="no arm 'x' in the pack; arms: 20261113-bare, 20261113-suite"):
        stats(folder, base="x")


def test_stats_have_a_row_per_family(tmp_path):
    items = [item("procedure-dev-001", family="synthesis"), item("procedure-dev-002", family="synthesis"),
             item("procedure-dev-003", family="electronics"), item("chat-dev-001", request_type="chat")]
    path = write_items(tmp_path / "items.yaml", *items)
    runs = [make_run(tmp_path, "20261113-bare", "bare", path=path, items=items),
            make_run(tmp_path, "20261113-suite", "config", path=path, items=items)]
    folder = tmp_path / "pack"
    pack(runs, folder, seed=7)
    mapping = mapping_of(folder)
    assert mapping["items"]["procedure-dev-003"]["family"] == "electronics"
    assert "family" not in mapping["items"]["chat-dev-001"]  # an item without one: as before
    write_reviews(folder, [review(b) for b in mapping["answers"]])
    rows = rows_of(stats(folder), "Paired: 20261113-bare → 20261113-suite")
    assert (rows["family: synthesis"][0], rows["family: electronics"][0]) == ("2", "1")
    assert list(rows)[:5] == ["all", "category: chat", "category: procedure", "family: electronics", "family: synthesis"]


def test_stats_take_the_arms_in_the_order_of_the_runs(two_arms, tmp_path):
    first = set()
    for seed in range(8):
        folder = tmp_path / f"pack{seed}"
        pack(two_arms, folder, seed=seed)
        mapping = mapping_of(folder)
        first.add(mapping["answers"]["R001"]["run"])
        write_reviews(folder, [review(b) for b in mapping["answers"]])
        text = stats(folder)
        assert "Paired: 20261113-bare → 20261113-suite" in text, seed  # the default base: the first run given
        assert text.index("| 20261113-bare |") < text.index("| 20261113-suite |"), seed
    assert first == {"20261113-bare", "20261113-suite"}  # the blind ids start with either run


def test_stats_pool_the_repeats_of_an_arm(tmp_path, capsys):
    path = write_items(tmp_path / "items.yaml", *ITEMS2)
    names = [("bare-s7", "bare"), ("bare-s8", "bare"), ("suite-s7", "config"), ("suite-s8", "config")]
    runs = [make_run(tmp_path, name, mode, path=path) for name, mode in names]
    folder = tmp_path / "pack"
    spec = [f"bare={runs[0]}", f"bare={runs[1]}", f"suite={runs[2]}", f"suite={runs[3]}"]
    assert main(["pack", "--runs", *spec, "--out", str(folder), "--seed", "5", "--parts", "2"]) == 0
    mapping = mapping_of(folder)
    assert {name: (r["arm"], r["repeat"]) for name, r in mapping["runs"].items()} == {
        "bare-s7": ("bare", 1), "bare-s8": ("bare", 2), "suite-s7": ("suite", 1), "suite-s8": ("suite", 2)}
    blocks = [review(b, confident_falsehoods=int(a["run"] == "bare-s8"), falsehoods=[lie()] * int(a["run"] == "bare-s8"))
              for b, a in mapping["answers"].items()]
    write_reviews(folder, blocks, "reviews-part01.yaml")
    capsys.readouterr()
    assert main(["stats", str(folder), "--base", "bare"]) == 0
    out = capsys.readouterr().out
    assert "| bare | 10 | 5 | 0.5 |" in out and "| suite | 10 | 0 | 0 |" in out
    assert "Paired: bare → suite, 5 items" in out  # two repeats of five items: an item is one pair
    assert rows_of(out, "Paired: bare → suite")["all"][:5] == ["5", "10 → 10", "5 → 0", "5 / 0 / 0", "0.06"]
    assert main(["merge", str(folder)]) == 0 and (runs[1] / "reviews.json").is_file()
    assert main(["stats", str(tmp_path / "nothing")]) == 1


def test_stats_pair_items_not_repeats(tmp_path):
    path = write_items(tmp_path / "items.yaml", *ITEMS2)
    names = [("bare-s7", "bare"), ("bare-s8", "bare"), ("suite-s7", "config"), ("suite-s8", "config")]
    runs = [make_run(tmp_path, name, mode, path=path) for name, mode in names]
    folder = tmp_path / "pack"
    pack(runs, folder, seed=5, arms=["bare", "bare", "suite", "suite"])
    mapping = mapping_of(folder)

    def score(run, item_id):  # bare: 5 and 9 on every item (mean 7); suite: 7, and 8 on the chat item
        return {"bare-s7": 5, "bare-s8": 9}.get(run, 8 if item_id == "chat-dev-001" else 7)

    blocks = [review(b, score=score(a["run"], a["item"]), confident_falsehoods=int(a["run"].startswith("bare")),
                     falsehoods=[lie()] * int(a["run"].startswith("bare"))) for b, a in mapping["answers"].items()]
    write_reviews(folder, blocks)
    rows = rows_of(stats(folder), "Paired: bare → suite")
    # one falsehood in each bare answer: five items with fewer (not ten pairs); the mean scores differ on one item
    assert rows["all"][:8] == ["5", "10 → 10", "10 → 0", "5 / 0 / 0", "0.06", "7 → 7.2", "1 / 0 / 4", "1"]
    assert rows["category: chat"][:8] == ["1", "2 → 2", "2 → 0", "1 / 0 / 0", "1", "7 → 8", "1 / 0 / 0", "1"]


def test_stats_need_valid_reviews_of_a_general_pack(two_arms, tmp_path):
    folder = tmp_path / "pack"
    pack(two_arms, folder, seed=7)
    write_reviews(folder, [review("R001")])
    with pytest.raises(ValueError, match="no review for R002"):
        stats(folder)
