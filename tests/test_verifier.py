"""Claim verifier (claim-verifier spec): split, numbers and formulas in code, judge, re-search, labels."""

import pytest
from fastapi.testclient import TestClient

from conftest import CONNECT_ERROR, content, finish, tool_call
from haisenberg.agent import Agent
from haisenberg.l1.store import ArticleRecord, Citation, L1Store
from haisenberg.l2.store import PassageRecord
from haisenberg.server import create_app
from haisenberg.llm import estimate_tokens
from haisenberg.tools import Knowledge, Source, Sources
from haisenberg.verifier import (
    CORRECTED,
    NO_CLAIM,
    REMOVED,
    SUPPORTED,
    UNVERIFIED,
    code_check,
    echoes,
    fix_arithmetic,
    is_computation,
    minimal_edit,
    parse_fixes,
    parse_verdicts,
    quoted,
    read_formulas,
    read_numbers,
    reference_conflict,
    render,
    results,
    run_checkers,
    Verifier,
    split_claims,
    unconfirmed,
)
from test_server import chat, events

# --- split -----------------------------------------------------------------------------------------


def texts(draft):
    return [c.text for c in split_claims(draft)]


def test_sentences_and_list_items():
    draft = "Caffeine is C8H10N4O2 [S1]. Its molar mass is 194.19 g/mol.\n- It is a stimulant, e.g. in coffee.\n"
    claims = split_claims(draft)
    assert [c.text for c in claims] == [
        "Caffeine is C8H10N4O2 [S1].", "Its molar mass is 194.19 g/mol.", "It is a stimulant, e.g. in coffee.",
    ]
    assert claims[0].cites == ["S1"] and claims[1].cites == []
    assert all(draft[c.start : c.end] == c.text for c in claims)


def test_not_claims():
    draft = "## Result\n**Working:**\n```\nx = 1.5 * 2\n```\n$$M = 194.19$$\n| a | 1 |\nWhy?\nOk.\n"
    assert texts(draft) == []


def test_trailing_citation_and_abbreviations():
    draft = "Ethanol boils at approx. 78 °C. [S1] Water boils at 100 °C [S2]. It is a liquid, i.e. not a gas."
    assert texts(draft) == [
        "Ethanol boils at approx. 78 °C. [S1]", "Water boils at 100 °C [S2].", "It is a liquid, i.e. not a gas.",
    ]
    assert split_claims(draft)[0].cites == ["S1"]


def test_numbered_list_and_quote_markers():
    assert texts("1. The value is 6.626e-34 J s.\n> Speed of light is 299 792 458 m/s.") == [
        "The value is 6.626e-34 J s.", "Speed of light is 299 792 458 m/s.",
    ]


def test_labelled_answer_and_verdict_lines_are_not_claims():
    draft = ("A1: S2, S4 and S7\n- **A2:** Correct, the sample weighs 3.1 g.\n**B3**: 42 kJ\nClaim D: Incorrect, it is 5 g.\n"
             "Statement 2: false, see 12 mL\n(c): it is 7 g heavier\nNote: water boils at 100 °C [S1].\nА1: 4,2 г\n")
    assert len(split_claims(draft)) == 7  # the Russian line has no Latin letter: never a claim
    assert [c.text for c in split_claims(draft, skip_labelled=True)] == ["Note: water boils at 100 °C [S1]."]


def test_bare_labels_are_not_claims():
    draft = "**S3. Harbour Instruments Ltd was founded in 1994.** ✗ WRONG\n**C2.\n- A1)\nVitamin B12. H2O.\n"
    assert texts(draft) == ["Harbour Instruments Ltd was founded in 1994.** ✗ WRONG", "Vitamin B12.", "H2O."]


# --- numbers and formulas --------------------------------------------------------------------------


@pytest.mark.parametrize("text, value", [
    ("c = 299 792 458 m/s", 299792458),
    ("about 10,000 K", 10000),
    (r"$2\,450$ J", 2450),
    ("h = 6.626 070 15 e-34 J Hz^-1", 6.62607015e-34),
    ("m_e = 9.109 383 7139 × 10^-31 kg", 9.1093837139e-31),  # CODATA: the last group may have four digits
    ("m_e = 9.109 383 7139\xa0× 10^-31 kg", 9.1093837139e-31),  # thin and no-break spaces
    ("h = 6.626e-34", 6.626e-34),
    ("h = 6.626 × 10^-34 J s", 6.626e-34),
    ("h = 6.626×10⁻³⁴ J·s", 6.626e-34),
    (r"$6.626 \times 10^{-34}$", 6.626e-34),
    ("melts at −114 °C", -114),
    ("78.37 °C", 78.37),
])
def test_number_notations(text, value):
    (n,) = read_numbers(text)
    assert n.value == pytest.approx(value)


@pytest.mark.parametrize("text, values", [
    ("1.5 1000 K", [1.5, 1000]),  # a four-digit group only after a three-digit one
    ("0.1234 5678", [0.1234, 5678]),
])
def test_four_digits_after_a_decimal_are_another_number(text, values):
    assert [n.value for n in read_numbers(text)] == pytest.approx(values)


def test_digits_in_words_are_not_numbers():
    assert read_numbers("Carbon-14 decays to nitrogen-14 [S2] in C8H10N4O2 and U-235.") == []


def test_precision_of_a_number():
    (n,) = read_numbers("194.2 g/mol")
    assert n.half == pytest.approx(0.05) and n.decimals == 1
    (n,) = read_numbers("3 × 10^8 m/s")
    assert n.half == pytest.approx(0.5e8) and n.scientific


def test_formulas():
    assert [h for h, _, _ in read_formulas("ethanol, CH3CH2OH or C₂H₆O, not NaCl [S1]")] == ["C2H6O", "C2H6O"]


# --- code checks -----------------------------------------------------------------------------------


@pytest.fixture
def store(built_store):
    return L1Store(built_store)


def sources_with(store, *records):
    s = Sources()
    for r in records:
        s.add("x", r, r.passage("", 1500))
    return s


def claim(text):
    return split_claims(text)[0]


def test_rounded_molar_mass_supported(store):
    caffeine = store.lookup_compound("caffeine")[0]
    s = sources_with(store, caffeine)
    outcome, ids, fixed, note = code_check(claim("The molar mass of caffeine is 194.2 g/mol [S1]."), s, 0.001)
    assert (outcome, ids, fixed) == (SUPPORTED, ["S1"], None) and "194.2" in note


def test_wrong_molar_mass_fixed(store):
    s = sources_with(store, store.lookup_compound("caffeine")[0])
    outcome, ids, fixed, _ = code_check(claim("Caffeine has a molar mass of 180.16 g/mol [S1]."), s, 0.001)
    assert outcome == CORRECTED and ids == ["S1"] and "194.19 g/mol" in fixed and "180.16" not in fixed


def test_constant_in_scientific_notation_supported_uncited(store):
    s = sources_with(store, store.lookup_compound("caffeine")[0], store.lookup_constant("Planck constant")[0])
    outcome, ids, _, _ = code_check(claim("The quantum of action is h = 6.626 × 10⁻³⁴ J·s."), s, 0.001)
    assert outcome == SUPPORTED and ids == ["S2"]


def test_wrong_constant_fixed(store):
    s = sources_with(store, store.lookup_constant("Planck constant")[0])
    outcome, ids, fixed, _ = code_check(claim("The Planck constant is 6.26 × 10^-34 J s [S1]."), s, 0.001)
    assert outcome == CORRECTED and "6.626 × 10^-34" in fixed


def test_compound_value_needs_citation_or_name(store):
    s = sources_with(store, store.lookup_compound("caffeine")[0])
    assert code_check(claim("Some substance has a molar mass of 194.19 g/mol."), s, 0.001) is None
    assert code_check(claim("Caffeine has a molar mass of 194.19 g/mol."), s, 0.001)[0] == SUPPORTED


def test_formula_support_and_contradiction(store):
    s = sources_with(store, store.lookup_compound("caffeine")[0])
    assert code_check(claim("The formula of caffeine is C8H10N4O2 [S1]."), s, 0.001)[0] == SUPPORTED
    outcome, _, fixed, _ = code_check(claim("The formula of caffeine is C8H10N4O3 [S1]."), s, 0.001)
    assert outcome == CORRECTED and "C8H10N4O2" in fixed


def test_melting_point_citing_a_compound_goes_to_the_judge(store):
    s = sources_with(store, store.lookup_compound("caffeine")[0])
    assert code_check(claim("Caffeine melts at 235 °C [S1]."), s, 0.001) is None


def test_every_number_must_match(store):
    s = sources_with(store, store.lookup_compound("caffeine")[0])
    assert code_check(claim("Caffeine (194.19 g/mol) is found in 95 mg per cup [S1]."), s, 0.001) is None


PUMP_TASK = "A pump lifts 0.04 m³/s of water by 6 m; its motor gives 2.2 kW. Is the motor big enough?"
PUMP_DRAFT = (
    "P = 1000 × 9.81 × 0.04 × 6 = 2354 W [S1]. So 2354 W is more than the 2.2 kW of the motor. "
    "The margin is (2.2 − 2.354)/2.354 = −6.5 %. A motor of 2.2 kW is too small. "
    "Water has a density of 1000 kg/m³ [S1]. The pump lifts 0.04 m³/s."
)


def test_computations_are_recognised():
    claims = split_claims(PUMP_DRAFT)
    given, computed = [n.value for n in read_numbers(PUMP_TASK)], results(PUMP_DRAFT)
    assert computed == [2354, -6.5]  # "= 1000 × …" starts an expression: not a result
    # equation; computed vs given; equation; only given numbers; a value of a source; restated given data
    assert [is_computation(c, given, computed, 0.001) for c in claims] == [True, True, True, False, False, False]


# --- judge reply -----------------------------------------------------------------------------------


def test_parse_verdicts():
    reply = "C1: supports S1\n**C2** - contradicts S2 | Ethanol boils at 78.23 °C.\nC3: no_claim\nC4: supports S9\nnoise"
    v = parse_verdicts(reply, {"S1", "S2"})
    assert v[1] == ("supports", ["S1"], None)
    assert v[2] == ("contradicts", ["S2"], "Ethanol boils at 78.23 °C.")
    assert v[3][0] == "no_claim" and v[4][0] == "not_said" and 5 not in v


# --- labels ----------------------------------------------------------------------------------------


def test_mixed_answer_rendering():
    draft = "Caffeine is a stimulant [S2]. It is bitter.\n- Its mass is 180 g/mol [S1].\n- It is blue [S3].\nHope this helps."
    c = split_claims(draft)
    c[0].outcome, c[0].ids = SUPPORTED, ["S2"]
    c[1].outcome = UNVERIFIED
    c[2].outcome, c[2].ids, c[2].fixed = CORRECTED, ["S1"], "Its mass is 194.19 g/mol [S1]."
    c[3].outcome, c[3].ids = REMOVED, ["S3"]
    c[4].outcome = NO_CLAIM
    out = render(draft, c)
    assert out.startswith("Caffeine is a stimulant [S2]. It is bitter *(unverified)*.\n"
                          "- Its mass is 194.19 g/mol *(corrected from [S1])*.\nHope this helps.")
    assert "blue" not in out
    assert out.endswith("*Verification: 4 claims: 1 supported, 1 unverified, 1 corrected, 1 removed.*")


def test_plain_mixed_answer_rendering():
    draft = "Caffeine is a stimulant [S2]. It is bitter.\n- Its mass is 180 g/mol [S1].\n- It is blue [S3].\nHope this helps."
    c = split_claims(draft)
    c[0].outcome, c[0].ids = SUPPORTED, ["S2"]
    c[1].outcome = UNVERIFIED
    c[2].outcome, c[2].ids, c[2].fixed = CORRECTED, ["S1"], "Its mass is 194.19 g/mol [S1]."
    c[3].outcome, c[3].ids = REMOVED, ["S3"]
    c[4].outcome = NO_CLAIM
    assert render(draft, c, annotated=False) == (
        "Caffeine is a stimulant. It is bitter.\n- Its mass is 194.19 g/mol.\nHope this helps."
    )


# --- through the loop -------------------------------------------------------------------------------


@pytest.fixture
def knowledge(cfg, built_store):
    return Knowledge(L1Store(built_store), cfg)


@pytest.fixture
def client(cfg, fake, knowledge):
    """Annotated answers: these tests read the labels (the plain default is tested below)."""
    cfg.verifier.answer_style = "annotated"
    with TestClient(create_app(cfg, fake.client(cfg.model_backend), knowledge)) as c:
        yield c


def deltas(r):
    return [e["choices"][0]["delta"] for e in events(r.text) if e != "[DONE]"]


def answer(r):
    return "".join(d.get("content", "") for d in deltas(r))


def reasoning_text(r):
    return "".join(d.get("reasoning_content", "") for d in deltas(r))


SEARCH_ETHANOL = [tool_call("search_knowledge", '{"query": "ethanol"}', call_id="c1"), finish("tool_calls")]


def test_verification_off(cfg, fake, built_store):
    cfg.reasoning_policy.enable_multi_hop_verification = False
    fake.scripts = [SEARCH_ETHANOL, [content("Ethanol is a liquid [S1].")]]
    with TestClient(create_app(cfg, fake.client(cfg.model_backend), Knowledge(L1Store(built_store), cfg))) as c:
        r = chat(c)
    assert len(fake.requests) == 2 and "Verification" not in answer(r)


def test_batched_judge_and_order_of_the_stream(client, fake, cfg):
    fake.scripts = [
        SEARCH_ETHANOL,
        [content("Ethanol is a volatile, flammable, colorless liquid [S1]. It is blue [S1]. Let me know more.")],
        [content("C1: supports S1\nC2: contradicts S1 | Ethanol is colorless.\nC3: no_claim")],
    ]
    r = chat(client, [{"role": "user", "content": "What is ethanol?"}])
    judge = fake.bodies[2]
    assert len(fake.requests) == 3
    assert judge["messages"][0] == {"role": "system", "content": cfg.verifier.judge_prompt}
    user = judge["messages"][1]["content"]
    assert user.startswith("Question: What is ethanol?") and "[S1] Ethanol\n" in user
    assert "C1: Ethanol is a volatile, flammable, colorless liquid." in user and "[S1]." not in user
    for k, v in cfg.verifier.request.items():
        assert judge[k] == v
    assert "tools" not in judge and "chat_template_kwargs" not in fake.bodies[0]
    text = answer(r)
    assert text.startswith("Ethanol is a volatile, flammable, colorless liquid [S1]. Ethanol is colorless "
                           "*(corrected from [S1])*. Let me know more.")
    assert "*Verification: 2 claims: 1 supported, 0 unverified, 1 corrected, 0 removed.*" in text
    assert "\n\nSources:\n- [S1] Ethanol" in text
    kinds = [next(iter(d)) for d in deltas(r) if d]
    assert kinds[-1] == "content" and kinds.count("content") == 1
    steps = reasoning_text(r)
    assert "Verification: 3 claims" in steps and "C2 judge: contradicts S1" in steps


def test_plain_answer_is_silent(cfg, fake, knowledge):
    fake.scripts = [
        SEARCH_ETHANOL,
        [content("Ethanol is a volatile, flammable, colorless liquid [S1]. It is blue [S1]. Let me know more.")],
        [content("C1: supports S1\nC2: contradicts S1 | Ethanol is colorless.\nC3: no_claim")],
    ]
    with TestClient(create_app(cfg, fake.client(cfg.model_backend), knowledge)) as c:
        r = chat(c)
    assert answer(r) == "Ethanol is a volatile, flammable, colorless liquid. Ethanol is colorless. Let me know more."
    steps = reasoning_text(r)  # the trace stays in the reasoning block
    assert "C2 judge: contradicts S1" in steps and "C2 corrected from S1" in steps


def test_found_on_re_search(client, fake):
    fake.scripts = [
        [content("Benzene boils at 80.1 °C.")],  # no tool call, no citation: straight to re-search
        [content("C1: supports S1")],
    ]
    r = chat(client)
    assert len(fake.requests) == 2  # draft + one judge call (the first judge is skipped: nothing cited)
    assert "Benzene boils at 80.1 °C [S1]." in answer(r)
    assert "re-search C1" in reasoning_text(r)


def test_still_unsupported(client, fake):
    fake.scripts = [[content("Benzene boils at 80.1 °C.")], [content("C1: not_said")]]
    assert "Benzene boils at 80.1 °C *(unverified)*." in answer(chat(client))


def test_judge_failure_leaves_claims_unverified(client, fake, caplog):
    fake.scripts = [SEARCH_ETHANOL, [content("Ethanol is a colorless liquid [S1].")], CONNECT_ERROR, CONNECT_ERROR]
    r = chat(client)
    assert r.status_code == 200 and "[DONE]" in r.text
    assert "*(unverified)*" in answer(r) and "judge failed" in reasoning_text(r)


def test_code_check_needs_no_judge(client, fake):
    fake.scripts = [
        [tool_call("lookup_compound", '{"query": "caffeine"}', call_id="c1"), finish("tool_calls")],
        [content("The molar mass of caffeine is 194.19 g/mol [S1].")],
    ]
    r = chat(client)
    assert len(fake.requests) == 2
    assert answer(r).startswith("The molar mass of caffeine is 194.19 g/mol [S1].\n\n*Verification: 1 claim: 1 supported")


@pytest.mark.anyio
async def test_report_for_in_process_callers(cfg, fake, knowledge):
    fake.scripts = [[content("Benzene boils at 80.1 °C.")], [content("C1: supports S1")]]
    items = [i async for i in Agent(cfg, fake.client(cfg.model_backend), knowledge).run(
        [{"role": "user", "content": "q"}])]
    report = next(i["verification"] for i in items if "verification" in i)
    assert report["counts"] == {"supported": 1, "unverified": 0, "corrected": 0, "removed": 0}
    assert report["claims"][0]["ids"] == ["S1"]


def test_report_not_on_the_wire(client, fake):
    fake.scripts = [[content("Benzene boils at 80.1 °C.")], [content("C1: supports S1")]]
    r = chat(client)
    assert "verification" not in r.text.replace("Verification", "")
    body = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "q"}]}).json()
    assert "verification" not in body["choices"][0]["message"]



class _Rec:
    def __init__(self, name):
        self.name = name

    def label(self):
        return self.name


@pytest.mark.anyio
async def test_judge_leaves_out_passages_that_do_not_fit(cfg, fake, knowledge):
    v = Verifier(cfg, fake.client(cfg.model_backend), knowledge)
    claims = split_claims("Ethanol is a liquid [S1]. It boils at 78 degrees [S2].")
    passages = [Source(f"S{i}", "text", _Rec(f"P{i}"), "y" * 2000) for i in (1, 2, 3)]
    nl = chr(10)
    bare = f"Question: q{nl}{nl}Passages:{nl}{nl}{nl}Claims:{nl}" + nl.join(f"C{c.n}: x" for c in claims)
    v.budget = estimate_tokens([{"role": "system", "content": cfg.verifier.judge_prompt},
                                {"role": "user", "content": bare}], cfg.model_backend.chars_per_token) + 1500
    fake.scripts = [[content("C1: supports S1" + nl + "C2: supports S2")]]
    lines = "".join([item["reasoning_content"] async for item in v._judge("q", claims, passages)])
    user = fake.bodies[0]["messages"][1]["content"]
    assert "[S1] P1" in user and "[S2] P2" not in user and "[S3] P3" not in user
    assert "S3 left out of the judge call" in lines and "S2 left out of the judge call" in lines
    assert estimate_tokens(fake.bodies[0]["messages"], cfg.model_backend.chars_per_token) <= v.budget
    assert (claims[0].outcome, claims[1].outcome) == (SUPPORTED, "")  # S2 was not given: not supported


# --- consistency review -------------------------------------------------------------------------------

DRAFT = "The theoretical yield is 23.75 g, so 23.4 g is a 98.6 % yield.\n\n| Check | Result |\n|---|---|\n| Yield | 85 % (consistent) |"
REVIEW = ("OLD: 85 % (consistent)\nNEW: 98.6 % (implausibly high)\nCONFLICT: so 23.4 g is a 98.6 % yield\n"
          "WHY: the calculation above gives 98.6 %")


@pytest.fixture
def reviewing(cfg, fake, knowledge):
    """A plain-answer client with the consistency review on; the draft cites nothing, so the only judge
    call is the re-search one, answered not_said."""
    cfg.verifier.consistency_check = True
    with TestClient(create_app(cfg, fake.client(cfg.model_backend), knowledge)) as c:
        yield c


def test_parse_fixes():
    assert parse_fixes("OK") == []
    assert parse_fixes('**OLD:** "a | b"\n**NEW**: c\nWHY: d\nOLD: lone') == [{"old": "a | b", "new": "c", "why": "d"}]
    assert parse_fixes("OLD: a\nNEW: b\n**CONFLICT:** `c = 2`\nWHY: d") == [
        {"old": "a", "new": "b", "why": "d", "conflict": "c = 2"}]


def test_conclusion_fixed(reviewing, fake, cfg):
    fake.chunks = [content("C1: not_said\nC2: not_said")]
    fake.scripts = [[content(DRAFT)], [content("C1: not_said\nC2: not_said")], [content(REVIEW)]]
    r = chat(reviewing, [{"role": "user", "content": "Check the report."}])
    assert answer(r).endswith("| Yield | 98.6 % (implausibly high) |")
    review = fake.bodies[-1]["messages"]
    assert review[0]["content"] == cfg.verifier.consistency_prompt
    assert review[1]["content"].startswith("Task:\nCheck the report.")
    assert "'85 % (consistent)' → '98.6 % (implausibly high)' (the calculation above gives 98.6 %)" in reasoning_text(r)


@pytest.mark.parametrize("old, why", [("98.6", "found 2 times"), ("90 % yield", "not in the answer")])
def test_ambiguous_or_invented_quote_skipped(reviewing, fake, old, why):
    fake.chunks = [content("C1: not_said\nC2: not_said")]
    draft = DRAFT.replace("85 %", "98.6 %")
    fake.scripts = [[content(draft)], [content("C1: not_said")], [content(f"OLD: {old}\nNEW: x")]]
    r = chat(reviewing)
    assert answer(r) == draft and f"skipped {old!r} ({why})" in reasoning_text(r)


@pytest.mark.anyio
async def test_nothing_to_fix_and_report(cfg, fake, knowledge):
    cfg.verifier.consistency_check = True
    fake.chunks = [content("C1: not_said")]
    fake.scripts = [[content(DRAFT)], [content("C1: not_said")], [content("OK")]]
    items = [i async for i in Agent(cfg, fake.client(cfg.model_backend), knowledge).run([{"role": "user", "content": "q"}])]
    assert next(i["verification"] for i in items if "verification" in i)["consistency"] == []
    assert next(i["content"] for i in items if "content" in i) == DRAFT


def test_switched_off_makes_no_review_call(client, fake, cfg):
    fake.scripts = [[content("Benzene boils at 80.1 °C.")], [content("C1: supports S1")]]
    chat(client)
    assert len(fake.requests) == 2 and not cfg.verifier.consistency_check


def test_failed_review_leaves_the_answer(reviewing, fake):
    fake.scripts = [[content(DRAFT)], [content("C1: not_said")], CONNECT_ERROR]
    r = chat(reviewing)
    assert answer(r) == DRAFT and "Consistency review failed" in reasoning_text(r)


def test_review_skipped_when_too_long(cfg, fake, knowledge):
    cfg.verifier.consistency_check = True
    cfg.verifier.request["max_tokens"] = cfg.context_tokens - 50  # leaves no room for the answer under review
    fake.chunks = [content("C1: not_said")]
    fake.scripts = [[content(DRAFT)]]
    with TestClient(create_app(cfg, fake.client(cfg.model_backend), knowledge)) as c:
        r = chat(c)
    assert "Consistency review skipped" in reasoning_text(r) and answer(r) == DRAFT


# --- conservative review (c2): a fix must resolve a conflict inside the answer ----------------------------

REPORT = ("The payback is 5000 / 1200 = 4.17 years, and the gap is 96 °C − (−18 °C), about 120 °C.\n"
          "ΔT = 96 − (−18) ≈ 114 °C, so the statement is false.\n"
          "The mixture is heated gently.\n\n"
          "A1: 4.17\n"
          "A2: 120\n"
          "- **B:** True")


async def review(cfg, fake, knowledge, reply, text=REPORT, task="Task."):
    """The answer after the consistency review answered `reply`, and the review's reasoning lines."""
    fake.scripts = [[content(reply)]]
    lines, result = [], None
    async for item in Verifier(cfg, fake.client(cfg.model_backend), knowledge)._consistency(task, text):
        if isinstance(item, tuple):
            result = item
        else:
            lines.append(item["reasoning_content"])
    return result[0], "".join(lines)


@pytest.mark.anyio
@pytest.mark.parametrize("old, new, conflict", [
    ("A2: 120", "A2: 114", "ΔT = 96 − (−18) ≈ 114 °C"),  # the value the other statement states
    ("A2: 120", "A2: 114", "96 °C − (−18 °C)"),  # ... or computes, recomputed in code
    ("A1: 4.17", "A1: 4.2", "5000 / 1200 = 4.17"),  # numbers compared with their rounding
    ("True", "False", "so the statement is false"),  # the verdict word the answer uses (on "- **B:** True")
    ("about 120 °C", "about 114 °C", "ΔT = 96 − (−18) ≈ 114 °C"),
])
async def test_review_fix_with_a_quoted_conflict_applied(cfg, fake, knowledge, old, new, conflict):
    cfg.verifier.consistency_keep_answer_lines = False  # the quote rule, on answer lines too
    text, steps = await review(cfg, fake, knowledge, f"OLD: {old}\nNEW: {new}\nCONFLICT: {conflict}\nWHY: w")
    assert text == REPORT.replace(old, new) and f"{old!r} → {new!r} (w)" in steps


@pytest.mark.anyio
@pytest.mark.parametrize("old, new, conflict, why", [
    ("A2: 120", "A2: 114", None, "no conflicting passage quoted"),
    ("A1: 4.17", "A1: 3.65", "5000 / 1370 = 3.65", "conflicting passage not elsewhere in the answer"),  # its own sum
    ("about 120 °C", "about 114 °C", "about 120 °C", "conflicting passage not elsewhere in the answer"),
    ("A1: 4.17", "A1: 3.65", "The payback is 5000 / 1200 = 4.17 years", "3.65 is not stated in the quoted passage"),
    ("A2: 120", "A2: 110", "96 °C − (−18 °C)", "110 is not stated in the quoted passage"),
    ("True", "Not established", "so the statement is false",
     "answer line: not, established not in the quoted passage"),
    ("heated gently", "warmed slowly", "so the statement is false",
     "no value changed: warmed, slowly not in the quoted passage"),  # a rewording
    ("The mixture is heated gently.", "The mixture is heated.", "so the statement is false",
     "it adds nothing from the quoted passage"),
    ("A1: 4.17", "4.2", "5000 / 1200 = 4.17", "the answer line would lose its label"),
])
async def test_review_fix_without_a_conflict_skipped(cfg, fake, knowledge, old, new, conflict, why):
    cfg.verifier.consistency_keep_answer_lines = False  # the quote rule, on answer lines too
    reply = f"OLD: {old}\nNEW: {new}\n" + (f"CONFLICT: {conflict}\n" if conflict else "") + "WHY: clarity"
    text, steps = await review(cfg, fake, knowledge, reply)
    assert text == REPORT and f"skipped {old!r} ({why})" in steps


@pytest.mark.anyio
async def test_review_quotes_switched_off(cfg, fake, knowledge):
    cfg.verifier.consistency_quotes = cfg.verifier.consistency_keep_answer_lines = False
    text, _ = await review(cfg, fake, knowledge, "OLD: A1: 4.17\nNEW: A1: 3.65\nWHY: the sheet's formula")
    assert text == REPORT.replace("A1: 4.17", "A1: 3.65")


@pytest.mark.anyio
@pytest.mark.parametrize("old, new, conflict, named", [
    ("A2: 120", "A2: 114", "ΔT = 96 − (−18) ≈ 114 °C", "'ΔT = 96 − (−18) ≈ 114 °C'"),  # even with a good quote
    ("A2: 120", "A2: 114", None, "'w'"),  # without one, its reason is named
    ("True", "False", "so the statement is false", "'so the statement is false'"),  # on "- **B:** True"
    ("A1: 4.17", "4.2", "5000 / 1200 = 4.17", "'5000 / 1200 = 4.17'"),  # would drop the label
])
async def test_review_keeps_answer_lines(cfg, fake, knowledge, old, new, conflict, named):
    assert cfg.verifier.consistency_keep_answer_lines  # shipped default
    reply = f"OLD: {old}\nNEW: {new}\n" + (f"CONFLICT: {conflict}\n" if conflict else "") + "WHY: w"
    text, steps = await review(cfg, fake, knowledge, reply)
    assert text == REPORT and f"skipped {old!r} (answer line kept; conflict: {named})" in steps


@pytest.mark.anyio
async def test_review_with_answer_lines_kept_still_fixes_other_text(cfg, fake, knowledge):
    reply = "OLD: about 120 °C\nNEW: about 114 °C\nCONFLICT: ΔT = 96 − (−18) ≈ 114 °C\nWHY: w"
    text, _ = await review(cfg, fake, knowledge, reply)
    assert text == REPORT.replace("about 120 °C", "about 114 °C")


@pytest.mark.parametrize("new, why", [("A4: неверно", ""), ("A4: 486,5", ""), ("A4: 486,4", "486,4 is not stated")])
def test_unconfirmed_reads_decimal_commas(new, why):
    text = "Среднее 486,5 мг, значит утверждение неверно.\nA4: верно"
    result = unconfirmed(text, {"old": "A4: верно", "new": new, "conflict": "Среднее 486,5 мг, значит утверждение неверно"})
    assert result.startswith(why) and bool(result) == bool(why)


def test_annotated_no_claim_loses_its_citation(client, fake):
    fake.scripts = [SEARCH_ETHANOL, [content("Ethanol is a liquid [S1]. Let me know if you need more [S1].")],
                    [content("C1: supports S1\nC2: no_claim")]]
    text = answer(chat(client))
    assert text.startswith("Ethanol is a liquid [S1]. Let me know if you need more.")


# --- conservative claim checks (c01) ---------------------------------------------------------------

LOOP_DRAFT = ("P = 1000 × 9.81 × 0.04 × 6 = 2354 W [S1]. So 2354 W is more than the 2.2 kW of the motor [S1]. "
              "Ethanol is blue [S1].\nA1: 2354 W\nClaim B: Incorrect, the motor is too small\n")


def test_computations_and_answer_lines_are_not_judged(cfg, fake, knowledge):
    fake.scripts = [SEARCH_ETHANOL, [content(LOOP_DRAFT)], [content("C3: contradicts S1 | Ethanol is colorless.")]]
    with TestClient(create_app(cfg, fake.client(cfg.model_backend), knowledge)) as c:
        r = chat(c, [{"role": "user", "content": PUMP_TASK}])
    user = fake.bodies[2]["messages"][1]["content"]
    assert user.endswith("Claims:\nC3: Ethanol is blue.")  # one claim judged: no computation, no answer line
    assert answer(r) == ("P = 1000 × 9.81 × 0.04 × 6 = 2354 W. So 2354 W is more than the 2.2 kW of the motor. "
                         "Ethanol is colorless.\nA1: 2354 W\nClaim B: Incorrect, the motor is too small")
    assert "C1 unverified (computation: checked in code, not against sources)" in reasoning_text(r)


def test_derived_claims_judged_when_switched_off(cfg, fake, knowledge):
    cfg.verifier.skip_derived_claims = False
    fake.chunks = [content("C1: not_said")]
    fake.scripts = [SEARCH_ETHANOL, [content(LOOP_DRAFT)]]
    with TestClient(create_app(cfg, fake.client(cfg.model_backend), knowledge)) as c:
        chat(c, [{"role": "user", "content": PUMP_TASK}])
    user = fake.bodies[2]["messages"][1]["content"]
    assert "C1: P = 1000 × 9.81" in user and "A1: 2354 W" in user


@pytest.mark.parametrize("keep, expected", [(True, "Ethanol is a liquid. It is blue."), (False, "Ethanol is a liquid.")])
def test_contradiction_without_a_correction(cfg, fake, knowledge, keep, expected):
    cfg.verifier.keep_uncorrected = keep
    fake.scripts = [SEARCH_ETHANOL, [content("Ethanol is a liquid [S1]. It is blue [S1].")],
                    [content("C1: supports S1\nC2: contradicts S1")]]
    with TestClient(create_app(cfg, fake.client(cfg.model_backend), knowledge)) as c:
        r = chat(c)
    assert answer(r) == expected
    assert ("no correction: kept" in reasoning_text(r)) is keep


def test_echoes(cfg):
    prompt = cfg.verifier.judge_prompt  # its format example: "C2: contradicts S1 | the claim rewritten so that ..."
    assert echoes("the claim rewritten so that it agrees with S1", prompt)
    assert echoes("The claim, rewritten so that it agrees with S3.", prompt)
    assert echoes("Formula: the claim rewritten so that it agrees with S2", prompt)
    assert echoes("not_said", prompt)
    assert not echoes("Ethanol is colorless.", prompt)
    assert not echoes("Ethanol boils at 78.23 °C, so the claim is wrong.", prompt)
    assert not echoes("", prompt)


ECHO = "C1: supports S1\nC2: contradicts S1 | the claim rewritten so that it agrees with S1"


@pytest.mark.parametrize("ignore, keep, expected", [
    (True, True, "Ethanol is a liquid. It is blue."),
    (True, False, "Ethanol is a liquid."),
    (False, True, "Ethanol is a liquid. the claim rewritten so that it agrees with S1"),
])
def test_judge_echo_not_applied(cfg, fake, knowledge, ignore, keep, expected):
    cfg.verifier.ignore_prompt_echo, cfg.verifier.keep_uncorrected = ignore, keep
    cfg.verifier.judge_minimal_edits = False  # this switch alone (minimal edits would also refuse the echo)
    fake.scripts = [SEARCH_ETHANOL, [content("Ethanol is a liquid [S1]. It is blue [S1].")], [content(ECHO)]]
    with TestClient(create_app(cfg, fake.client(cfg.model_backend), knowledge)) as c:
        r = chat(c)
    assert answer(r) == expected
    assert ("correction repeats the judge instructions: ignored" in reasoning_text(r)) is ignore


BOILS = "Ethanol\nIts boiling point is 78.37 °C."


@pytest.mark.parametrize("claim, passage, correction, fixed", [
    ("Ethanol boils at 79 °C [S1].", BOILS, "Ethanol boils at 78.37 °C", "Ethanol boils at 78.37 °C [S1]."),
    ("Ethanol boils at 79 °C [S1].", BOILS, "Ethanol boils at 78.4 °C.", "Ethanol boils at 78.4 °C [S1]."),  # rounded
    ("Ethanol boils at **79 °C** [S1].", BOILS, "Ethanol boils at 78.37 °C", "Ethanol boils at **78.37 °C** [S1]."),
    ("It is blue [S1].", "Ethanol\nIt is a colorless liquid.", "Ethanol is colorless.", "Ethanol is colorless [S1]."),
    ("Этанол кипит при 79 °C [S1].", "Этанол\nТемпература кипения 78,37 °C.", "Этанол кипит при 78,37 °C.",
     "Этанол кипит при 78,37 °C [S1]."),
])
def test_minimal_edit_applied(claim, passage, correction, fixed):
    assert minimal_edit(claim, correction, passage) == (fixed, "")  # only the edited terms change


@pytest.mark.parametrize("claim, correction, why", [
    ("Ethanol boils at 79 °C [S1].", "Ethanol has a boiling point of 78.37 °C.",
     "not a minimal edit: rewrites 'boils at 79' as 'has a boiling point of 78.37'"),
    ("Ethanol boils at 79 °C [S1].", "Ethanol boils at 78.37 °C, not 79 °C as the answer says.",
     "not a minimal edit: adds"),
    ("Ethanol boils at 79 °C [S1].", "Ethanol boils at 80.1 °C.", "not stated in the source: 80.1"),
    ("Ethanol boils at 79 °C [S1].", "Ethanol boils at 78.0 °C.", "not stated in the source: 78.0"),  # finer than 78.37
    ("Ethanol boils at 79 °C [S1].", "**Ethanol boils at 79 °C.**", "not a minimal edit: it changes no value or term"),
    ("The equation is balanced [S1].", "The equation is not balanced.", "not a minimal edit: adds 'not'"),
    ("The equation is not balanced [S1].", "The equation is balanced.", "not a minimal edit: drops 'not'"),
    ("The equation is balanced [S1].", "The equation is unbalanced.",
     "not a minimal edit: changes 'balanced' to 'unbalanced'"),
    ("Statement B is true [S1].", "Statement B is false.", "not stated in the source: false"),
])
def test_minimal_edit_refused(claim, correction, why):
    fixed, said = minimal_edit(claim, correction, f"{BOILS} An unbalanced equation is not balanced.")
    assert fixed is None and said.startswith(why)


REWRITE = "C1: supports S1\nC2: contradicts S1 | It is not blue, ethanol is a colorless liquid."


@pytest.mark.parametrize("on, expected", [
    (True, "Ethanol is a liquid. It is blue."),
    (False, "Ethanol is a liquid. It is not blue, ethanol is a colorless liquid."),
])
def test_judge_correction_not_a_minimal_edit(cfg, fake, knowledge, on, expected):
    cfg.verifier.judge_minimal_edits = on
    fake.scripts = [SEARCH_ETHANOL, [content("Ethanol is a liquid [S1]. It is blue [S1].")], [content(REWRITE)]]
    with TestClient(create_app(cfg, fake.client(cfg.model_backend), knowledge)) as c:
        r = chat(c)
    assert answer(r) == expected
    steps = reasoning_text(r)
    assert ("C2 judge: contradicts S1, correction not applied (not a minimal edit: adds 'not'): It is not blue, "
            "ethanol is a colorless liquid." in steps) is on
    assert ("C2 corrected from S1" in steps) is not on


# --- arithmetic ---------------------------------------------------------------------------------------


def test_wrong_product_fixed_in_place():
    draft = "The mass of CaO is 0.0500 × 56.08 = 3.80 g, so 5.00 − 2.80 = 2.20 g of CO2 leaves."
    fixed, fixes, checked = fix_arithmetic(draft, 0.005)
    assert fixed == "The mass of CaO is 0.0500 × 56.08 = 2.80 g, so 5.00 − 2.80 = 2.20 g of CO2 leaves."
    assert fixes == [{"expression": "0.0500 * 56.08", "stated": "3.80", "fixed": "2.80"}] and checked == 2


@pytest.mark.parametrize("draft, expected", [
    ("Margin:\n- (800 - 2) / 800 × 100 = 99.75%", None),
    ("Margin:\n  - (800 - 2) / 800 × 100 = 90.00%", "Margin:\n  - (800 - 2) / 800 × 100 = 99.75%"),
    ("> * (6 - 1) × 2 = 10 kg", None),
    ("-5 × 2 = -10", None),
    ("-5 × 2 = 10", "-5 × 2 = -10"),
])
def test_list_marker_results_not_negated(draft, expected):
    fixed, fixes, checked = fix_arithmetic(draft, 0.005)
    assert checked == 1 and fixed == (expected or draft) and len(fixes) == (expected is not None)


@pytest.mark.parametrize("draft", [
    "n = 2.00 g / 138.12 g/mol = 0.01448 mol",
    r"$$\frac{2.10}{2.609} \times 100\% = 80.5\%$$",
    "1 atm = 101.325 kPa",
    "E = mc^2 = 9.0 × 10^16 J",
    "The yield is 2.10 / 2.609 × 100 % ≈ 80 %.",
])
def test_right_or_not_arithmetic_left_alone(draft):
    fixed, fixes, _ = fix_arithmetic(draft, 0.005)
    assert fixed == draft and fixes == []


# Regression (user's vanillin test on #28): correct percent results were rewritten as "0.8%" and "0.0%".
@pytest.mark.parametrize("draft", [
    "The yield is 0.0986/0.1236 = 79.8% ≈ 80%.",
    r"$$\frac{80.0\%}{87.4\%} = 91.5\%$$",
    "x = 0.0986/0.1236 × 100% = 79.8%",
    r"\frac{0.0986}{0.1236} \times 100\% = 79.8\%",
    "x = 0.0986/0.1236 × 100% = 79.8",
    "2.10/2.609 × 100 = 80.5%",
    "0.0986/0.1236 = 0.798",
    "80% / 90% = 88.9%",
    "0.25 × 0.80 = 20%",
    "15% × 200 = 30",
    "purity = 96% × 0.90 = 86.4%",
])
def test_percent_results_read_every_way(draft):
    fixed, fixes, checked = fix_arithmetic(draft, 0.005)
    assert fixed == draft and fixes == [] and checked == 1


@pytest.mark.parametrize("draft, expected", [
    ("0.0986/0.1236 = 50%", "0.0986/0.1236 = 80%"),
    ("0.0986/0.1236 = 50.0%", "0.0986/0.1236 = 79.8%"),
    ("0.0986/0.1236 × 100% = 50.0%", "0.0986/0.1236 × 100% = 79.8%"),
    ("2.10/2.609 × 100 = 50.0%", "2.10/2.609 × 100 = 80.5%"),
    ("80% / 90% = 50.0%", "80% / 90% = 88.9%"),
    ("0.0986/0.1236 = 0.5", "0.0986/0.1236 = 0.8"),
])
def test_percent_results_wrong_every_way_are_fixed(draft, expected):
    assert fix_arithmetic(draft, 0.005)[0] == expected


@pytest.mark.parametrize("draft", [
    "15% × 200 = 40",  # 30 or 3000: the intended reading is not clear
    "3 / 1 = 50%",  # 300 % or 3 %: not clear
])
def test_ambiguous_wrong_results_left_alone(draft):
    fixed, fixes, _ = fix_arithmetic(draft, 0.005)
    assert fixed == draft and fixes == []


def test_scientific_and_russian_results():
    fixed, fixes, _ = fix_arithmetic("N = 0.0555 × 6.022 × 10^23 = 3.00 × 10^22 молекул", 0.005)
    assert fixes[0]["fixed"] == "3.342 × 10^22" and "= 3.342 × 10^22 молекул" in fixed
    fixed, fixes, _ = fix_arithmetic("m = 0,0500 моль × 56,08 г/моль = 3,80 г", 0.005)
    assert fixes[0]["stated"] == "3,80" and fixed.endswith("= 2.80 г")


# c01: a result in another SI prefix of its chain is not wrong; grouped numbers are one number.
@pytest.mark.parametrize("draft", [
    "P = 1000 × 9.81 × 0.04 × 6 W = 2.35 kW",
    "P = 1000 × 9.81 × 0.04 × 6 = 2.35 kW",
    "P = 2.5 × 980 = 2.45 = 2.45 kW",  # the prefix after the next "="
    "E = 15 000 mA·h × 3.7 V = 55.5 W·h",
    "m = 250 mg × 6 = 1.5 g",
    "t = 0.004 s × 500 = 2000 ms",
    "E = 18 500 мА·ч × 3,7 В = 68,45 Вт·ч",
])
def test_prefix_readings_left_alone(draft):
    fixed, fixes, checked = fix_arithmetic(draft, 0.005)
    assert fixed == draft and fixes == [] and checked == 1


@pytest.mark.parametrize("draft, expected", [
    ("P = 1000 × 9.81 × 0.04 × 6 W = 3.35 kW", "P = 1000 × 9.81 × 0.04 × 6 W = 2.35 kW"),
    ("m = 250 mg × 6 = 2.5 g", "m = 250 mg × 6 = 1.5 g"),
    ("E = 15 000 mA·h × 3.7 V = 60.0 W·h", "E = 15 000 mA·h × 3.7 V = 55.5 W·h"),
    ("t = 0.004 s × 500 = 3000 ms", "t = 0.004 s × 500 = 2000 ms"),
    ("Q = 12 500 × 4 = 40 000", "Q = 12 500 × 4 = 50000"),
])
def test_wrong_under_every_reading_still_fixed(draft, expected):
    assert fix_arithmetic(draft, 0.005)[0] == expected


def test_prefix_readings_switched_off():
    assert fix_arithmetic("m = 250 mg × 6 = 1.5 g", 0.005, prefixes=False)[0] == "m = 250 mg × 6 = 1500.0 g"


@pytest.mark.parametrize("draft", [
    "Q = 12 500 × 4 = 50 000",
    "Q = 12 500 × 4 = 50 000",
    r"$Q = 12\,500 \times 4 = 50\,000$",
    "x = 1,234.5 × 2 = 2,469",
    "x = 1 234,5 × 2 = 2 469 г",
    "h = 6.626 070 15 × 2 = 13.25",  # cannot be read as one expression: skipped
    "E = 24 000 мА·ч × 3,6 В = 86,4 Вт·ч",
])
def test_grouped_numbers_left_alone(draft):
    fixed, fixes, _ = fix_arithmetic(draft, 0.005)
    assert fixed == draft and fixes == []


def test_plain_answer_shows_the_fixed_result(cfg, fake, knowledge):
    fake.chunks = [content("C1: not_said")]
    fake.scripts = [[content("The mass of CaO is 0.0500 × 56.08 = 3.80 g.")]]
    with TestClient(create_app(cfg, fake.client(cfg.model_backend), knowledge)) as c:
        r = chat(c)
    assert answer(r) == "The mass of CaO is 0.0500 × 56.08 = 2.80 g."
    assert "Arithmetic: 1 calculation recomputed, 1 fixed" in reasoning_text(r)
    assert "0.0500 * 56.08 = 3.80 → 2.80" in reasoning_text(r)


def test_arithmetic_check_switched_off(cfg, fake, knowledge):
    cfg.verifier.arithmetic_check = False
    fake.chunks = [content("C1: not_said")]
    fake.scripts = [[content("It is 2 × 3 = 7.")]]
    with TestClient(create_app(cfg, fake.client(cfg.model_backend), knowledge)) as c:
        r = chat(c)
    assert answer(r) == "It is 2 × 3 = 7." and "Arithmetic:" not in reasoning_text(r)


@pytest.mark.anyio
async def test_report_lists_arithmetic_fixes(cfg, fake, knowledge):
    fake.chunks = [content("C1: not_said")]
    fake.scripts = [[content("It is 2 × 3 = 7.")]]
    items = [i async for i in Agent(cfg, fake.client(cfg.model_backend), knowledge).run([{"role": "user", "content": "q"}])]
    report = next(i["verification"] for i in items if "verification" in i)
    assert report["arithmetic"] == [{"expression": "2 * 3", "stated": "7", "fixed": "6"}]


# --- domain checkers ----------------------------------------------------------------------------------


def test_checker_fixes_the_answer_silently(cfg, fake, knowledge):
    fake.chunks = [content("C1: not_said")]
    fake.scripts = [[content("Water forms as H2 + O2 → H2O in a flame.")]]
    with TestClient(create_app(cfg, fake.client(cfg.model_backend), knowledge)) as c:
        r = chat(c)
    assert answer(r) == "Water forms as 2H2 + O2 → 2H2O in a flame."
    assert "Check element_balance: 1 finding" in reasoning_text(r) and "(rebalanced)" in reasoning_text(r)


@pytest.mark.anyio
async def test_checker_findings_in_the_report_and_off_switch(cfg, fake, knowledge):
    fake.chunks = [content("C1: not_said")]
    fake.scripts = [[content("It is NaCl → Na2O here.")]]
    items = [i async for i in Agent(cfg, fake.client(cfg.model_backend), knowledge).run([{"role": "user", "content": "q"}])]
    (finding,) = next(i["verification"] for i in items if "verification" in i)["checks"]
    assert finding["checker"] == "element_balance" and finding["fixed"] is None
    cfg.verifier.checkers = []
    fake.scripts = [[content("Water forms as H2 + O2 → H2O.")]]
    items = [i async for i in Agent(cfg, fake.client(cfg.model_backend), knowledge).run([{"role": "user", "content": "q"}])]
    assert next(i["content"] for i in items if "content" in i) == "Water forms as H2 + O2 → H2O."


@pytest.mark.parametrize("clear_only, out", [(True, "Then S1 → S2."), (False, "Then 2S1 → S2.")])
def test_doubtful_fix_reported(clear_only, out):
    got, (finding,) = run_checkers("Then S1 → S2.", ["element_balance"], "", clear_only=clear_only)
    assert got == out
    assert finding["note"] == ("left as written: S1 has an explicit count of 1 (an index or a label); "
                               "rebalanced: 2S1 → S2" if clear_only else "rebalanced")


@pytest.mark.anyio
@pytest.mark.parametrize("keep", [True, False])
async def test_task_labels_reach_the_checker(cfg, fake, knowledge, keep):
    cfg.verifier.keep_quotes = keep
    fake.chunks = [content("C1: not_said")]
    fake.scripts = [[content("So S5 → S6.")]]
    task = "Order the steps.\nS4. Mix.\nS5. Heat.\nS6. Filter."
    items = [i async for i in Agent(cfg, fake.client(cfg.model_backend), knowledge).run([{"role": "user", "content": task}])]
    assert next(i["content"] for i in items if "content" in i) == "So S5 → S6."


# --- quotations of the task (c3) ------------------------------------------------------------------------

EQ_TASK = "The lab report gives Fe₂O₃ + CO → Fe + CO₂ for the furnace step. Audit it."
EQ_DRAFT = "The report's equation Fe2O3 + CO → Fe + CO2 is not balanced: one Fe on the right, two on the left."
AUDIT_TASK = "Audit the company profile.\nS3. Harbour Instruments Ltd was founded in 1994.\nS4. It has 120 staff."


def test_quoted_runs():
    task = "The report gives “Fe₂O₃ + CO → Fe + CO₂” for sodium benzoate (C₆H₅COONa) and H2 + O2 → H2O."
    text = "Its **Fe2O3 + CO → Fe + CO2** is wrong; sodium benzoate is C6H5COONa; H2 + O2 → H2O too."
    mask = quoted(text, task)

    def marks(part):
        at = text.index(part)
        return mask[at : at + len(part)]

    assert all(marks("Fe2O3 + CO → Fe + CO2"))  # subscripts, emphasis and quote marks ignored
    assert not any(marks("It")) and not any(marks("is wrong"))  # ("s " is shared too: "gives Fe…")
    for short in ("sodium benzoate", "C6H5COONa", "H2 + O2 → H2O"):  # a name, a formula, a short equation
        assert not any(marks(short))
    assert not any(quoted(text, ""))


@pytest.mark.parametrize("task, draft, fixed", [
    (EQ_TASK, EQ_DRAFT, None),  # quoted from the task: reported, left as written
    ("Audit the furnace step of the lab report.", EQ_DRAFT, "Fe2O3 + 3CO → 2Fe + 3CO2"),  # the answer's own
    ("", EQ_DRAFT, "Fe2O3 + 3CO → 2Fe + 3CO2"),  # keep_quotes off
    ("Explain H2 + O2 → H2O.", "In a flame, H2 + O2 → H2O releases heat.", "2H2 + O2 → 2H2O"),  # too short
])
def test_quoted_equation_kept(task, draft, fixed):
    out, (finding,) = run_checkers(draft, ["element_balance"], task)
    assert finding["fixed"] == fixed
    if fixed:
        assert out == draft.replace(finding["text"], fixed) and finding["note"] == "rebalanced"
    else:
        assert out == draft
        assert finding["note"] == "quoted from the task, left as written; rebalanced: Fe2O3 + 3CO → 2Fe + 3CO2"


@pytest.mark.parametrize("keep", [True, False])
def test_quoted_equation_kept_in_the_answer(cfg, fake, knowledge, keep):
    cfg.verifier.keep_quotes = keep
    fake.chunks = [content("C1: not_said")]
    fake.scripts = [[content(EQ_DRAFT)]]
    with TestClient(create_app(cfg, fake.client(cfg.model_backend), knowledge)) as c:
        r = chat(c, [{"role": "user", "content": EQ_TASK}])
    assert answer(r) == (EQ_DRAFT if keep else EQ_DRAFT.replace("Fe2O3 + CO → Fe + CO2", "Fe2O3 + 3CO → 2Fe + 3CO2"))
    assert ("quoted from the task, left as written" in reasoning_text(r)) is keep


@pytest.mark.parametrize("keep", [True, False])
def test_restated_claim_not_judged(cfg, fake, knowledge, keep):
    cfg.verifier.keep_quotes = keep
    cfg.verifier.judge_minimal_edits = False  # this switch alone (S1 does not state 1987: minimal edits would refuse)
    draft = "**S3. Harbour Instruments Ltd was founded in 1994.** ✗ WRONG [S1]\nEthanol is blue [S1].\n"
    verdicts = "C1: contradicts S1 | Harbour Instruments Ltd was founded in 1987.\nC2: contradicts S1 | Ethanol is colorless."
    fake.scripts = [SEARCH_ETHANOL, [content(draft)], [content(verdicts)]]
    with TestClient(create_app(cfg, fake.client(cfg.model_backend), knowledge)) as c:
        r = chat(c, [{"role": "user", "content": AUDIT_TASK}])
    user = fake.bodies[2]["messages"][1]["content"]
    if keep:  # the label and the verdict mark stay; the answer's own claim is still judged
        assert user.endswith("Claims:\nC2: Ethanol is blue.")
        assert answer(r) == "**S3. Harbour Instruments Ltd was founded in 1994.** ✗ WRONG\nEthanol is colorless."
        assert "C1 unverified (quoted from the task: not checked)" in reasoning_text(r)
    else:
        assert "C1: Harbour Instruments Ltd was founded in 1994. ✗ WRONG" in user
        assert answer(r) == "**S3. Harbour Instruments Ltd was founded in 1987.\nEthanol is colorless."


QUOTING = ("**S3. Harbour Instruments Ltd was founded in 1994.** ✗ WRONG: the register gives 1987.\n"
           "The report is 1994 − 1987 = 7 years off, so the gap is about 9 years.")


@pytest.mark.anyio
@pytest.mark.parametrize("keep", [True, False])
async def test_review_fix_on_a_quote_skipped(cfg, fake, knowledge, keep):
    cfg.verifier.keep_quotes = keep
    reply = ("OLD: founded in 1994\nNEW: founded in 1987\nCONFLICT: the register gives 1987\nWHY: w\n"
             "OLD: about 9 years\nNEW: about 7 years\nCONFLICT: 1994 − 1987 = 7\nWHY: w")
    text, steps = await review(cfg, fake, knowledge, reply, QUOTING, AUDIT_TASK)
    own = QUOTING.replace("about 9", "about 7")  # the answer's own text: fixed either way
    assert text == (own if keep else own.replace("founded in 1994", "founded in 1987"))
    assert ("skipped 'founded in 1994' (quoted from the task)" in steps) is keep


COST_TASK = "The report computes the batch cost as 12 × 45 = 560 euros. Check it."
COST_DRAFT = "The report's 12 × 45 = 560 euros is wrong."


@pytest.mark.parametrize("task, kept", [
    (COST_TASK, True),  # quoted from the task: left as written, listed with a note
    ("Check the batch cost of 12 crates at 45 euros.", False),  # the answer's own product: fixed
    ("Is 12 × 45 = 560?", False),  # shared, but shorter than a quotation
    ("", False),  # keep_quotes off
])
def test_quoted_wrong_product(task, kept):
    out, (fix,), checked = fix_arithmetic(COST_DRAFT, 0.005, task=task)
    assert checked == 1 and out == (COST_DRAFT if kept else COST_DRAFT.replace("560", "540"))
    assert fix == {"expression": "12 * 45", "stated": "560", "fixed": "540"} | (
        {"note": "quoted from the task, left as written"} if kept else {})


@pytest.mark.anyio
async def test_quoted_wrong_product_only_in_the_trace(cfg, fake, knowledge):
    fake.chunks = [content("C1: not_said")]
    fake.scripts = [[content(COST_DRAFT)]]
    question = [{"role": "user", "content": COST_TASK}]
    items = [i async for i in Agent(cfg, fake.client(cfg.model_backend), knowledge).run(question)]
    assert next(i["content"] for i in items if "content" in i) == COST_DRAFT
    assert next(i["verification"] for i in items if "verification" in i)["arithmetic"] == []
    steps = "".join(i.get("reasoning_content", "") for i in items)
    assert "1 calculation recomputed, 0 fixed\n- 12 * 45 = 560: quoted from the task, left as written (would be 540)" in steps


# --- reference values over literature ----------------------------------------------------------------


def article(title, infobox, lead="It is a fixture lead."):
    return ArticleRecord(title, title, f"{title} {lead}", infobox, "", 1.0, ["chemistry"],
                         Citation("wiki", "w.zim", "1", "CC BY-SA", {"path": title}, f"https://wiki/{title}"))


def literature(title, text):
    return PassageRecord(title, "Section", title, text, 1.0, ["chemistry"], "Wikipedia Chemistry",
                         Citation("lit", "l.zim", "1", "CC BY-SA", {"path": title, "passage": 1}, f"https://lit/{title}"))


WATER = article("Water", "Critical point: 647.096 K, 22.064 MPa, 322 kg/m³\nDensity: 997 kg/m³")
PHASE = literature("Phase diagram", "The critical point of water is at 647 K and 22.064 MPa, density 356 kg/m³.")
GADOLINIUM = article("Gadolinium", "Melting point: 1585 K\nCurie point: 293.4 K")
FERRO = literature("Ferromagnetism", "Gadolinium has a Curie temperature of 289 K.")


def retrieved(*records):
    s = Sources()
    for r in records:
        s.add("text", r, r.passage("", 1500))
    return s


def supported_by(text, *ids):
    c = claim(text)
    c.outcome, c.ids = SUPPORTED, list(ids)
    return c


def test_sources_know_literature():
    s = retrieved(WATER, PHASE)
    assert [x.literature for x in s.items] == [False, True]


@pytest.mark.parametrize("records, text, expected", [
    ((WATER, PHASE), "The critical density of water is 356 kg/m³ [S2].", "356 kg/m3, but reference S1 gives 322 kg/m3"),
    ((GADOLINIUM, FERRO), "Gadolinium has a Curie temperature of 289 K [S2].", "289 K, but reference S1 gives 293.4 K"),
    ((FERRO, GADOLINIUM), "Gadolinium has a Curie temperature of 289 K [S1].", "reference S2 gives 293.4 K"),  # uncited
])
def test_literature_value_against_a_reference(records, text, expected):
    lit = next(x.id for x in retrieved(*records).items if x.literature)
    assert expected in reference_conflict(supported_by(text, lit), retrieved(*records), 0.001)


@pytest.mark.parametrize("records, text, ids", [
    ((WATER, PHASE), "The critical density of water is 322 kg/m³ [S2].", ["S2"]),  # the reference agrees
    ((WATER, PHASE), "The critical density of water is 0.356 g/cm³ [S2].", ["S2"]),  # another unit: not compared
    ((article("Water", "Density: 997 kg/m³"), PHASE), "The critical density of water is 356 kg/m³ [S2].", ["S2"]),
    ((WATER, PHASE), "The critical density of carbon dioxide is 468 kg/m³ [S2].", ["S2"]),  # other words
    ((GADOLINIUM, FERRO), "Gadolinium is ferromagnetic below 289 K [S2].", ["S2"]),  # words name no reference line
    ((WATER, PHASE), "The critical density of water is 356 kg/m³ [S1][S2].", ["S1", "S2"]),  # reference support
    ((WATER, article("Steam", "Density: 0.6 kg/m³")), "The critical density of water is 356 kg/m³ [S2].", ["S2"]),  # base
])
def test_no_conflict(records, text, ids):
    assert reference_conflict(supported_by(text, *ids), retrieved(*records), 0.001) == ""


async def verify(cfg, fake, knowledge, draft, sources, judge):
    fake.scripts = [[content(judge)]]
    items = [i async for i in Verifier(cfg, fake.client(cfg.model_backend), knowledge).verify("q", draft, sources)]
    return items[-1], "".join(i.get("reasoning_content", "") for i in items[:-1])


@pytest.mark.anyio
@pytest.mark.parametrize("on", [True, False])
async def test_literature_only_support_unverified(cfg, fake, knowledge, on):
    cfg.verifier.reference_over_literature = on
    draft = "The critical density of water is 356 kg/m³ [S2]."
    result, steps = await verify(cfg, fake, knowledge, draft, retrieved(WATER, PHASE), "C1: supports S2")
    c = result.claims[0]
    assert result.answer == "The critical density of water is 356 kg/m³."  # plain answer: nothing rewritten
    if on:
        assert (c.outcome, c.ids) == (UNVERIFIED, [])
        assert "C1 unverified (only literature supports 356 kg/m3, but reference S1 gives 322 kg/m3)" in steps
    else:
        assert (c.outcome, c.ids) == (SUPPORTED, ["S2"]) and "only literature" not in steps
