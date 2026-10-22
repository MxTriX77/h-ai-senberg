"""Claim verifier (claim-verifier spec): split, numbers and formulas in code, judge, re-search, labels."""

import pytest
from fastapi.testclient import TestClient

from conftest import CONNECT_ERROR, content, finish, tool_call
from haisenberg.agent import Agent
from haisenberg.l1.store import L1Store
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
    parse_fixes,
    parse_verdicts,
    read_formulas,
    read_numbers,
    render,
    Verifier,
    split_claims,
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


# --- numbers and formulas --------------------------------------------------------------------------


@pytest.mark.parametrize("text, value", [
    ("c = 299 792 458 m/s", 299792458),
    ("about 10,000 K", 10000),
    ("h = 6.626 070 15 e-34 J Hz^-1", 6.62607015e-34),
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
REVIEW = "OLD: 85 % (consistent)\nNEW: 98.6 % (implausibly high)\nWHY: the calculation above gives 98.6 %"


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


def test_annotated_no_claim_loses_its_citation(client, fake):
    fake.scripts = [SEARCH_ETHANOL, [content("Ethanol is a liquid [S1]. Let me know if you need more [S1].")],
                    [content("C1: supports S1\nC2: no_claim")]]
    text = answer(chat(client))
    assert text.startswith("Ethanol is a liquid [S1]. Let me know if you need more.")
