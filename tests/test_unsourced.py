"""Numbers the answer states without any source (claim-verifier spec, Unsourced numbers flagged): pure function."""

import pytest

from haisenberg.unsourced import trace_line, unsourced


def flagged(answer, conversation=(), tool_results=()):
    return [f["text"] for f in unsourced(answer, list(conversation), list(tool_results))]


def test_invented_specifics_flagged():  # the c10a baseline shapes (kb R007)
    answer = ("**Yield:** Typically 60–70 % for student laboratory preparations.\n"
              "| TLC | Single spot at Rf ~0.5-0.7 (ethanol:water 1:1) |")
    found = unsourced(answer, ["Plan the synthesis of aspirin from salicylic acid."], [])
    assert [(f["text"], f["value"], f["unit"]) for f in found] == [("60–70 %", [60.0, 70.0], "%"),
                                                                   ("0.5-0.7", [0.5, 0.7], "")]
    assert found[0]["context"].startswith("**Yield:** Typically 60–70 %") and "\n" not in found[1]["context"]
    assert trace_line(found) == "> unsourced numbers: 2 (60–70 %, 0.5-0.7)" and trace_line([]) == "> unsourced numbers: 0"


def test_values_from_a_tool_result_or_the_request_not_flagged():
    passage = "[S1] Caffeine (compound)\nmolar mass: 194.194 g/mol; melting point 235 °C"
    assert flagged("Caffeine has a molar mass of 194.19 g/mol and melts at 235 °C.", tool_results=[passage]) == []
    assert flagged("Caffeine melts at 238 °C.", tool_results=[passage]) == ["238 °C"]
    request = "Copper(II) oxide charged: 7.95 g (M = 79.545 g/mol)"
    assert flagged("About 8 g, exactly 7.95 g of CuO (79.545 g/mol) was used.", [request]) == []  # rounding of either
    assert flagged("The calculation used 23.40 / 23.75 = 0.9853.", tool_results=["23.4 / 23.75 = 0.9852631579"]) == []


def test_the_answers_own_computation_not_flagged():
    answer = "Yield = 23.4 / 23.75 = 98.5 %, so a 98.5 % yield is implausibly high; (1100 − 800) / 1100 ≈ 27.3 %."
    assert flagged(answer) == []
    assert flagged("ΔT = 96 °C − (−18 °C), so the gap is 114 °C.", ["96 °C and −18 °C"]) == []  # a computed value
    latex = "$$M = 2.016 \\, \\text{g/mol} + 31.998 \\, \\text{g/mol}$$\n$$M = 34.014 \\, \\text{g/mol}$$"
    assert flagged(latex) == ["2.016", "31.998"]  # the sum is computed; its operands have no `= c` and no source


def test_restatement_counts_only_when_the_conversion_is_written():
    assert flagged("The yield is 98.5 %.", ["yield fraction 0.985"]) == ["98.5 %"]
    assert flagged("0.985 = 98.5 %, so the yield is 98.5 %.", ["yield fraction 0.985"]) == []
    assert flagged("Энергия 118,3 Вт·ч = 0,1183 кВт·ч.", ["Энергия 118,3 Вт·ч"]) == []
    assert flagged("Энергия 0,1183 кВт·ч.", ["Энергия 118,3 Вт·ч"]) == ["0,1183"]


def test_ranges():
    assert flagged("Use 60–70 % ethanol.", ["60 % to 70 %"]) == []
    assert flagged("Use 60–75 % ethanol.", ["60 % ethanol"]) == ["60–75 %"]  # one end unsourced: the range counts
    assert flagged("Wait 10 to 15 minutes; heat to 85-90 °C.") == ["10 to 15 minutes", "85-90 °C"]
    assert flagged("The tolerance is 220 × (1 - 0.20).", ["220 µF ± 20 %"]) == ["0.20"]  # a minus, not a range


def test_russian_decimal_comma():
    assert flagged("Масса 20,4 г, выход 87,5 %.", ["Получено 20,40 г продукта."]) == ["87,5 %"]
    found = unsourced("Выход 87,5 %.", [], [])
    assert found[0]["value"] == 87.5 and found[0]["unit"] == "%"


def test_ids_years_dates_small_integers_and_list_numbers_not_flagged():
    answer = ("### 3.2 Claims\n"
              "11. C3 is wrong, У2 too, and S12 was never checked.\n"
              "In 1957 to 1947 terms: the plant opened in 1893, on 14 March 2023 and closed on 2023-03-14.\n"
              "There are 3 claims, 7 steps and 2 errors.\n")
    assert flagged(answer, ["C3. ... У2. ... S12. ..."]) == []
    assert flagged("There are 3 g and 12 claims.") == ["3 g", "12"]  # with a unit, or above 10: counted


def test_numbers_inside_quoted_task_text_not_flagged():
    request = "The supplier charged 1,450 kg of ore to the furnace."  # 1450 in English
    answer = "Отчёт: «The supplier charged 1,450 kg of ore to the furnace» — масса не проверена."  # 1,450 in Russian
    assert flagged(answer, [request]) == []
    assert flagged("Масса 1,450 кг не проверена.", [request]) == ["1,450"]


@pytest.mark.parametrize("answer", ["The batch is 450 kg.", "Партия 450 кг."])
def test_earlier_turns_are_sources(answer):
    assert flagged(answer, ["How heavy is the batch?", "The batch is 450 kg.", "Is that right?"]) == []
