from __future__ import annotations

import pytest
from skyrl_bird_sql_v1.scoring import parse_grading_method, results_match
from skyrl_bird_sql_v1.sqlite import result_from_rows


def test_set_ignores_order_duplicates_and_column_order() -> None:
    gold = result_from_rows((("Ada", "London"), ("Grace", "New York")))
    predicted = result_from_rows((("New York", "Grace"), ("London", "Ada"), ("Ada", "London")))
    assert results_match(predicted, gold, "set")


def test_multiset_keeps_duplicate_counts_but_not_order() -> None:
    gold = result_from_rows(((1,), (1,), (2,)))
    assert results_match(result_from_rows(((2,), (1,), (1,))), gold, " multiset")
    assert not results_match(result_from_rows(((2,), (1,))), gold, "multiset")


def test_list_preserves_row_order() -> None:
    gold = result_from_rows((("Ada",), ("Grace",)))
    assert results_match(gold, gold, "list")
    assert not results_match(result_from_rows((("Grace",), ("Ada",))), gold, "list")


def test_subset_supports_exact_and_minimum_counts() -> None:
    gold = result_from_rows((("Ada",), ("Grace",), ("Linus",)))
    assert results_match(result_from_rows((("Ada",),)), gold, "subset,=,1")
    assert results_match(result_from_rows((("Ada",), ("Grace",))), gold, "subset,>=,1")
    assert not results_match(result_from_rows((("Unknown",),)), gold, "subset,=,1")


def test_scalar_numeric_tolerance_and_blank_row_removal() -> None:
    assert results_match(result_from_rows(((100.9,),)), result_from_rows(((100.0,),)), "set")
    assert not results_match(result_from_rows(((101.1,),)), result_from_rows(((100.0,),)), "set")
    assert results_match(result_from_rows(((None,), ("Ada",))), result_from_rows((("Ada",),)), "set")


def test_unknown_grading_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        parse_grading_method("bag")
