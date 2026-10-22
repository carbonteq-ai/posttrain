import pytest
from progress_training_corpus import intended_split, schedule


def source():
    names = [f"{d}.task{i}" for d, count in {"finance": 3, "hr": 6, "marketing": 3, "operations": 5,
                                           "sales": 3, "simple": 22, "support": 4}.items() for i in range(count)]
    return {"candidate_tasks": names, "source_qualified_tasks": names[:], "protected_tasks": {"finance.eval": []}}


def test_full_corpus_is_covered_without_within_step_duplicates_or_outcome_selection():
    audit = source()
    result = schedule(audit)
    assert result == schedule(audit)
    assert result["expected_online_episodes"] == 640
    assert set(result["task_exposures"]) == set(audit["candidate_tasks"])
    assert all(len(r["tasks"]) == len(set(r["tasks"])) == 32 for r in result["steps"])
    assert all(len(r["category_counts"]) == 7 for r in result["steps"])
    assert not result["guidance_and_native_batch_qualified"]


def test_failing_or_protected_task_cannot_be_dropped_to_make_schedule_pass():
    audit = source()
    audit["source_qualified_tasks"] = audit["source_qualified_tasks"][:-1]
    with pytest.raises(ValueError, match="repair every candidate"):
        schedule(audit)
    audit = source()
    audit["protected_tasks"][audit["candidate_tasks"][0]] = ["heldout"]
    with pytest.raises(ValueError, match="protected"):
        schedule(audit)


def test_intended_split_retains_prior_training_and_reviews_old_reserves():
    names = [f"simple.train{i}" for i in range(124)]
    heldout = [f"simple.eval{i}" for i in range(62)]
    policy = intended_split({"task_names": names}, {"task_names": heldout},
                            {"reasons": {names[0]: ["old-training.json"], names[1]: ["old-reserve.json"]}})
    assert policy["training_task_names"] == names
    assert not set(names) & set(policy["reasons"])
    assert len(policy["historical_exclusion_review"]) == 2
    assert not policy["additional_exclusions"]
    assert not policy["prior_training_disqualifies"]
    with pytest.raises(ValueError, match="overlap"):
        intended_split({"task_names": names}, {"task_names": names[:1] + heldout[1:]}, {})


def test_full_124_schedule_keeps_all_tasks():
    counts = {"finance": 11, "hr": 11, "marketing": 10, "operations": 16,
              "sales": 5, "support": 10, "simple": 61}
    names = [f"{d}.task{i}" for d, count in counts.items() for i in range(count)]
    result = schedule({"candidate_tasks": names, "source_qualified_tasks": names[:], "protected_tasks": {}})
    assert result["task_count"] == 124
    assert len(result["task_exposures"]) == 124
    assert all(len(set(row["tasks"])) == 32 for row in result["steps"])
