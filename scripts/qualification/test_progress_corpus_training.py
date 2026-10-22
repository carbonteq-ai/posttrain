import hashlib
import json
import shutil
from pathlib import Path

import pytest
from progress_corpus_training import gate, retained_population
from progress_training_corpus import schedule


def fixture(tmp_path):
    counts = {"finance": 11, "hr": 11, "marketing": 10, "operations": 16,
              "sales": 5, "simple": 61, "support": 10}
    tasks = [f"{d}.task{i}" for d, count in counts.items() for i in range(count)]
    plan = schedule({"candidate_tasks": tasks, "source_qualified_tasks": tasks, "protected_tasks": {}})
    heldout = {"task_names": [f"simple.eval{i}" for i in range(62)]}
    (tmp_path / "evidence/run").mkdir(parents=True)
    (tmp_path / "evidence/cpu").mkdir()
    (tmp_path / "driver").mkdir()
    for name in ("progress_teacher_context.py", "sdpo_compact_guidance.py", "sdpo_prompting.py"):
        shutil.copyfile(Path(__file__).parent / name, tmp_path / "driver" / name)
    (tmp_path / "cohorts.json").write_text(json.dumps({"train": tasks}))
    (tmp_path / "heldout62.json").write_text(json.dumps(heldout))
    (tmp_path / "experiment.json").write_text('{}')
    (tmp_path / "evidence/cpu/materials.json").write_text('[]')
    (tmp_path / "evidence/native-audit.json").write_text('{}')
    (tmp_path / "evidence/teacher-token-bindings.json").write_text('{}')
    (tmp_path / "source-sha256.txt").write_text(''.join(
        hashlib.sha256(p.read_bytes()).hexdigest() + '  ' + str(p.relative_to(tmp_path)) + '\n'
        for p in sorted((tmp_path / "driver").iterdir())))
    (tmp_path / "evidence/run/receipt.json").write_text(json.dumps({"runtime_completed": True,
        "optimizer_updates": 0, "native_episodes": 496, "same_decision_generations": 372}))
    approval = {"qualified": True, "semantic_qualified": True, "native_execution_qualified": True,
        "exact_token_bindings_qualified": True, "known_blockers": [], "task_count": 124,
        "task_reviews": [{"task": t, "decision": "pass", "evidence": "unit-fixture-only"} for t in tasks],
        "qualification_files_sha256": {str(p.relative_to(tmp_path)): hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in tmp_path.rglob('*') if p.is_file()}}
    (tmp_path / "evidence/qualification-gate.json").write_text(json.dumps(approval))
    return plan, heldout, approval


def test_missing_semantic_review_blocks_even_a_completed_runtime(tmp_path):
    plan, heldout, approval = fixture(tmp_path)
    approval["semantic_qualified"] = False
    (tmp_path / "evidence/qualification-gate.json").write_text(json.dumps(approval))
    with pytest.raises(ValueError, match="has not passed"):
        gate(tmp_path, plan, heldout)


def test_all124_reviews_and_source_integrity_required(tmp_path):
    plan, heldout, approval = fixture(tmp_path)
    assert gate(tmp_path, plan, heldout)["task_count"] == 124
    approval["task_reviews"].pop()
    (tmp_path / "evidence/qualification-gate.json").write_text(json.dumps(approval))
    with pytest.raises(ValueError, match="each task"):
        gate(tmp_path, plan, heldout)


def test_no_task_removal_duplicate_or_heldout_substitution(tmp_path):
    plan, heldout, _ = fixture(tmp_path)
    heldout["task_names"][0] = plan["corpus_tasks"][0]
    with pytest.raises(ValueError, match="exact124"):
        gate(tmp_path, plan, heldout)


def test_qualified_source_change_invalidates_gate(tmp_path):
    plan, heldout, _ = fixture(tmp_path)
    path = tmp_path / "driver/progress_teacher_context.py"
    path.write_text(path.read_text() + '\n# changed after qualification\n')
    with pytest.raises(ValueError, match="source snapshot changed"):
        gate(tmp_path, plan, heldout)


def test_different_disjoint62_tasks_do_not_substitute_for_the_frozen_evaluation(tmp_path):
    plan, heldout, _ = fixture(tmp_path)
    heldout["task_names"][0] = "simple.different_unseen_task"
    with pytest.raises(ValueError, match="exact124"):
        gate(tmp_path, plan, heldout)


def test_transport_copy_cannot_replace_retained_source_fields_or_duplicate_tasks():
    names = [f"simple.t{i}" for i in range(32)]
    wire = [{"id": f"{i}-{replica}", "task": {"data": {"name": name}}}
            for i, name in enumerate(names) for replica in range(2)]
    retained = [{**row, "original_source_extension": "preserved"} for row in wire]
    result = retained_population(wire, retained, names)
    assert all(row["original_source_extension"] == "preserved" for row in result)
    assert "original_source_extension" not in wire[0]
    retained[-1] = retained[0]
    with pytest.raises(ValueError, match="identity differs"):
        retained_population(wire, retained, names)
