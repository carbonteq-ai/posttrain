import asyncio
import copy
from types import SimpleNamespace

import pytest
from prepare_sdpo_guidance_five import BoundedRequests, parallel_checks_enabled
from sdpo_checkwise_preparation import prepare


def material():
    return {"task": "simple.example", "user_request": "Read then create one contact.",
        "native_data": {"initial_state": {}},
        "tool_definitions": [{"type": "function", "function": {"name": "create"}}],
        "goal_manifest": {"sources": {}, "checks": [{"check_id": name, "role": "goal",
            "operator": "effects.required_when@1"} for name in ("first", "second", "third")]},
        "reference_actions": [{"call": {"name": "create", "arguments": {}},
                               "result": {"observed_result": {"success": True}}}]}


def test_request_bound_applies_globally_and_releases_after_an_error():
    async def exercise():
        requests = BoundedRequests(3)

        async def work(index):
            await asyncio.sleep(0)
            if index == 4:
                raise ValueError("Retain the failed model response")
            return index

        results = await asyncio.gather(*(requests.call(lambda i=i: work(i)) for i in range(10)),
                                       return_exceptions=True)
        assert requests.evidence()["peak_in_flight"] == 3
        assert requests.evidence()["in_flight_at_snapshot"] == 0
        assert requests.evidence()["submitted"] == 10
        assert isinstance(results[4], ValueError)
        assert results[:4] == [0, 1, 2, 3]

    asyncio.run(exercise())


def test_twenty_task_compilation_uses_forty_requests_without_skipping_reviews():
    async def exercise():
        requests = BoundedRequests(40)
        reviewed = set()

        async def task(index):
            source = material()
            source["task"] = f"simple.task-{index}"

            async def response(stage, prompt, data):
                await asyncio.sleep(0)
                if stage == "check_brief":
                    return {"check_id": data["check"]["check_id"], "role": "goal",
                            "instruction": "Create one contact."}
                if stage == "check_review":
                    reviewed.add((index, data["check"]["check_id"]))
                if stage.startswith("workflow") or stage == "bundle_review":
                    assert all((index, name) in reviewed for name in ("first", "second", "third"))
                if stage.endswith("review"):
                    return {"approved": True, "issues": [], "reason": "Source preserved."}
                return {"steps": [{"tool": "create", "instruction": "Create using observed values."}]}

            async def generate(stage, prompt, data):
                return await requests.call(lambda: response(stage, prompt, data))

            return await prepare(source, generate, lambda *_: None,
                                 private_preparation=True, parallel_checks=True)

        results = await asyncio.gather(*(task(i) for i in range(20)))
        assert len(results) == 20 and all(r["review_approved"] for r in results)
        assert len(reviewed) == 60
        assert requests.evidence()["peak_in_flight"] == 40
        assert requests.evidence()["submitted"] == 180
        assert requests.evidence()["in_flight_at_snapshot"] == 0

    asyncio.run(exercise())


def test_parallel_first_stage_preflight_observes_every_check_before_raising():
    captured = []

    class Captured(Exception):
        pass

    async def generate(stage, prompt, data):
        assert stage == "check_brief"
        captured.append(data["check"]["check_id"])
        raise Captured()

    with pytest.raises(Captured):
        asyncio.run(prepare(material(), generate, lambda *_: None, parallel_checks=True))
    assert captured == ["first", "second", "third"]


def test_parallel_checks_preserve_source_order_and_wait_for_every_review():
    source = material()
    original = copy.deepcopy(source)
    reviewed, started, stages = set(), [], []
    release_first = asyncio.Event()

    async def generate(stage, prompt, data):
        stages.append(stage)
        if stage == "check_brief":
            name = data["check"]["check_id"]
            started.append(name)
            if name == "first":
                await release_first.wait()
            if name == "third":
                release_first.set()
            return {"check_id": name, "role": "goal", "instruction": "Create one contact."}
        if stage == "check_review":
            reviewed.add(data["check"]["check_id"])
        if stage.startswith("workflow") or stage == "bundle_review":
            assert reviewed == {"first", "second", "third"}
            assert [r["check_id"] for r in data["requirements"]] == ["first", "second", "third"]
        if stage.endswith("review"):
            return {"approved": True, "issues": [], "reason": "Source preserved."}
        return {"steps": [{"tool": "create", "instruction": "Create using observed values."}]}

    result = asyncio.run(prepare(source, generate, lambda *_: None,
                                 private_preparation=True, parallel_checks=True))
    assert started == ["first", "second", "third"]
    assert result["review_approved"]
    assert stages.count("check_review") == 3 and stages[-1] == "bundle_review"
    assert source == original


def test_one_rejected_check_rejects_the_task_after_other_children_are_observed():
    reviewed = set()
    workflow_calls = []

    async def generate(stage, prompt, data):
        await asyncio.sleep(0)
        if stage.startswith("check_brief"):
            return {"check_id": data["check"]["check_id"], "role": "goal",
                    "instruction": "Create one contact."}
        if stage == "check_review":
            name = data["check"]["check_id"]
            reviewed.add(name)
            return {"approved": name != "second", "issues": ["Unsupported"] if name == "second" else [],
                    "reason": "Source review."}
        workflow_calls.append(stage)
        raise AssertionError("A rejected task cannot advance to assembly")

    with pytest.raises(ValueError, match="checkwise review rejected: second"):
        asyncio.run(prepare(material(), generate, lambda *_: None, parallel_checks=True))
    assert reviewed == {"first", "second", "third"}
    assert not workflow_calls


def test_individual_approvals_still_cannot_override_global_contradiction_rejection():
    async def generate(stage, prompt, data):
        await asyncio.sleep(0)
        if stage == "check_brief":
            return {"check_id": data["check"]["check_id"], "role": "goal",
                    "instruction": "Create one contact."}
        if stage == "bundle_review":
            return {"approved": False, "issues": ["Contradictory target rule"], "reason": "Reject"}
        if stage.endswith("review"):
            return {"approved": True, "issues": [], "reason": "Source preserved."}
        return {"steps": [{"tool": "create", "instruction": "Create using observed values."}]}

    with pytest.raises(ValueError, match="assembled source review rejected"):
        asyncio.run(prepare(material(), generate, lambda *_: None,
                            private_preparation=True, parallel_checks=True))


def test_parallel_mode_is_opt_in_and_rejects_noncheckwise_recipes():
    assert not parallel_checks_enabled(SimpleNamespace(recipe="staged"))
    assert parallel_checks_enabled(SimpleNamespace(recipe="private-checkwise", parallel_checks=True))
    with pytest.raises(ValueError, match="checkwise recipe"):
        parallel_checks_enabled(SimpleNamespace(recipe="staged", parallel_checks=True))
