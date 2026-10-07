from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest
from posttrain_observatory import ObservatoryService
from posttrain_observatory.note_render import build_query, data_blocks
from posttrain_observatory.run_cards import GENERIC, TEMPLATE_FAMILIES, TemplateSet
from posttrain_observatory.semantic_layer.framework import FRAMEWORK_MODEL
from posttrain_observatory.semantic_layer.model import Measure
from posttrain_observatory.semantic_layer.views import parse_statement, plan, table_columns
from posttrain_tracking_trackio import TrackioDataSource

_FAMILIES = sorted({*TEMPLATE_FAMILIES.values(), GENERIC})


@pytest.mark.parametrize("family", _FAMILIES)
def test_framework_templates_read_only_what_their_job_kinds_provide(family: str) -> None:
    kinds = [kind for kind, name in TEMPLATE_FAMILIES.items() if name == family]
    template = TemplateSet().for_job_kind(kinds[0] if kinds else "model.transform")
    assert template.template_id.startswith(f"{family}@")
    tables = table_columns(FRAMEWORK_MODEL)
    for name, body in data_blocks(template.body):
        planned = plan(FRAMEWORK_MODEL, parse_statement(build_query(body, "run").sql))
        for table, columns in planned.items():
            for column in columns:
                owner = tables[table][column]
                if isinstance(owner, Measure) and kinds:
                    assert "*" in owner.job_kinds or all(kind in owner.job_kinds for kind in kinds), (
                        family,
                        name,
                        column,
                    )
    dimensions = {dimension.name for dimension in FRAMEWORK_MODEL.dimensions}
    for reference in re.findall(r"\{\{\s*(run\.[a-z_]+)", template.body):
        assert reference in dimensions, (family, reference)


def test_cards_render_from_recorded_runs_without_unresolved_references(trackio_project: TrackioDataSource) -> None:
    service = ObservatoryService({"local": trackio_project})
    grpo = asyncio.run(service.run_card("grpo-a"))
    assert grpo.template == "group-policy@4" and grpo.unresolved == ()
    assert "| Updates | 3 of ? (last update 3) |" in grpo.text
    # A run that trained on each population once has one collection per update.
    assert "| Collections | 3 (last at update 3) |" in grpo.text
    assert "| Reward, first → last 10 collections |" in grpo.text
    assert "| Collection time spent in rollouts | 85.5% |" in grpo.text  # 530 / 620
    sampo = asyncio.run(service.run_card("sampo-b"))
    assert sampo.template == "sampo@4" and sampo.unresolved == ()
    assert "| Error | OutOfMemoryError |" in sampo.text
    assert "| Failed in | actor_update (update 2) |" in sampo.text


def test_cards_count_collections_and_updates_of_the_resolved_engine(resolved_project: TrackioDataSource) -> None:
    service = ObservatoryService({"local": resolved_project})
    for run_id in ("resolved-tagged", "resolved-untagged"):
        card = asyncio.run(service.run_card(run_id))
        assert card.template == "sampo@4" and card.unresolved == (), run_id
        assert "| Collections | 2 (last at update 3) |" in card.text, run_id
        assert "| Updates | 4 of ? (last update 4) |" in card.text, run_id
        assert "| Reward, first → last 10 collections | 0.500 → 0.500 |" in card.text, run_id
        # Each collection's rollout time over its two updates' step time: 600 / (2 * 150).
        assert "| Collection time spent in rollouts | 200.0% |" in card.text, run_id


def test_project_templates_override_framework_ones(tmp_path: Path) -> None:
    (tmp_path / "train.grpo.md").write_text("template: lab-grpo@3\n---\nLearning rate {{run.learning_rate}}.\n")
    templates = TemplateSet(tmp_path)
    assert templates.for_job_kind("train.grpo").template_id == "lab-grpo@3"
    assert templates.for_job_kind("train.gdpo").template_id == "group-policy@4"
    assert templates.for_job_kind("unknown.kind").template_id == "generic@2"
