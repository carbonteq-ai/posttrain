from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest
from posttrain_observatory import FixtureRunDataSource, ObservatoryService
from posttrain_observatory.note_render import build_query, data_blocks, parse_block
from posttrain_observatory.run_cards import GENERIC, TEMPLATE_FAMILIES, TemplateSet
from posttrain_observatory.semantic_layer.execute import applies
from posttrain_observatory.semantic_layer.framework import FRAMEWORK_MODEL
from posttrain_observatory.semantic_layer.query import SemanticQuery, split_measure

_FAMILIES = sorted({*TEMPLATE_FAMILIES.values(), GENERIC})


@pytest.mark.parametrize("family", _FAMILIES)
def test_framework_templates_use_only_names_their_job_kinds_provide(family: str) -> None:
    kinds = [kind for kind, name in TEMPLATE_FAMILIES.items() if name == family] or ["*"]
    template = TemplateSet().for_job_kind(kinds[0] if family != GENERIC else "model.transform")
    assert template.template_id.startswith(f"{family}@")
    dimensions = {dimension.name for dimension in FRAMEWORK_MODEL.dimensions}
    metrics = {metric.name for metric in FRAMEWORK_MODEL.metrics}
    names: set[str] = set()
    for _, body in data_blocks(template.body):
        query = build_query(parse_block(body), "run")
        assert isinstance(query, SemanticQuery)
        for value in query.measures:
            name, _ = split_measure(value)
            names.add(name)
            if name in metrics:
                continue
            measure = FRAMEWORK_MODEL.measure(name)
            assert all(applies(measure, kind) for kind in kinds if kind != "*"), (family, name)
        assert set(query.by) <= dimensions, (family, query.by)
    for reference in re.findall(r"\{\{\s*(run\.[a-z_]+)", template.body):
        assert reference in dimensions, (family, reference)
    assert names or family == GENERIC


def test_fixture_runs_render_cards_without_unresolved_references() -> None:
    service = ObservatoryService({"fixture": FixtureRunDataSource()})
    for run_id in (
        "runs/sft-calm-harbor",
        "runs/dpo-amber-field",
        "runs/grpo-silver-pine",
        "runs/eval-violet-river",
        "runs/serve-cedar-point",
    ):
        card = asyncio.run(service.run_card(run_id))
        assert card.unresolved == (), (run_id, card.unresolved)
        assert card.template is not None and "{{" not in card.text


def test_project_templates_override_framework_ones(tmp_path: Path) -> None:
    (tmp_path / "train.grpo.md").write_text("template: lab-grpo@3\n---\nLearning rate {{run.learning_rate}}.\n")
    templates = TemplateSet(tmp_path)
    assert templates.for_job_kind("train.grpo").template_id == "lab-grpo@3"
    assert templates.for_job_kind("train.gdpo").template_id == "group-policy@1"
    assert templates.for_job_kind("unknown.kind").template_id == "generic@1"
