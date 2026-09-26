"""Run cards: the note template of a run's job kind, rendered for that run.

Framework templates live beside this module in ``note_templates/``, one per
job family (group-policy training shares one template, for example). A
project overrides any job kind by placing ``<job kind>.md`` (or the family
file) in its ``.posttrain/note_templates/`` directory. A template starts with
a front-matter line naming its id and revision, ``template: <id>@<n>``,
followed by ``---``, so a card records which template produced it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

TEMPLATE_FAMILIES: dict[str, str] = {
    "train.grpo": "group-policy",
    "train.gdpo": "group-policy",
    "train.capo": "group-policy",
    "train.sampo": "sampo",
    "train.sft": "supervised",
    "train.dpo": "preference",
    "train.distill": "distill",
    "eval.general": "evaluation",
    "eval.domain": "evaluation",
    "serve.benchmark": "serving-benchmark",
    "serve.smoke": "serving-smoke",
    "data.prepare": "data",
}
GENERIC = "generic"
_FRONT_MATTER = re.compile(r"\Atemplate:\s*(?P<id>[a-z0-9._-]+@\d+)\s*\n---\s*\n")


@dataclass(frozen=True, slots=True)
class NoteTemplate:
    template_id: str
    body: str
    path: str


def split_front_matter(text: str, *, fallback_id: str) -> tuple[str, str]:
    match = _FRONT_MATTER.match(text)
    if match is None:
        return fallback_id, text
    return match.group("id"), text[match.end() :]


class TemplateSet:
    """Framework templates overlaid by an optional project directory."""

    def __init__(self, project_dir: str | Path | None = None) -> None:
        self._project = Path(project_dir) if project_dir else None

    def for_job_kind(self, job_kind: str) -> NoteTemplate:
        family = TEMPLATE_FAMILIES.get(job_kind, GENERIC)
        for name in dict.fromkeys((job_kind, family, GENERIC)):
            if self._project is not None:
                path = self._project / f"{name}.md"
                if path.is_file():
                    template_id, body = split_front_matter(
                        path.read_text(encoding="utf-8"), fallback_id=f"project:{name}@0"
                    )
                    return NoteTemplate(template_id, body, str(path))
            framework = resources.files(__package__).joinpath("note_templates", f"{name}.md")
            if framework.is_file():
                template_id, body = split_front_matter(framework.read_text(encoding="utf-8"), fallback_id=f"{name}@0")
                return NoteTemplate(template_id, body, f"note_templates/{name}.md")
        raise LookupError(f"no note template for job kind {job_kind!r}")

    def framework_names(self) -> tuple[str, ...]:
        return tuple(sorted({*TEMPLATE_FAMILIES.values(), GENERIC}))


__all__ = ["GENERIC", "NoteTemplate", "TEMPLATE_FAMILIES", "TemplateSet", "split_front_matter"]
