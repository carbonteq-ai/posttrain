"""Focused tests for loss-only SFT validation and its evidence contract."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from posttrain.common import (
    CatalogRef,
    EventObservation,
    ExecutionTarget,
    MetricBatchObservation,
    MetricObservation,
    ProducedArtifact,
    RunContext,
    TraceFactUpdateObservation,
    TraceObservation,
)
from posttrain.common.variants import QWEN_35_2B
from posttrain.data import SupervisedDataset, SupervisedExample, SupervisedMedia
from posttrain.train import (
    QWEN35_RENDERER,
    QLoRAUpdate,
    SFTRequest,
    SFTSettings,
    SFTValidationSettings,
    TrainingBinding,
    TrainingLoop,
    TrainingParallelism,
)
from posttrain.train.backends.trl.common import callback_type
from posttrain.train.backends.trl.sft import (
    _emit_rendered_profile,
    _evaluated_at_step,
    _observed_sft_trainer_type,
    _sft_arguments,
    _visual_modality,
    _visual_row,
    _visual_text_token_counts,
)
from posttrain.train.catalog_schema import decode_training_selection
from posttrain.train.rendering import RenderedSFTExample


@dataclass
class CaptureObserver:
    events_seen: list[EventObservation] = field(default_factory=list)
    metrics_seen: list[MetricBatchObservation] = field(default_factory=list)

    def event(self, observation: EventObservation) -> None:
        self.events_seen.append(observation)

    def metric(self, observation: MetricObservation) -> None:
        self.metrics_seen.append(MetricBatchObservation({observation.name: observation.value}, observation.step))

    def metrics(self, observation: MetricBatchObservation) -> None:
        self.metrics_seen.append(observation)

    def trace(self, observation: TraceObservation) -> None:
        del observation

    def trace_fact_update(self, observation: TraceFactUpdateObservation) -> None:
        del observation

    def artifact(self, artifact: ProducedArtifact) -> None:
        del artifact


def _context(tmp_path: Path, observer: CaptureObserver) -> RunContext:
    return RunContext(
        project_id="projects/test",
        work_package_id="work-packages/sft-validation",
        run_id="runs/sft-validation",
        job_kind="train.sft",
        job_definition_version="1",
        workspace=tmp_path,
        observer=observer,
    )


def _request() -> SFTRequest:
    loop = TrainingLoop(max_steps=4, per_device_batch_size=2)
    settings = SFTSettings(
        "qwen3.5-2b/sft-validation-test",
        loop,
        validation=SFTValidationSettings(
            steps=2,
            per_device_batch_size=1,
            on_start=True,
            at_end=True,
        ),
    )
    target = ExecutionTarget("targets/test", "1", "cpu", placement={"world_size": 1})
    training = TrainingBinding(
        "training/test",
        "1",
        "trl@test",
        QWEN35_RENDERER,
        QLoRAUpdate(),
        target,
        TrainingParallelism(sequence_length_divisor=2),
    )
    train = SupervisedDataset(
        "dataset/train",
        "a" * 40,
        (
            SupervisedExample(
                "example/train",
                (
                    {"role": "user", "content": "Prompt"},
                    {"role": "assistant", "content": "Answer"},
                ),
                (1,),
            ),
        ),
    )
    validation = SupervisedDataset(
        "dataset/validation",
        "b" * 40,
        (
            SupervisedExample(
                "example/validation",
                (
                    {"role": "user", "content": "Held-out prompt"},
                    {"role": "assistant", "content": "Held-out answer"},
                ),
                (1,),
            ),
        ),
    )
    return SFTRequest(QWEN_35_2B, train, settings, training, validation_data=validation)


def test_catalog_decodes_explicit_sft_validation_schedule() -> None:
    settings = decode_training_selection(
        CatalogRef("training", "qwen3.5-2b/sft-validation-test"),
        {
            "selection_type": "sft-settings",
            "id": "qwen3.5-2b/sft-validation-test",
            "revision": "2",
            "loop": {"max_steps": 8},
            "validation": {
                "steps": 4,
                "per_device_batch_size": 2,
                "on_start": True,
                "at_end": True,
            },
            "visual_no_truncation": False,
        },
        {},
    )

    assert isinstance(settings, SFTSettings)
    assert settings.validation == SFTValidationSettings(4, 2, True, True)
    assert settings.visual_no_truncation is False


def test_sft_arguments_enable_loss_only_validation() -> None:
    arguments = _sft_arguments(_request(), Path("/tmp/sft-validation"))

    assert arguments["eval_strategy"] == "steps"
    assert arguments["eval_steps"] == 2
    assert arguments["eval_on_start"] is True
    assert arguments["per_device_eval_batch_size"] == 1
    assert arguments["prediction_loss_only"] is True


def test_visual_sft_arguments_enable_trl_vlm_preparation_without_truncation() -> None:
    arguments = _sft_arguments(_request(), Path("/tmp/sft-validation"), visual=True)

    assert "dataset_kwargs" not in arguments
    assert arguments["completion_only_loss"] is True
    assert arguments["assistant_only_loss"] is False
    assert arguments["packing"] is False
    assert arguments["padding_free"] is False
    assert arguments["max_length"] is None


def test_visual_modality_rejects_mixed_examples() -> None:
    request = _request()
    train = request.data.load()
    media = SupervisedMedia("assets/document/page.png", "a" * 64, "image/png")
    mixed = SupervisedDataset(
        "dataset/mixed",
        "1",
        (
            train.examples[0],
            SupervisedExample(
                "example/visual",
                train.examples[0].messages,
                (1,),
                media=(media,),
            ),
        ),
    )

    try:
        _visual_modality(mixed, None, request)
    except ValueError as error:
        assert "cannot mix" in str(error)
    else:
        raise AssertionError("mixed-modality SFT dataset was accepted")


def test_visual_row_preserves_page_order_and_verifies_digests(tmp_path: Path) -> None:
    first = b"first"
    second = b"second"
    root = tmp_path.resolve()
    (root / "assets/document").mkdir(parents=True)
    (root / "assets/document/page-0001.png").write_bytes(first)
    (root / "assets/document/page-0002.png").write_bytes(second)
    example = SupervisedExample(
        "example/visual",
        (
            {"role": "user", "content": "Extract JSON"},
            {"role": "assistant", "content": "{}"},
        ),
        (1,),
        media=(
            SupervisedMedia("assets/document/page-0002.png", hashlib.sha256(second).hexdigest(), "image/png"),
            SupervisedMedia("assets/document/page-0001.png", hashlib.sha256(first).hexdigest(), "image/png"),
        ),
    )
    opened: list[str] = []

    class FakeImage:
        def __init__(self, path: Path) -> None:
            opened.append(path.name)

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def convert(self, mode: str):
            assert mode == "RGB"
            return self

        def copy(self):
            return object()

    image_module = SimpleNamespace(open=lambda path: FakeImage(path))

    row = _visual_row(example, root, {"Image": image_module})

    assert opened == ["page-0002.png", "page-0001.png"]
    assert row["prompt"] == [{"role": "user", "content": "Extract JSON"}]
    assert row["completion"] == [{"role": "assistant", "content": "{}"}]


def test_visual_profile_counts_text_tokens_without_labeling_images_as_tokens() -> None:
    train = _request().data.load()

    class Tokenizer:
        def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool):
            assert messages == [{"role": "user", "content": "Prompt"}]
            assert tokenize is True
            assert add_generation_prompt is True
            return [1, 2, 3]

        def __call__(self, content: str, *, add_special_tokens: bool):
            assert content == "Answer"
            assert add_special_tokens is False
            return {"input_ids": [4, 5]}

    assert _visual_text_token_counts(train, SimpleNamespace(tokenizer=Tokenizer())) == (3, 2)


def test_validation_logs_use_training_validation_namespace(tmp_path: Path) -> None:
    observer = CaptureObserver()
    callback = callback_type(_context(tmp_path, observer), {"TrainerCallback": object})()
    callback.on_log(
        SimpleNamespace(max_grad_norm=1.0),
        SimpleNamespace(global_step=4),
        SimpleNamespace(),
        logs={"eval_loss": 0.75, "grad_norm": 1.2},
    )

    values = observer.metrics_seen[-1].values
    assert values["train/validation/loss"] == 0.75
    assert values["train/grad_norm"] == 1.2
    assert values["train/gradient_clipped"] == 1.0
    assert "train/eval_loss" not in values


def test_every_sft_validation_call_has_its_own_runtime_phase(tmp_path: Path) -> None:
    observer = CaptureObserver()
    context = _context(tmp_path, observer)

    class Trainer:
        def evaluate(self, marker: str) -> str:
            return marker

    observed = _observed_sft_trainer_type(Trainer, context)()

    assert observed.evaluate("held-out") == "held-out"
    assert [event.name for event in observer.events_seen] == [
        "runtime_phase_started",
        "runtime_phase_completed",
    ]
    assert observer.events_seen[0].attributes["phase"] == "evaluation"
    assert observer.events_seen[0].attributes["backend"] == "trl"
    assert observer.events_seen[0].attributes["phase_id"] == observer.events_seen[1].attributes["phase_id"]


def test_rendered_profile_records_supervision_and_truncation(tmp_path: Path) -> None:
    observer = CaptureObserver()
    samples = (
        RenderedSFTExample("one", (1, 2, 3, 4), (-100, -100, 3, 4), 6, 3),
        RenderedSFTExample("two", (1, 2), (-100, 2), 2, 1),
    )

    _emit_rendered_profile(_context(tmp_path, observer), samples, 8, "train/data")

    values = observer.metrics_seen[-1].values
    assert values["train/data/examples"] == 2
    assert values["train/data/supervision_token_ratio"] == 0.5
    assert values["train/data/truncation_rate"] == 0.5
    assert values["train/data/truncated_tokens"] == 2
    assert values["train/data/truncated_supervised_tokens"] == 1
    assert values["train/data/sequence_length_p90"] == 4


def test_final_validation_is_not_repeated_when_final_step_was_evaluated() -> None:
    assert _evaluated_at_step([{"step": 4, "eval_loss": 0.5}], 4)
    assert not _evaluated_at_step([{"step": 2, "eval_loss": 0.7}], 4)
