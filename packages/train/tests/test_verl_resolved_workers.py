"""Qualified score-engine specialization uses native construction hooks."""

from types import SimpleNamespace

import pytest
from posttrain.train.backends.verl.resolved_workers import resolved_worker_types
from posttrain.train.update_records import InvalidPolicyUpdate


def worker_types(*, validate_engine=None, **changes):
    created = []

    class NativeEngine:
        def __init__(self, **kwargs):
            created.append(kwargs)

    class NativeTrainingWorker:
        def create_engine(self):
            raise AssertionError("recipe should specialize construction")

    class NativeActorWorker:
        actor_worker_cls = NativeTrainingWorker

    registry = SimpleNamespace(get_engine_cls=lambda **kwargs: NativeEngine)
    training, actor = resolved_worker_types(NativeTrainingWorker, NativeActorWorker, registry,
        validate_engine=validate_engine)
    worker = object.__new__(training)
    worker.config = SimpleNamespace(model_type="language_model")
    worker.model_config = {"use_remove_padding": False}
    options = dict(strategy="fsdp", use_fused_kernels=False, use_dynamic_bsz=False, micro_batch_size_per_gpu=1)
    options.update(changes)
    worker.engine_config = SimpleNamespace(**options)
    worker.optimizer_config, worker.checkpoint_config = object(), object()
    return worker, actor, NativeActorWorker, created


def test_factory_selects_score_subclass_without_mutating_native_worker():
    worker, actor, original, created = worker_types()
    engine = worker.create_engine()
    assert hasattr(engine, "prepare_model_outputs")
    assert actor.actor_worker_cls is type(worker)
    assert original.actor_worker_cls is not type(worker)
    assert created == [dict(model_config=worker.model_config, engine_config=worker.engine_config,
                            optimizer_config=worker.optimizer_config, checkpoint_config=worker.checkpoint_config)]


@pytest.mark.parametrize("changes", [{"strategy": "fsdp2"}, {"use_fused_kernels": True},
                                      {"use_dynamic_bsz": True}, {"micro_batch_size_per_gpu": 2}])
def test_factory_rejects_unqualified_execution_before_engine_creation(changes):
    worker, _, _, created = worker_types(**changes)
    with pytest.raises(InvalidPolicyUpdate, match="qualified dense"):
        worker.create_engine()
    assert not created


def test_factory_rejects_native_worker_without_extension():
    with pytest.raises(InvalidPolicyUpdate, match="engine factory extension"):
        resolved_worker_types(object, object, object())


def test_effective_profile_validation_rejects_before_native_model_allocation():
    seen = []

    def reject(model, engine):
        seen.append((model, engine))
        raise InvalidPolicyUpdate("effective profile mismatch")

    worker, _, _, created = worker_types(validate_engine=reject)
    with pytest.raises(InvalidPolicyUpdate, match="effective profile mismatch"):
        worker.create_engine()
    assert seen == [(worker.model_config, worker.engine_config)]
    assert not created
