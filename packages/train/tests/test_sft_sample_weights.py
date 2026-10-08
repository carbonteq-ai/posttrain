"""Per-example SFT sampling weights balance tasks with many examples against tasks with few."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from posttrain.data import SupervisedDataset, SupervisedExample
from posttrain.train.backends.trl.sft import _observed_sft_trainer_type, sample_weights

MESSAGES = ({"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"})


def dataset(*weights: float | None) -> SupervisedDataset:
    return SupervisedDataset(
        "weights",
        "e" * 40,
        tuple(
            SupervisedExample(
                f"case-{index}", MESSAGES, (1,), metadata={} if weight is None else {"sample_weight": weight}
            )
            for index, weight in enumerate(weights)
        ),
    )


def test_sample_weights_are_read_only_when_every_example_has_one() -> None:
    assert sample_weights(dataset(None, None)) is None
    assert sample_weights(dataset(0.25, 1.0)) == [0.25, 1.0]
    with pytest.raises(ValueError, match="sample_weight"):
        sample_weights(dataset(0.5, None))
    with pytest.raises(ValueError, match="sample_weight"):
        sample_weights(dataset(0.5, 0.0))


def test_weighted_trainer_draws_a_seeded_weighted_sampler() -> None:
    torch = pytest.importorskip("torch")

    class Parent:
        def __init__(self) -> None:
            self.args = SimpleNamespace(seed=7)

        def _get_train_sampler(self, train_dataset=None):
            return "sequential"

    context = SimpleNamespace(phase=None)
    unweighted = _observed_sft_trainer_type(Parent, context)()  # type: ignore[arg-type]
    assert unweighted._get_train_sampler() == "sequential"
    weighted = _observed_sft_trainer_type(Parent, context, [0.1, 0.9])()  # type: ignore[arg-type]
    sampler = weighted._get_train_sampler()
    assert isinstance(sampler, torch.utils.data.WeightedRandomSampler)
    draws = [index for _ in range(50) for index in weighted._get_train_sampler()]
    assert draws.count(1) > draws.count(0)
    assert list(weighted._get_train_sampler()) == list(weighted._get_train_sampler())
