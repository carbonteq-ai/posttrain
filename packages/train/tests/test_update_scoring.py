from types import SimpleNamespace

import pytest
from posttrain.environment.verifiers_conditioning import materialize_native_conditioning, native_conditioning_records
from posttrain.train.update_records import (
    ConditioningView,
    InvalidPolicyUpdate,
    PolicyVersions,
    PopulationSnapshot,
)

torch = pytest.importorskip("torch")

from posttrain.train.backends.policy_update_scoring import (  # noqa: E402
    dense_scores,
    freeze_population_scores,
    sampled_logprobs,
    score_views,
)


def inputs():
    trace = SimpleNamespace(
        id="trace",
        calls=[SimpleNamespace(node=1)],
        nodes=[
            SimpleNamespace(
                parent=None, sampled=False, message={"role": "system"}, token_ids=[1, 2], mask=[False, False]
            ),
            SimpleNamespace(
                parent=0, sampled=True, message={"role": "assistant"}, token_ids=[3, 4, 5], mask=[False, True, True]
            ),
        ],
    )
    record = native_conditioning_records(trace, sampled_node_indices=(1,), context_contract="causal-text@1")[0]
    return materialize_native_conditioning(trace, record)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_original_coordinates_use_previous_logit_and_exclude_context_gradient(dtype, temperature):
    logits = torch.arange(35, dtype=torch.float32).reshape(1, 5, 7).div(13).to(dtype).requires_grad_()
    actual = sampled_logprobs(logits, inputs(), sampled_indices=(1, 2), score_temperature=temperature)
    expected = torch.stack(
        [
            logits[0, 2].float()[4] / temperature - torch.logsumexp(logits[0, 2].float() / temperature, 0),
            logits[0, 3].float()[5] / temperature - torch.logsumexp(logits[0, 3].float() / temperature, 0),
        ]
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    (-actual.sum()).backward()
    gradient = logits.grad.float()
    assert gradient[0, [0, 1, 4]].count_nonzero() == 0
    for position, target in ((2, 4), (3, 5)):
        reference = (logits.detach()[0, position].float() / temperature).softmax(0)
        reference[target] -= 1
        reference /= temperature
        tolerance = 0.003 if dtype == torch.bfloat16 else 0.0005
        torch.testing.assert_close(gradient[0, position], reference, rtol=tolerance, atol=tolerance)


def test_nonfinite_unused_context_does_not_poison_selected_score():
    logits = torch.zeros(1, 5, 7, requires_grad=True)
    with torch.no_grad():
        logits[0, 0] = float("nan")
    assert sampled_logprobs(logits, inputs(), sampled_indices=(2,), score_temperature=1)[0].isfinite()
    with torch.no_grad():
        logits[0, 3, 0] = float("inf")
    with pytest.raises(InvalidPolicyUpdate, match="non-finite"):
        sampled_logprobs(logits, inputs(), sampled_indices=(2,), score_temperature=1)


@pytest.mark.parametrize("indices", [(0,), (1, 1), (True,)])
def test_scoring_rejects_context_and_duplicate_coordinates(indices):
    with pytest.raises(InvalidPolicyUpdate, match="ineligible|duplicates"):
        sampled_logprobs(torch.zeros(1, 5, 7), inputs(), sampled_indices=indices, score_temperature=1)


def test_shifted_or_truncated_logits_cannot_be_used():
    with pytest.raises(InvalidPolicyUpdate, match="complete original"):
        sampled_logprobs(torch.zeros(1, 4, 7), inputs(), sampled_indices=(1,), score_temperature=1)


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf"), True])
def test_invalid_score_temperature_rejected(temperature):
    with pytest.raises(InvalidPolicyUpdate, match="temperature"):
        sampled_logprobs(torch.zeros(1, 5, 7), inputs(), sampled_indices=(1,), score_temperature=temperature)


class CausalModel(torch.nn.Module):
    """Small independent causal operator; this does not qualify an architecture.

    Exposes Hugging Face's causal-LM split: ``get_decoder()`` returns the final
    hidden states and ``get_output_embeddings()`` the (biased) output head.
    """

    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(7, 3)
        self.head = torch.nn.Linear(3, 7)

    def hidden(self, input_ids, attention_mask, position_ids, use_cache):
        assert not use_cache
        assert attention_mask.tolist() == [[1] * input_ids.shape[1]]
        assert position_ids.tolist() == [list(range(input_ids.shape[1]))]
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids).cumsum(dim=1))

    def get_decoder(self):
        return self.hidden

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids, attention_mask, position_ids, use_cache):
        hidden = self.hidden(input_ids, attention_mask, position_ids, use_cache).last_hidden_state
        return SimpleNamespace(logits=self.head(hidden))


def score_population():
    source = inputs()
    view = ConditioningView(
        "context",
        "native",
        "tokens",
        "causal",
        "positions",
        "template@1",
        source.record.input_digest,
        len(source.token_ids),
        "episode",
        "branch",
        tuple(local for local, _ in source.action_positions),
    )
    snapshot = PopulationSnapshot(
        "population",
        "native",
        "native-digest",
        (view,),
        (),
        (),
        PolicyVersions("sample@1", "old@1", "current@1", None),
        "all@1",
    )
    return snapshot, source, snapshot.actions()


def test_model_scoring_retains_full_prefix_gradients_and_frozen_old_scores():
    torch.manual_seed(7)
    model = CausalModel()
    snapshot, source, actions = score_population()
    views = tuple(range(len(snapshot.conditioning)))
    kwargs = dict(read_input=lambda view: source, device=torch.device("cpu"), score_temperature=0.7)
    old = freeze_population_scores(
        model, snapshot, policy_version="old@1", score_contract="causal-temperature@1", **kwargs
    )
    assert old.values.shape == (snapshot.size,) == (len(actions),)
    before = old.values.clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(2):
        optimizer.zero_grad()
        current = score_views(model, snapshot, views, **kwargs)
        assert current.positions.tolist() == list(range(snapshot.size))
        loss = -current.values.sum()
        loss.backward()
        # The first system token participates through causal context, despite
        # having no policy loss. Detached context/KV would erase this gradient.
        assert model.embedding.weight.grad[1].abs().sum() > 0
        assert model.embedding.weight.grad[5].count_nonzero() == 0
        optimizer.step()
    refreshed = dense_scores(
        snapshot.size, (score_views(model, snapshot, views, **kwargs),), device=torch.device("cpu")
    )
    assert any(not torch.equal(refreshed[position], before[position]) for position in range(snapshot.size))
    assert torch.equal(old.values, before) and not old.values.requires_grad
    old.validate(snapshot, policy_version="old@1", score_contract="causal-temperature@1", score_temperature=0.7)
    with pytest.raises(InvalidPolicyUpdate, match="different evidence, policy or score contract"):
        old.validate(snapshot, policy_version="old@1", score_contract="causal-temperature@1", score_temperature=1)


def test_model_scoring_rejects_mismatched_materialized_input_before_forward():
    from dataclasses import replace

    snapshot, source, _ = score_population()
    source = replace(source, record=replace(source.record, input_digest="other-context"))
    with pytest.raises(InvalidPolicyUpdate, match="frozen conditioning view"):
        score_views(
            CausalModel(),
            snapshot,
            (0,),
            read_input=lambda view: source,
            device=torch.device("cpu"),
            score_temperature=1,
        )
