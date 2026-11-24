"""ches_rank.py: positions, the CHES formula, selection under the caps, and the hidden-state mean on a tiny Qwen3.5."""
import pytest

from ches_rank import ches, pick, predicting_positions


def test_predicting_positions_are_one_before_each_trained_token():
    # prompt of 3; side tokens 0..4 with tokens 1, 2 and 4 trained -> predicted at 3+0, 3+1, 3+3
    assert predicting_positions(3, [0, 1, 1, 0, 1]) == [3, 4, 6]


def test_ches_matches_the_paper_formula_and_ranks_similar_pairs_higher():
    np = pytest.importorskip("numpy")
    a = np.array([1.0, 0.0])
    assert ches(a, np.array([1.0, 0.0])) == 0.0  # identical means
    assert ches(a, np.array([0.0, 1.0])) == -1.0  # orthogonal
    assert ches(a, np.array([-1.0, 0.0])) == -2.0  # opposite: least similar
    assert ches(a, np.array([0.9, 0.1])) > ches(a, np.array([0.1, 0.9]))


def cand(c, r, score, classes=("wrong content",)):
    return {"id": f"{c}-{r}", "metadata": {"chosen_id": c, "rejected_id": r, "ches": score,
                                           "rejected_failure_classes": {k: 1 for k in classes}}}


def test_pick_covers_classes_first_then_lowest_ches_within_caps():
    cands = [cand("c0", f"r{i}", -float(i)) for i in range(5)]  # r4 lowest CHES
    cands.append(cand("c1", "rF", 5.0, ("forbidden action",)))  # most similar but the only forbidden one
    cands += [cand("c1", f"r{i}", -10.0 - i) for i in range(2)]
    got = pick(cands)
    ids = [p["id"] for p in got]
    assert "c1-rF" in ids  # class coverage beats CHES
    assert len(got) == 4
    uses = {}
    for p in got:
        for k in ("chosen_id", "rejected_id"):
            uses[p["metadata"][k]] = uses.get(p["metadata"][k], 0) + 1
    assert max(uses.values()) <= 2
    # forbidden covered by c1-rF, wrong content by the lowest c1-r1; c1 is then at its cap of 2, so c1-r0 is skipped
    assert ids == ["c1-rF", "c1-r1", "c0-r4", "c0-r3"]


def test_mean_hidden_equals_the_models_last_hidden_states_at_predicting_positions():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from ches_rank import mean_hidden

    config = transformers.Qwen3_5Config(
        text_config={"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 4,
                     "layer_types": ["linear_attention"] * 3 + ["full_attention"], "num_attention_heads": 2,
                     "num_key_value_heads": 1, "head_dim": 16, "linear_num_key_heads": 2, "linear_num_value_heads": 2,
                     "linear_key_head_dim": 8, "linear_value_head_dim": 8, "vocab_size": 97, "eos_token_id": 1},
        vision_config={"depth": 1, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2, "out_hidden_size": 32})
    torch.manual_seed(0)
    model = transformers.Qwen3_5ForConditionalGeneration(config).float().eval()
    prompt, ids, mask = [5, 6, 7], [8, 9, 10, 11, 12], [0, 1, 1, 0, 1]
    got = mean_hidden(model, prompt, ids, mask, device="cpu")
    with torch.no_grad():
        out = model(input_ids=torch.tensor([prompt + ids]), output_hidden_states=True)
    want = out.hidden_states[-1][0, [3, 4, 6]].mean(dim=0)
    assert torch.allclose(got, want, atol=1e-5)


def test_score_then_select_end_to_end_on_a_tiny_qwen35(tmp_path):
    """score loads base + adapter (merged), writes one vector per distinct side; select picks within the caps."""
    import json

    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    from ches_rank import score, select

    config = transformers.Qwen3_5Config(
        text_config={"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 4,
                     "layer_types": ["linear_attention"] * 3 + ["full_attention"], "num_attention_heads": 2,
                     "num_key_value_heads": 1, "head_dim": 16, "linear_num_key_heads": 2, "linear_num_value_heads": 2,
                     "linear_key_head_dim": 8, "linear_value_head_dim": 8, "vocab_size": 97, "eos_token_id": 1},
        vision_config={"depth": 1, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2, "out_hidden_size": 32})
    torch.manual_seed(0)
    transformers.Qwen3_5ForConditionalGeneration(config).save_pretrained(tmp_path / "base")
    torch.manual_seed(1)
    lora = peft.LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"], init_lora_weights=False)
    peft.get_peft_model(transformers.Qwen3_5ForConditionalGeneration(config), lora).save_pretrained(tmp_path / "ad")

    def row(c, r, seed):
        g = torch.Generator().manual_seed(seed)
        ids = lambda n: torch.randint(2, 97, (n,), generator=g).tolist()
        return {"id": f"p-{c}-{r}", "metadata": {
            "task": "t", "type": "A", "split": "train", "chosen_id": c, "rejected_id": r,
            "rejected_failure_classes": {"wrong content": 1},
            "token_ids": {"prompt": [5, 6, 7], "chosen": ids(6), "chosen_mask": [1, 1, 0, 0, 1, 1],
                          "rejected": ids(5), "rejected_mask": [1, 0, 1, 1, 1]}}}

    rows = [row(f"c{i}", f"r{j}", 10 * i + j) for i in range(2) for j in range(3)]
    cand = tmp_path / "cand.jsonl"
    cand.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert score(str(cand), str(tmp_path / "base"), str(tmp_path / "ad"), str(tmp_path / "ches")) == 0
    assert len(json.load(open(tmp_path / "ches.json"))["keys"]) == 12  # 6 pairs x 2 distinct sides
    assert select(str(cand), str(tmp_path / "ches"), str(tmp_path / "out.jsonl"), str(tmp_path / "rep.json")) == 0
    picked = [json.loads(line) for line in open(tmp_path / "out.jsonl")]
    assert len(picked) == 4 and all("ches" in p["metadata"] for p in picked)


def test_scoring_refuses_an_adapter_that_does_not_attach(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    from ches_rank import require_attached

    config = transformers.Qwen3_5TextConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                            layer_types=["linear_attention", "full_attention"], num_attention_heads=2,
                                            num_key_value_heads=1, head_dim=16, linear_num_key_heads=2,
                                            linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
                                            vocab_size=97, eos_token_id=1)
    torch.manual_seed(1)
    lora = peft.LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"], init_lora_weights=False)
    trained = peft.get_peft_model(transformers.Qwen3_5ForCausalLM(config), lora)
    trained.save_pretrained(tmp_path / "ad")
    require_attached(peft.PeftModel.from_pretrained(transformers.Qwen3_5ForCausalLM(config), tmp_path / "ad"),
                     tmp_path / "ad")
    torch.manual_seed(2)
    other = peft.get_peft_model(transformers.Qwen3_5ForCausalLM(config), lora)  # same names, other weights
    with pytest.raises(ValueError, match="did not attach"):
        require_attached(other, tmp_path / "ad")


def test_pick_prefers_the_attempt_covering_most_classes_when_one_chosen_limits_pairs():
    """One chosen attempt (2 uses): three classes are only coverable if the multi-class attempt is taken."""
    cands = [cand("c0", "rF", -9.0, ("forbidden action",)), cand("c0", "rM", -8.0, ("missed action",)),
             cand("c0", "rFMW", 1.0, ("forbidden action", "missed action", "wrong content")),
             cand("c0", "rW", -7.0, ("wrong content",))]
    got = pick(cands)
    covered = set().union(*[set(p["metadata"]["rejected_failure_classes"]) for p in got])
    assert covered == {"forbidden action", "missed action", "wrong content"} and len(got) == 2
