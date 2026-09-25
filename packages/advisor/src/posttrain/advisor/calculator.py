"""Suggest vLLM engine settings from hardware, model architecture and task.

A pure calculator: no GPU, no engine start. It applies rules measured on the
CarbonTeq vLLM fork (``docs/plan/agentic-workload-inference-optimization.md``,
``docs/architecture/vllm-inference-optimization.md``) and a memory budget built
from the checkpoint's own architecture. Every suggested value carries its reason;
estimates are labelled as estimates, and anything the inputs cannot determine is
reported as unknown rather than guessed.

The configuration rules (``posttrain.advisor.rules``) check the same measured
settings: a binding that follows these suggestions produces no rule findings
about them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

GIB = 1024**3

Purpose = Literal["rollout", "eval", "screen", "judge"]


@dataclass(frozen=True, slots=True)
class AcceleratorFacts:
    memory_gb: float
    bandwidth_gb_s: float
    compute_capability: tuple[int, int]
    flash_attn_version: int
    source: str


# Vendor specifications; SM120 cannot use native FlashAttention 4 in vLLM.
ACCELERATORS: dict[str, AcceleratorFacts] = {
    "RTXPRO6000": AcceleratorFacts(96, 1792, (12, 0), 2, "NVIDIA RTX PRO 6000 Blackwell workstation specification"),
    "RTXPRO4500": AcceleratorFacts(32, 896, (12, 0), 2, "NVIDIA RTX PRO 4500 Blackwell specification"),
    "H100": AcceleratorFacts(80, 3350, (9, 0), 3, "NVIDIA H100 SXM specification"),
    "H200": AcceleratorFacts(141, 4800, (9, 0), 3, "NVIDIA H200 specification"),
    "RTX3070TI": AcceleratorFacts(8, 608, (8, 6), 2, "NVIDIA GeForce RTX 3070 Ti specification"),
}


@dataclass(frozen=True, slots=True)
class Hardware:
    memory_gb: float
    accelerator_model: str | None = None
    accelerator_count: int = 1
    supports_bf16: bool | None = None

    @property
    def facts(self) -> AcceleratorFacts | None:
        key = (self.accelerator_model or "").replace(" ", "").replace("-", "").upper()
        return ACCELERATORS.get(key)


@dataclass(frozen=True, slots=True)
class Architecture:
    """KV-relevant geometry read from a checkpoint's ``config.json``."""

    parameters: int
    hidden_size: int
    layers: int
    attention_layers: int
    kv_heads: int
    head_dim: int
    sliding_window_layers: int = 0
    sliding_window: int | None = None
    recurrent_layers: int = 0
    max_position_embeddings: int | None = None
    global_kv_heads: int | None = None
    """KV heads of the full-attention layers when they differ from the rest (Gemma 4)."""
    global_head_dim: int | None = None
    active_parameters: int | None = None
    """Parameters read per decoded token for mixture-of-experts models."""
    mtp_layers: int = 0
    """Native multi-token-prediction layers shipped in the checkpoint (Qwen3.5, DeepSeek)."""

    @property
    def decode_parameters(self) -> int:
        return self.active_parameters or self.parameters

    @classmethod
    def from_config(cls, config: Mapping[str, Any], parameters: int) -> Architecture:
        nested = config.get("text_config")
        text: Mapping[str, Any] = nested if isinstance(nested, Mapping) else config
        layers = int(text["num_hidden_layers"])
        heads = int(text["num_attention_heads"])
        kv_heads = int(text.get("num_key_value_heads") or heads)
        hidden = int(text["hidden_size"])
        head_dim = int(text.get("head_dim") or hidden // heads)
        types = [str(value) for value in text.get("layer_types") or ()]
        if not types and text.get("full_attn_idxs") is not None:
            full = set(text["full_attn_idxs"])
            types = ["full_attention" if index in full else "conv" for index in range(layers)]
        if not types and text.get("full_attention_interval"):
            # Qwen3-Next: linear-attention layers with full attention every Nth layer.
            interval = int(text["full_attention_interval"])
            types = ["full_attention" if (index + 1) % interval == 0 else "linear_attention" for index in range(layers)]
        if types:
            full = sum(kind in {"full_attention", "attention", "global_attention"} for kind in types)
            sliding = sum(
                kind in {"sliding_attention", "local_attention", "sliding_window_attention"} for kind in types
            )
            recurrent = len(types) - full - sliding
        else:
            full, sliding, recurrent = layers, 0, 0
        return cls(
            parameters=parameters,
            hidden_size=hidden,
            layers=layers,
            attention_layers=full,
            kv_heads=kv_heads,
            head_dim=head_dim,
            sliding_window_layers=sliding,
            sliding_window=text.get("sliding_window"),
            recurrent_layers=recurrent,
            max_position_embeddings=text.get("max_position_embeddings"),
            global_kv_heads=text.get("num_global_key_value_heads"),
            global_head_dim=text.get("global_head_dim"),
            active_parameters=_active_parameters(text, parameters),
            mtp_layers=int(text.get("mtp_num_hidden_layers") or text.get("num_nextn_predict_layers") or 0),
        )

    def kv_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        """Per-token KV bytes of the full-attention layers (keys and values).

        Keys and values are counted separately even where a model ties them
        (Gemma 4 ``attention_k_eq_v``): an overestimate reserves enough memory."""
        heads = self.global_kv_heads or self.kv_heads
        head_dim = self.global_head_dim or self.head_dim
        return 2 * self.attention_layers * heads * head_dim * dtype_bytes

    def windowed_kv_bytes_per_sequence(self, context: int, dtype_bytes: int = 2) -> int:
        """Per-sequence KV bytes of sliding-window layers, bounded by the window."""
        if not self.sliding_window_layers:
            return 0
        window = min(context, self.sliding_window or context)
        return 2 * self.sliding_window_layers * self.kv_heads * self.head_dim * dtype_bytes * window


@dataclass(frozen=True, slots=True)
class Task:
    purpose: Purpose
    concurrency: int
    prompt_tokens: int
    completion_tokens: int
    temperature: float = 1.0
    colocated_trainer_gb: float | None = None
    """Device memory the colocated trainer keeps while rollouts run; None for a dedicated engine."""
    co_tenant_gb: float = 0.0
    """Device memory other engines of the same job reserve on this target (for example a judge beside rollouts)."""
    reproducible_logprobs: bool = False
    lora_rank: int | None = None
    kv_cache_dtype: str = "auto"
    """TurboQuant KV (``turboquant_*``) requires float16 compute."""
    speculative_tokens: int = 0
    """Draft tokens proposed per step; each step verifies this many plus one per sequence."""
    draft: Architecture | None = None
    """A separate draft model (MTP assistant, DSpark/DFlash block drafter); None for native MTP heads."""
    step_sequences: int | None = None
    """Rollout rows one training step requests (prompts per step x generations)."""
    group_size: int | None = None
    """Generations per prompt; batch advice moves in whole groups."""
    oversampling: bool = False
    """The step refills zero-variance groups (active or dynamic sampling)."""
    typical_context_fraction: float = 0.35
    """Average live context as a fraction of the maximum. Measured: AutomationBench
    episodes averaged about 35% of their 24,576-token budget, and a KV cache sized
    for that fraction recomputed only 0.1% of reusable context at 32 concurrent
    episodes (docs/plan/agentic-workload-inference-optimization.md)."""


@dataclass(frozen=True, slots=True)
class Setting:
    key: str
    value: Any
    reason: str


@dataclass(frozen=True, slots=True)
class Suggestion:
    engine: dict[str, Any]
    environment: dict[str, str]
    settings: tuple[Setting, ...]
    memory_gb: dict[str, float]
    max_concurrency: int | None
    decode_tokens_per_s_upper_bound: float | None
    notes: tuple[str, ...] = field(default=())
    step: StepCapacity | None = None


# Oversampling takes at most this share of a step's rows; the rest of any margin
# goes to more prompts per step.
OVERSAMPLE_SHARE = 0.2
# Useful concurrency stops where memory-bound decode reaches this share of its
# asymptotic throughput; past it, more sequences mostly lengthen the step.
THROUGHPUT_KNEE = 0.8


@dataclass(frozen=True, slots=True)
class StepOption:
    """One way to size a step, with its collection shape and relative cost."""

    label: str
    prompts_per_step: int
    oversample_groups: int
    rows: int
    waves: int
    relative_step_time: float
    """Decode-bound collection time relative to the current step (1.0)."""
    relative_rows_per_second: float
    """Rollout rows collected per second relative to the current step."""


@dataclass(frozen=True, slots=True)
class StepCapacity:
    """How a training step's rollout rows compare with what the engine can serve at once."""

    step_sequences: int
    fits: int
    useful: int
    waves: int
    margin_sequences: int
    oversample_groups: int
    extra_prompts_per_step: int
    recommended_prompts_per_step: int
    recommended_in_flight: int
    reason: str
    options: tuple[StepOption, ...] = ()


_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)


def suggest(hardware: Hardware, architecture: Architecture, task: Task, *, weight_bytes: int = 2) -> Suggestion:
    """Return engine settings, their reasons, and the memory budget behind them."""

    settings: list[Setting] = []
    notes: list[str] = []
    facts = hardware.facts
    context = _round_up(task.prompt_tokens + task.completion_tokens, 256)
    native = architecture.max_position_embeddings
    if native is not None and context > native:
        notes.append(
            f"prompt + completion ({task.prompt_tokens + task.completion_tokens}) exceeds the model's "
            f"{native}-token context; episodes past it cannot run"
        )
        context = native
    settings.append(
        Setting(
            "max_model_len",
            context,
            "prompt plus completion budget, rounded to 256; shorter contexts fail long episodes outright (4% of LFM2.5 AutomationBench episodes at 24,576)",
        )
    )

    bf16 = hardware.supports_bf16 is not False and (facts is None or facts.compute_capability >= (8, 0))
    turboquant = task.kv_cache_dtype.startswith("turboquant_")
    if turboquant:
        settings.append(Setting("dtype", "float16", "TurboQuant KV cache requires float16 compute"))
        settings.append(Setting("kv_cache_dtype", task.kv_cache_dtype, "the selected TurboQuant KV format"))
    else:
        dtype = "bfloat16" if bf16 else "float16"
        settings.append(
            Setting(
                "dtype",
                dtype,
                "bf16 checkpoints must compute in bf16 where the GPU supports it; float16's narrower exponent can overflow and shifts logprobs away from training",
            )
        )

    # Memory budget, per device.
    devices = max(hardware.accelerator_count, 1)
    weights = architecture.parameters * weight_bytes / GIB / devices
    kv_token = architecture.kv_bytes_per_token() / devices
    window = architecture.windowed_kv_bytes_per_sequence(context) / devices
    batched_tokens = 8192 if hardware.memory_gb <= 12 else 16384
    # Speculative decoding: a separate drafter adds its weights and its own KV per
    # token; native MTP heads add one attention layer's KV per MTP layer.
    draft_weights = 0.0
    draft_kv_token = 0.0
    draft_window = 0.0
    if task.speculative_tokens:
        if task.draft is not None:
            draft_weights = task.draft.parameters * weight_bytes / GIB / devices
            draft_kv_token = task.draft.kv_bytes_per_token() / devices
            draft_window = task.draft.windowed_kv_bytes_per_sequence(context) / devices
        elif architecture.mtp_layers:
            draft_kv_token = 2 * architecture.mtp_layers * architecture.kv_heads * architecture.head_dim * 2 / devices
        target_kv_token = kv_token
        kv_token += draft_kv_token
        window += draft_window
        verified = task.concurrency * (task.speculative_tokens + 1)
        batched_tokens = max(batched_tokens, _round_up(verified, 256))
        notes.append(
            f"speculative decoding verifies {task.speculative_tokens + 1} tokens per sequence per step"
            + (
                f"; the drafter adds {draft_weights:.2f} GB of weights and {draft_kv_token / 1024:.1f} KiB of KV per "
                f"token ({draft_kv_token / target_kv_token:.0%} of the target's)"
                if task.draft is not None and target_kv_token
                else f"; native MTP heads add {draft_kv_token / 1024:.1f} KiB of KV per token"
                if draft_kv_token
                else ""
            )
        )
    # CUDA-graph pools and activation workspace scale with model width and the
    # batched-token budget; this is an estimate to verify at engine start.
    workspace = max(0.5, batched_tokens * architecture.hidden_size * 2 * 12 / GIB)
    graphs = max(0.25, min(3.0, 0.04 * weights + 0.02 * task.concurrency))  # sized for the requested demand
    trainer = task.colocated_trainer_gb or 0.0
    others = task.co_tenant_gb
    shared = trainer + others
    reserve = 0.06 * hardware.memory_gb if shared == 0 else 1.0
    available_for_kv = hardware.memory_gb - shared - weights - draft_weights - workspace - graphs - reserve
    typical = task.typical_context_fraction * context
    per_sequence = kv_token * typical + window
    wanted_kv = task.concurrency * per_sequence / GIB
    worst_kv = task.concurrency * (kv_token * context + window) / GIB

    eager = False
    if available_for_kv <= 0:
        eager_available = available_for_kv + graphs
        notes.append(
            f"weights, workspace{' and the trainer' if trainer else ''} leave no room for KV cache on "
            f"{hardware.memory_gb:g} GB; choose a larger target or a smaller model"
        )
        eager = eager_available > 0
        available_for_kv = max(eager_available, 0)
    settings.append(
        Setting(
            "enforce_eager",
            eager,
            (
                "CUDA graphs stay on: eager decode was 2.32x slower on the LFM2.5 rollout replay"
                if not eager
                else "the CUDA-graph pool does not fit beside the trainer; acknowledge VLLM_EAGER_DISABLES_CUDA_GRAPHS in the binding with this budget"
            ),
        )
    )
    kv_gb = max(min(wanted_kv, available_for_kv), 0.0)
    max_concurrency = int(available_for_kv * GIB // per_sequence) if per_sequence > 0 and available_for_kv > 0 else None
    # Sequences in flight: the demand, bounded by what fits. A larger batch than fits
    # is collected in waves rather than failing or thrashing the KV cache.
    in_flight = max(1, min(task.concurrency, max_concurrency)) if max_concurrency else task.concurrency
    if max_concurrency is not None and max_concurrency < task.concurrency:
        notes.append(
            f"about {max_concurrency} sequences of typical length fit in the KV budget; {task.concurrency} requested "
            f"sequences are collected in {math.ceil(task.concurrency / max_concurrency)} waves"
        )
        wanted_kv = in_flight * per_sequence / GIB
        worst_kv = in_flight * (kv_token * context + window) / GIB
    kv_bytes = int(math.ceil(kv_gb * GIB / (256 * 1024**2)) * 256 * 1024**2) if kv_gb > 0 else 0

    settings.append(
        Setting(
            "max_num_seqs",
            in_flight,
            "the requested concurrency"
            + (
                f", capped at the {max_concurrency} typical-length sequences that fit"
                if in_flight < task.concurrency
                else ""
            )
            + "; a lower cap serializes requests",
        )
    )
    settings.append(
        Setting(
            "max_num_batched_tokens",
            batched_tokens,
            "long agentic prompts prefill in few chunks at 16,384 (no gain measured at 32,768); 8,192 on GPUs of 12 GB or less",
        )
    )
    settings.append(
        Setting(
            "enable_prefix_caching",
            True,
            "multi-turn and grouped prompts share context; 80-89% of prompt tokens were cache hits on the LFM2.5 replay; policy updates reset it",
        )
    )
    settings.append(Setting("enable_chunked_prefill", True, "keeps decode running while long prompts prefill"))
    if facts is not None:
        settings.append(
            Setting(
                "flash_attn_version",
                facts.flash_attn_version,
                "SM120 supports FlashAttention 2 in native vLLM (FA4 is rejected)"
                if facts.compute_capability[0] == 12
                else f"best native FlashAttention for compute capability {facts.compute_capability[0]}.{facts.compute_capability[1]}",
            )
        )
    if kv_bytes and shared:
        # Beside a trainer the KV cache must be pinned; a dedicated engine sizes it
        # from its memory fraction and would waste the rest if pinned.
        settings.append(
            Setting(
                "kv_cache_memory_bytes",
                kv_bytes,
                f"{in_flight} sequences x {typical:,.0f} typical tokens x {kv_token:,.0f} bytes/token"
                + (" + sliding-window state" if window else "")
                + f" = {kv_gb:.2f} GiB (worst case {worst_kv:.2f} GiB)",
            )
        )
    used = weights + draft_weights + workspace + graphs + kv_gb + (0.5 if shared else 0.0)
    if shared:
        fraction = min(0.95, math.ceil((used / hardware.memory_gb) * 100) / 100)
        beside = " and ".join(
            part
            for part in (
                f"a colocated trainer holding {trainer:g} GB" if trainer else "",
                f"other engines of this job reserving {others:g} GB" if others else "",
            )
            if part
        )
        reason = f"engine share beside {beside}: weights + KV + graphs + workspace = {used:.1f} GB"
    else:
        fraction = 0.9
        reason = "dedicated engine: reserve 90% of the device and let vLLM size the remaining KV"
    settings.append(Setting("gpu_memory_utilization", fraction, reason))
    if task.purpose == "rollout":
        settings.append(
            Setting(
                "mode",
                "colocate" if trainer else "server",
                "colocated with the trainer" if trainer else "dedicated rollout server",
            )
        )
        if trainer:
            settings.append(
                Setting(
                    "sleep_during_optimization", True, "release rollout memory to the optimizer between collections"
                )
            )
    if task.lora_rank:
        rank = next((value for value in _LORA_RANKS if value >= task.lora_rank), task.lora_rank)
        settings.append(Setting("max_lora_rank", rank, "vLLM supports these adapter ranks; the trained rank rounds up"))

    environment: dict[str, str] = {}
    if task.reproducible_logprobs:
        environment["VLLM_BATCH_INVARIANT"] = "1"
        notes.append(
            "batch invariance costs 2-6% decode throughput on the CarbonTeq fork (Gemma-4-12B 2.0%, E4B 5.9% at c4)"
        )

    engine = {item.key: item.value for item in settings}
    memory = {
        "weights": round(weights, 2),
        **({"draft_weights": round(draft_weights, 2)} if draft_weights else {}),
        "kv_cache": round(kv_gb, 2),
        "cuda_graphs_estimate": 0.0 if eager else round(graphs, 2),
        "workspace_estimate": round(workspace, 2),
        "trainer": round(trainer, 2),
        **({"other_engines": round(others, 2)} if others else {}),
        "free": round(
            hardware.memory_gb - shared - weights - draft_weights - workspace - (0 if eager else graphs) - kv_gb, 2
        ),
    }
    bound = None
    read_weights = architecture.decode_parameters * weight_bytes / devices
    if facts is not None:
        # Memory-bound decode: each step reads the weights once and each sequence's KV.
        per_step_bytes = read_weights + in_flight * per_sequence
        bound = round(in_flight * facts.bandwidth_gb_s * 1e9 / per_step_bytes, 0)
    step = _step_capacity(task, max_concurrency, read_weights, per_sequence)
    return Suggestion(engine, environment, tuple(settings), memory, max_concurrency, bound, tuple(notes), step)


def _step_capacity(task: Task, fits: int | None, read_weights: float, per_sequence: float) -> StepCapacity | None:
    """Advise how a training step can use the engine's spare concurrency.

    Decode throughput at ``n`` sequences is about ``n / (W + n * k)`` times the
    memory bandwidth (``W`` weight bytes, ``k`` KV bytes per sequence), so it
    reaches ``THROUGHPUT_KNEE`` of its limit at ``n = knee / (1 - knee) * W / k``.
    Concurrency beyond that knee mostly lengthens the step, so the margin is the
    smaller of what fits and the knee, minus the step's rows. Oversampling takes
    whole groups up to ``OVERSAMPLE_SHARE`` of the resulting step; the rest
    becomes prompts. Options carry waves and decode-bound step time relative to
    the current step, so a user can trade throughput against step length.
    """

    step, group = task.step_sequences, task.group_size
    if not step or not group or not fits:
        return None
    knee = (
        math.ceil(THROUGHPUT_KNEE / (1 - THROUGHPUT_KNEE) * read_weights / per_sequence) if per_sequence > 0 else fits
    )
    useful = max(min(fits, knee), 1)
    margin = max(useful - step, 0)
    target = step + margin
    oversample_rows = (min(int(OVERSAMPLE_SHARE * target), margin) // group) * group if task.oversampling else 0
    extra_prompts = (margin - oversample_rows) // group
    prompts = step // group + extra_prompts
    waves = math.ceil(step / fits)
    if margin < group:
        reason = f"the step's {step} rows already use the engine's useful concurrency ({useful})" + (
            f" and collect in {waves} waves" if waves > 1 else ""
        )
    else:
        reason = (
            f"{useful} sequences can decode at once before throughput flattens ({fits} fit in memory); the step "
            f"uses {step}, leaving {margin}"
        )

    def cost(rows: int) -> float:
        # Each wave decodes min(rows, fits) sequences; per-token time is (W + n * k) / bandwidth.
        waves_needed = math.ceil(rows / fits)
        full, last = divmod(rows, fits)
        return (
            full * (read_weights + fits * per_sequence) + (read_weights + last * per_sequence if last else 0.0)
            if waves_needed
            else 0.0
        )

    base = cost(step)

    def option(label: str, prompt_count: int, oversample_groups: int) -> StepOption:
        rows = (prompt_count + oversample_groups) * group
        relative = cost(rows) / base if base > 0 else 1.0
        return StepOption(
            label,
            prompt_count,
            oversample_groups,
            rows,
            math.ceil(rows / fits),
            round(relative, 2),
            round(rows / step / relative, 2) if relative > 0 else 1.0,
        )

    options = [option("current", step // group, 0)]
    if margin >= group:
        options.append(option("recommended", prompts, oversample_rows // group))
    if waves > 1 and fits >= group:
        options.append(option("one wave", fits // group, 0))
    if fits > useful and fits // group > prompts:
        fill_oversample = int(OVERSAMPLE_SHARE * fits) // group if task.oversampling else 0
        options.append(option("fill memory", fits // group - fill_oversample, fill_oversample))
    return StepCapacity(
        step_sequences=step,
        fits=fits,
        useful=useful,
        waves=waves,
        margin_sequences=margin,
        oversample_groups=oversample_rows // group,
        extra_prompts_per_step=extra_prompts,
        recommended_prompts_per_step=prompts,
        recommended_in_flight=prompts * group + oversample_rows,
        reason=reason,
        options=tuple(options),
    )


def _active_parameters(text: Mapping[str, Any], parameters: int) -> int | None:
    """Parameters touched per token in a mixture-of-experts model.

    Each routed expert is a gated MLP (three hidden x intermediate matrices);
    only ``num_experts_per_tok`` of ``num_experts`` run per token.
    """
    experts = text.get("num_experts") or text.get("n_routed_experts")
    active = text.get("num_experts_per_tok")
    width = text.get("moe_intermediate_size")
    if not (experts and active and width):
        return None
    per_expert = 3 * int(text["hidden_size"]) * int(width)
    moe_layers = int(text["num_hidden_layers"])
    idle = (int(experts) - int(active)) * per_expert * moe_layers
    return max(parameters - idle, 0) or None


def _round_up(value: int, step: int) -> int:
    return int(math.ceil(value / step) * step)


__all__ = [
    "ACCELERATORS",
    "AcceleratorFacts",
    "Architecture",
    "Hardware",
    "Setting",
    "StepCapacity",
    "StepOption",
    "Suggestion",
    "Task",
    "suggest",
]
