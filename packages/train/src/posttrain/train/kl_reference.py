"""Which policy the online-RL KL penalty measures distance from."""

from __future__ import annotations

from typing import Literal

type KLReference = Literal["off", "base", "start"]


def resolved_kl_reference(beta: float, kl_reference: str | None, policy_form: str) -> KLReference:
    """Name the reference a run uses: "off" (beta 0), "base" (foundation), or "start".

    "start" is the checkpoint the run started from. A run that starts from the
    foundation model, including a fresh LoRA adapter (which starts at zero), has
    the base model as both, so the setting only matters when a run continues a
    trained adapter or full-parameter checkpoint. Settings without the field (GDPO,
    CAPO) keep TRL's default, a frozen copy of the starting adapter.
    """

    if beta == 0:
        return "off"
    if policy_form == "foundation":
        return "base"
    return "base" if kl_reference == "base" else "start"


def kl_reference_problem(backend: str, beta: float, kl_reference: str | None, policy_form: str) -> str | None:
    """Explain a KL reference the selected training backend cannot provide.

    TRL continues a PEFT adapter with either reference (``peft_reference``) and loads
    no other non-foundation form for training. veRL computes a LoRA reference with
    the adapter disabled and otherwise loads the reference from the actor's own path,
    the starting checkpoint; Posttrain passes it no adapter path, so it cannot
    continue a PEFT adapter at all.
    """

    if backend.split("@", 1)[0] != "verl":
        return None
    if policy_form in {"adapter", "peft-adapter"}:
        return (
            "the veRL backend cannot continue a PEFT adapter (it would load the adapter directory as the model); "
            "select the TRL backend"
        )
    if beta > 0 and kl_reference == "base" and policy_form != "foundation":
        return (
            f"veRL uses the starting checkpoint as the KL reference, so kl_reference: base cannot hold for a "
            f"{policy_form} starting model; set kl_reference: start or select the TRL backend"
        )
    return None


def describe_kl_reference(beta: float, kl_reference: str | None) -> str:
    """One plan line; the starting model is only known once a run selects it."""

    if beta == 0:
        return "KL penalty: off (beta 0)"
    if kl_reference == "base":
        return f"KL reference: base model (kl_reference: base, beta {beta:g})"
    return (
        f"KL reference: starting checkpoint (kl_reference: start, beta {beta:g}); "
        "the base model when the run starts from the foundation model or a fresh adapter"
    )
