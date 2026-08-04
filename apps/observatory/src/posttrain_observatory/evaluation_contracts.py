"""Versioned readers for resolved Verifiers evaluation contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from posttrain.common import JsonValue

type EvaluationContractState = Literal["versioned", "legacy", "unsupported"]


@dataclass(frozen=True, slots=True)
class NormalizedEvaluationContract:
    """The schema-neutral contract consumed by Observatory projections."""

    state: EvaluationContractState
    contract_id: str | None
    contract_version: int | None
    plan: Mapping[str, JsonValue]
    environment: Mapping[str, JsonValue]
    signal_manifest: Mapping[str, JsonValue]
    native_evidence: Mapping[str, JsonValue]
    population: Mapping[str, JsonValue]
    raw: Mapping[str, JsonValue]


class EvaluationContractReader:
    contract_id = "posttrain.eval.verifiers-observation"
    schema_version = 1

    def decode(self, payload: Mapping[str, JsonValue]) -> NormalizedEvaluationContract:
        contract = payload.get("contract")
        if not isinstance(contract, Mapping):
            raise ValueError("evaluation contract envelope is missing contract metadata")
        if contract.get("id") != self.contract_id or contract.get("schema_version") != self.schema_version:
            raise ValueError("evaluation contract version does not match the v1 reader")
        return NormalizedEvaluationContract(
            state="versioned",
            contract_id=self.contract_id,
            contract_version=self.schema_version,
            plan=_mapping(payload.get("plan")),
            environment=_mapping(payload.get("environment")),
            signal_manifest=_mapping(payload.get("signal_manifest")),
            native_evidence=_mapping(payload.get("native_evidence")),
            population=_mapping(payload.get("population")),
            raw=payload,
        )


class EvaluationContractV2Reader(EvaluationContractReader):
    """Reader for run-owned typed success definitions."""

    schema_version = 2


class EvaluationContractV3Reader(EvaluationContractReader):
    """Reader for run-owned success definitions and compound breakdowns."""

    schema_version = 3


_READERS: dict[tuple[str, int], EvaluationContractReader] = {
    (EvaluationContractReader.contract_id, EvaluationContractReader.schema_version): EvaluationContractReader(),
    (EvaluationContractV2Reader.contract_id, EvaluationContractV2Reader.schema_version): EvaluationContractV2Reader(),
    (EvaluationContractV3Reader.contract_id, EvaluationContractV3Reader.schema_version): EvaluationContractV3Reader(),
}


def read_evaluation_contract(inputs: Mapping[str, JsonValue]) -> NormalizedEvaluationContract:
    """Decode the run-snapshotted contract without consulting the catalog."""

    payload = inputs.get("evaluation")
    if not isinstance(payload, Mapping):
        return NormalizedEvaluationContract(
            state="legacy",
            contract_id=None,
            contract_version=None,
            plan={},
            environment={},
            signal_manifest={},
            native_evidence={},
            population={},
            raw={},
        )
    contract = payload.get("contract")
    if not isinstance(contract, Mapping):
        return NormalizedEvaluationContract(
            state="unsupported",
            contract_id=None,
            contract_version=None,
            plan={},
            environment={},
            signal_manifest={},
            native_evidence={},
            population={},
            raw=payload,
        )
    contract_id = contract.get("id")
    contract_version = contract.get("schema_version")
    if not isinstance(contract_id, str) or not isinstance(contract_version, int) or isinstance(contract_version, bool):
        return NormalizedEvaluationContract(
            state="unsupported",
            contract_id=contract_id if isinstance(contract_id, str) else None,
            contract_version=contract_version if isinstance(contract_version, int) else None,
            plan={},
            environment={},
            signal_manifest={},
            native_evidence={},
            population={},
            raw=payload,
        )
    reader = _READERS.get((contract_id, contract_version))
    if reader is None:
        return NormalizedEvaluationContract(
            state="unsupported",
            contract_id=contract_id,
            contract_version=contract_version,
            plan={},
            environment={},
            signal_manifest={},
            native_evidence={},
            population={},
            raw=payload,
        )
    return reader.decode(payload)


def _mapping(value: object) -> Mapping[str, JsonValue]:
    return value if isinstance(value, Mapping) else {}


__all__ = ["EvaluationContractReader", "NormalizedEvaluationContract", "read_evaluation_contract"]
