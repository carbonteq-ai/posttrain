"""OCI Distribution manifest deletion for the purge lifecycle."""

from __future__ import annotations

import base64
import json
import os
import ssl
import subprocess
import urllib.error
import urllib.request
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from posttrain.common import ContractError
from posttrain.execution import (
    RegistryManifestDeletePlan,
    RegistryManifestDeleteReceipt,
    RegistryManifestRef,
)

_ACCEPT = ", ".join(
    (
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
    )
)


class RegistryTransport(Protocol):
    def head(self, reference: RegistryManifestRef) -> bool: ...

    def delete(self, reference: RegistryManifestRef) -> bool: ...


class DistributionRegistryLifecycleAdmin:
    """Preview and delete exact OCI manifests through Distribution HTTP."""

    def __init__(self, transport: RegistryTransport) -> None:
        self._transport = transport

    def plan_manifest_delete(self, reference: RegistryManifestRef) -> RegistryManifestDeletePlan:
        exists = self._transport.head(reference)
        return RegistryManifestDeletePlan.build(reference, exists=exists)

    def delete_manifest(self, plan: RegistryManifestDeletePlan) -> RegistryManifestDeleteReceipt:
        if plan.blockers:
            raise ContractError("registry manifest deletion is blocked: " + "; ".join(plan.blockers))
        if plan.digest != RegistryManifestDeletePlan.build(plan.reference, exists=plan.exists).digest:
            raise ContractError("registry manifest delete plan is stale")
        exists_now = self._transport.head(plan.reference)
        if exists_now != plan.exists:
            raise ContractError("registry manifest presence changed; obtain a new preview")
        deleted = self._transport.delete(plan.reference) if exists_now else False
        return RegistryManifestDeleteReceipt(
            reference=plan.reference,
            plan_digest=plan.digest,
            deleted=deleted,
            completed_at=datetime.now(UTC),
        )


class RegistryPurgeActionExecutor:
    """Bridge framework registry actions to digest-bound Distribution calls."""

    def __init__(self, admin: DistributionRegistryLifecycleAdmin) -> None:
        self._admin = admin
        self._plans: dict[str, RegistryManifestDeletePlan] = {}

    def revalidate(self, action: object) -> None:
        target = getattr(action, "target", {})
        action_id = getattr(action, "action_id", None)
        reference = target.get("reference") if isinstance(target, Mapping) else None
        if not isinstance(action_id, str) or not isinstance(reference, str):
            raise ContractError("registry purge action must name an exact reference")
        plan = self._admin.plan_manifest_delete(RegistryManifestRef.parse(reference))
        if plan.blockers:
            raise ContractError("registry purge is blocked: " + "; ".join(plan.blockers))
        self._plans[action_id] = plan

    def apply(self, action: object) -> None:
        action_id = getattr(action, "action_id", None)
        if not isinstance(action_id, str) or action_id not in self._plans:
            raise ContractError("registry purge action was not revalidated")
        self._admin.delete_manifest(self._plans.pop(action_id))


class UrllibDistributionTransport:
    """Small credential-aware OCI Distribution HTTP transport.

    Credentials are read from Docker config or a configured credential helper
    for the request and are never written by this adapter.
    """

    def __init__(
        self,
        *,
        scheme: str = "https",
        trust_bundle: Path | None = None,
        docker_config: Path | None = None,
    ) -> None:
        if scheme not in {"http", "https"}:
            raise ContractError("registry scheme must be http or https")
        self._scheme = scheme
        self._context = ssl.create_default_context(cafile=str(trust_bundle) if trust_bundle else None)
        self._docker_config = docker_config or self._default_docker_config()

    def head(self, reference: RegistryManifestRef) -> bool:
        return self._request("HEAD", reference) is not None

    def delete(self, reference: RegistryManifestRef) -> bool:
        response = self._request("DELETE", reference)
        return response is not None

    def _request(self, method: str, reference: RegistryManifestRef) -> bytes | None:
        host, repository = self._split(reference)
        url = f"{self._scheme}://{host}/v2/{repository}/manifests/{reference.digest}"
        request = urllib.request.Request(
            url,
            method=method,
            headers={"Accept": _ACCEPT, **self._authorization(host)},
        )
        try:
            with urllib.request.urlopen(request, context=self._context, timeout=30) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            raise ContractError(f"registry {method} failed for {reference.value}: HTTP {error.code}") from error
        except (urllib.error.URLError, TimeoutError) as error:
            raise ContractError(f"registry {method} could not reach {host}") from error

    @staticmethod
    def _split(reference: RegistryManifestRef) -> tuple[str, str]:
        parts = reference.repository.split("/", 1)
        if len(parts) != 2:
            raise ContractError("OCI deletion requires a fully qualified registry/repository")
        return parts[0], parts[1]

    @staticmethod
    def _default_docker_config() -> Path:
        configured = os.environ.get("DOCKER_CONFIG")
        return (Path(configured) / "config.json") if configured else (Path.home() / ".docker/config.json")

    def _authorization(self, host: str) -> Mapping[str, str]:
        if not self._docker_config.is_file():
            return {}
        try:
            payload = json.loads(self._docker_config.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ContractError(f"Docker credential config is invalid: {self._docker_config}") from error
        if not isinstance(payload, dict):
            raise ContractError("Docker credential config must be an object")
        helper = payload.get("credHelpers", {}).get(host) if isinstance(payload.get("credHelpers"), dict) else None
        if helper is None:
            helper = payload.get("credsStore")
        if isinstance(helper, str) and helper:
            if not helper.replace("-", "").isalnum():
                raise ContractError("Docker credential helper name is invalid")
            try:
                result = subprocess.run(
                    [f"docker-credential-{helper}", "get"],
                    input=(host + "\n").encode(),
                    capture_output=True,
                    check=True,
                    timeout=10,
                )
                credentials = json.loads(result.stdout.decode())
            except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
                raise ContractError(f"Docker credential helper failed for {host}") from error
            if isinstance(credentials, dict):
                username = credentials.get("Username")
                secret = credentials.get("Secret")
                if isinstance(username, str) and isinstance(secret, str):
                    return {"Authorization": "Basic " + _basic(username, secret)}
            return {}
        auths = payload.get("auths")
        if not isinstance(auths, dict):
            return {}
        entry = auths.get(host) or auths.get(f"https://{host}")
        if not isinstance(entry, dict) or not isinstance(entry.get("auth"), str):
            return {}
        return {"Authorization": "Basic " + entry["auth"]}


def _basic(username: str, secret: str) -> str:
    return base64.b64encode(f"{username}:{secret}".encode()).decode()


__all__ = [
    "DistributionRegistryLifecycleAdmin",
    "RegistryPurgeActionExecutor",
    "RegistryTransport",
    "UrllibDistributionTransport",
]
