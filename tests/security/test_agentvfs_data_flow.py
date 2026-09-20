"""Agentvfs commands enforce the same data-flow boundary as other providers."""

from __future__ import annotations

import socket
from typing import Any

import pytest

from agent_libos import Runtime
from agent_libos.models import DataFlowContext, EventType, ObjectMetadata, ObjectType, SinkTrustRule
from agent_libos.models.exceptions import CapabilityDenied
from tests.security.test_agentvfs_authority import agentvfs_runtime
from tests.security.test_agentvfs_module import COMMIT_B
from tests.support.public_errors import assert_public_error_message


_SECRET = "AGENTVFS_SECRET_SENTINEL"
pytestmark = pytest.mark.skipif(
    not hasattr(socket, "AF_UNIX"), reason="agentvfs control requires Unix sockets"
)


def _secret_context(runtime: Runtime, pid: str) -> DataFlowContext:
    source = runtime.memory.create_object(
        pid,
        ObjectType.EVIDENCE,
        {"value": _SECRET},
        metadata=ObjectMetadata(sensitivity="secret"),
    )
    return runtime.data_flow.context_from_source_oids(pid, [source.oid])


@pytest.mark.parametrize(
    ("tool", "right", "args"),
    [
        ("agentvfs_status", "read", {}),
        ("agentvfs_checkpoint", "write", {"label": _SECRET}),
        ("agentvfs_rollback", "admin", {"target": _SECRET}),
        ("agentvfs_checkpoint", "write", {"label": _SECRET, "pair_libos": True}),
        ("agentvfs_rollback", "admin", {"target": _SECRET, "pair_libos": True}),
    ],
    ids=["status", "checkpoint", "rollback", "paired-checkpoint", "paired-rollback"],
)
def test_secret_commands_deny_before_socket_and_preserve_authority(
    agentvfs_runtime: Any, tool: str, right: str, args: dict[str, Any]
) -> None:
    runtime, pid, harness = agentvfs_runtime
    args = dict(args)
    if tool == "agentvfs_rollback" and args.get("pair_libos"):
        args["libos_checkpoint_id"] = runtime.checkpoint.create(
            pid,
            "paired fixture",
            actor=pid,
            metadata={
                "agentvfs_workspace": harness._workspace,
                "agentvfs_label": _SECRET,
                "agentvfs_commit": COMMIT_B,
            },
        )
    resource = f"agentvfs:{harness._workspace}"
    capability = runtime.capability.issue_trusted(
        subject=pid, resource=resource, rights=[right], issued_by="test", uses_remaining=1
    )
    context = _secret_context(runtime, pid)

    result = runtime.tools.call(
        pid, tool, args, context_metadata={"data_flow_context": context}
    )

    assert not result.ok
    assert_public_error_message(
        result.error, code="permission_denied", error_type="DataFlowDenied"
    )
    assert harness._fake.received == []
    assert runtime.store.get_capability(capability.cap_id).uses_remaining == 1
    assert runtime.store.list_external_effects(pid=pid) == []
    denied = runtime.store.list_data_flow_decisions(pid=pid, outcome="deny")
    assert len(denied) == 1
    assert denied[0].sink == resource
    assert denied[0].labels.sensitivity.value == "secret"
    audit = next(
        record for record in runtime.audit.trace()
        if record.action == "data_flow.egress"
        and record.decision.get("decision_id") == denied[0].decision_id
    )
    event = next(
        event for event in runtime.events.list(target=f"data_flow_sink:{resource}")
        if event.type == EventType.DATA_FLOW_DECISION
        and event.payload.get("decision_id") == denied[0].decision_id
    )
    assert audit.decision["outcome"] == event.payload["outcome"] == "deny"
    assert _SECRET not in str(audit.decision)
    assert _SECRET not in str(event.payload)


@pytest.mark.parametrize("operation", ["checkpoint", "rollback"])
def test_direct_adapter_commands_enforce_ambient_data_labels(
    agentvfs_runtime: Any, operation: str
) -> None:
    runtime, pid, harness = agentvfs_runtime
    runtime.capability.grant(
        pid, f"agentvfs:{harness._workspace}", ["write", "admin"], issued_by="test"
    )
    adapter = runtime.module_state.get("_agent_libos_agentvfs_adapter")
    with runtime.data_flow.activate(_secret_context(runtime, pid)):
        with pytest.raises(CapabilityDenied, match="data-flow denied egress"):
            getattr(adapter, operation)(pid, _SECRET)
    assert harness._fake.received == []


@pytest.mark.parametrize(
    ("tool", "right", "args", "command"),
    [
        ("agentvfs_status", "read", {}, "status"),
        ("agentvfs_checkpoint", "write", {"label": _SECRET}, f"checkpoint {_SECRET}"),
        ("agentvfs_rollback", "admin", {"target": _SECRET}, f"rollback {_SECRET}"),
    ],
    ids=["status", "checkpoint", "rollback"],
)
def test_host_clearance_allows_secret_commands_and_preserves_ingress_labels(
    agentvfs_runtime: Any, tool: str, right: str, args: dict[str, Any], command: str
) -> None:
    runtime, pid, harness = agentvfs_runtime
    resource = f"agentvfs:{harness._workspace}"
    runtime.capability.grant(
        pid, runtime.config.data_flow.registry_resource, ["admin"], issued_by="test"
    )
    runtime.register_sink_trust(
        SinkTrustRule(pattern=resource, trust_level="trusted", max_sensitivity="secret"),
        actor=pid,
    )
    context = _secret_context(runtime, pid)
    denied = runtime.tools.call(
        pid, tool, args, context_metadata={"data_flow_context": context}
    )
    assert not denied.ok
    assert harness._fake.received == []
    capability = runtime.capability.issue_trusted(
        subject=pid, resource=resource, rights=[right], issued_by="test", uses_remaining=1
    )

    result = runtime.tools.call(
        pid, tool, args, context_metadata={"data_flow_context": context}
    )

    assert result.ok, result.error
    assert harness._fake.received == [command]
    assert runtime.store.get_capability(capability.cap_id).uses_remaining == 0
    assert result.result_handle is not None
    stored = runtime.store.get_object(result.result_handle.oid)
    assert stored.metadata.sensitivity == "secret"
    assert stored.metadata.integrity == "untrusted"
    assert stored.metadata.trust_level == "untrusted"
    effect = runtime.store.list_external_effects(pid=pid)[-1]
    assert effect.provider_metadata["data_flow"]["sink"] == resource
    assert effect.provider_metadata["data_flow"]["trust_id"] is not None
