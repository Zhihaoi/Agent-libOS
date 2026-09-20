"""agentvfs trusted Runtime Module.

Exposes model-facing tools (agentvfs_status, agentvfs_checkpoint,
agentvfs_rollback) over one Host-bound agentvfs workspace control socket.
The workspace binding is Host-only composition: the Host sets the substrate
attribute ``agentvfs`` (workspace name, or mapping with ``workspace`` and
optional explicit ``socket``); the startup hook attaches to the already-running
workspace via session.json discovery and fails closed when it is not running.
The module never starts or stops the FUSE daemon. Tools enforce capability
authority on ``agentvfs:<workspace>`` before any socket traffic: READ for
status, WRITE for checkpoint, ADMIN for the destructive rollback.

Paired mode gives one tool call two-plane semantics. A paired checkpoint
snapshots the agentvfs filesystem first, then records a libOS checkpoint whose
metadata embeds the agentvfs commit hash (order forced by the metadata), after
probing the process's self-checkpoint authority so a denial leaves no orphan
on either plane. A paired rollback validates the named libOS checkpoint and
workspace first, rolls the filesystem back to its immutable commit, then attempts
the legitimate in-tool restore (``CheckpointManager.restore`` refuses while the
scheduler runs a quantum, so a model-invoked call reports
``libos_restore=pending_host_restore`` with a Host hint instead of bypassing
quiescence; a quiescent Host-driven call with admin authority restores both planes).
"""

from __future__ import annotations

import json
import math
import os
import re
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agent_libos.models import (
    AgentImage,
    CapabilityDecision,
    CapabilityRight,
    DataFlowDirection,
    DataSink,
    EventType,
    ExternalEffectClassification,
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
)
from agent_libos.models.exceptions import CapabilityDenied, ValidationError
from agent_libos.sdk import (
    ProtectedOperationContract,
    ProtectedOperationEvidence,
    ProtectedOperationInvocation,
    ProviderPhase,
    ResourcePolicy,
)
from agent_libos.substrate import ProviderEffectNotStarted
from agent_libos.tools.base import (
    SyncAgentTool,
    ToolContext,
    ToolErrorCode,
    ToolExecutionError,
    ToolPolicy,
)

_ADAPTER_ATTR = "_agent_libos_agentvfs_adapter"
_UNBOUND = "unbound"
_WORKSPACE_NAME = re.compile(r"^[A-Za-z0-9._-]{1,80}$")
MODULE_ID = "agent-libos-agentvfs:v0"

LIBOS_RESTORE_RESTORED = "restored"
LIBOS_RESTORE_WARNINGS = "restored_with_warnings"
LIBOS_RESTORE_PENDING = "pending_host_restore"
LIBOS_RESTORE_SKIPPED = "skipped"
_HOST_RESTORE_HINT = (
    "Host must run runtime.checkpoint.restore(actor, checkpoint_id, "
    "require_capability=False) for the paired checkpoint once the process is quiescent"
)


class AgentVfsControlError(RuntimeError):
    """The agentvfs control daemon rejected or failed one request."""

    def __init__(self, request: str, detail: str) -> None:
        super().__init__(f"agentvfs control request {request!r} failed: {detail}")
        self.request = request
        self.detail = detail


class AgentVfsControlClient:
    """Newline-delimited JSON AF_UNIX client for one agentvfs daemon socket."""

    def __init__(self, socket_path: str | Path, *, timeout_s: float = 30.0) -> None:
        self.socket_path = str(socket_path)
        self.timeout_s = timeout_s

    def _connect(self) -> socket.socket:
        sock = None
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(self.timeout_s)
            sock.connect(self.socket_path)
        except (OSError, ValueError, AttributeError) as exc:
            if sock is not None:
                sock.close()
            # No command bytes have been sent, so the provider cannot have
            # changed state and reserved authority can safely be returned.
            raise ProviderEffectNotStarted("agentvfs control connection failed") from exc
        return sock

    def request(self, line: str) -> dict[str, Any]:
        with self._connect() as sock:
            sock.sendall((line + "\n").encode())
            buffer = bytearray()
            while not buffer.endswith(b"\n"):
                chunk = sock.recv(4096)
                if not chunk:
                    break
                buffer += chunk
        text = bytes(buffer).decode(errors="replace").strip()
        if not text:
            raise AgentVfsControlError(line, "empty response")
        try:
            response = json.loads(text)
        except ValueError as exc:
            raise AgentVfsControlError(line, f"invalid JSON reply: {text!r}") from exc
        if not isinstance(response, dict) or response.get("ok") is not True:
            detail = response.get("error") if isinstance(response, dict) else text
            raise AgentVfsControlError(line, str(detail))
        result_field = {"checkpoint": "commit", "rollback": "rolled_back_to"}.get(
            line.split(" ", 1)[0]
        )
        if result_field is not None:
            commit = response.get(result_field)
            if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{64}", commit) is None:
                raise AgentVfsControlError(line, f"invalid {result_field} in reply")
        return response


@dataclass(frozen=True)
class AgentVfsBinding:
    """Host-owned attachment to one running agentvfs workspace."""

    workspace: str
    socket: str | None = None
    request_timeout_s: float = 30.0


def _coerce_binding(value: Any) -> AgentVfsBinding:
    if isinstance(value, AgentVfsBinding):
        binding = value
    elif isinstance(value, str):
        binding = AgentVfsBinding(workspace=value)
    elif isinstance(value, dict):
        binding = AgentVfsBinding(**value)
    else:
        binding = AgentVfsBinding(
            **{
                field_name: getattr(value, field_name)
                for field_name in AgentVfsBinding.__dataclass_fields__
                if hasattr(value, field_name)
            }
        )
    if (
        not isinstance(binding.workspace, str)
        or binding.workspace in {".", ".."}
        or not _WORKSPACE_NAME.fullmatch(binding.workspace)
    ):
        raise ValidationError(
            f"agentvfs workspace name {binding.workspace!r} must match [A-Za-z0-9._-]{{1,80}}"
        )
    if (
        isinstance(binding.request_timeout_s, bool)
        or not isinstance(binding.request_timeout_s, (int, float))
        or not math.isfinite(binding.request_timeout_s)
        or binding.request_timeout_s <= 0
    ):
        raise ValidationError("agentvfs request_timeout_s must be finite and positive")
    return binding


def _runtime_root() -> Path:
    """Mirror agentvfs default_workspace_root(): $XDG_RUNTIME_DIR/agentvfs
    when set, otherwise /tmp/agentvfs-<uid>."""
    base = os.environ.get("XDG_RUNTIME_DIR")
    return Path(base) / "agentvfs" if base else Path(f"/tmp/agentvfs-{os.getuid()}")


def _discover_socket(binding: AgentVfsBinding) -> str:
    if binding.socket:
        return str(binding.socket)
    session_path = _runtime_root() / binding.workspace / "session.json"
    try:
        record = json.loads(session_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValidationError(
            f"agentvfs workspace {binding.workspace!r} is not attachable: "
            f"cannot read session file {session_path}"
        ) from exc
    if not isinstance(record, dict):
        raise ValidationError("agentvfs session file must contain a JSON object")
    socket_path = record.get("socket")
    if (
        record.get("status") != "started"
        or not isinstance(socket_path, str)
        or not socket_path
    ):
        raise ValidationError(
            f"agentvfs workspace {binding.workspace!r} is not running "
            f"(session status={record.get('status')!r})"
        )
    return str(socket_path)


class AgentVfsAdapter:
    """Capability-gated bridge from process authority to the control socket."""

    def __init__(self, host: Any, binding: AgentVfsBinding) -> None:
        self.host = host
        self.binding = binding
        self.resource = f"agentvfs:{binding.workspace}"
        self.client = AgentVfsControlClient(
            _discover_socket(binding), timeout_s=binding.request_timeout_s
        )
        for operation in ("status", "checkpoint", "rollback"):
            host.protected_operations.register_contract(
                ProtectedOperationContract(
                    name=f"module.agentvfs.{operation}",
                    provider="agentvfs",
                    operation=operation,
                    evidence_roles=("audit", "event", "effect"),
                    resource_policy=ResourcePolicy.NONE,
                    state_mutation=operation != "status",
                    information_flow=True,
                    data_flow_direction=DataFlowDirection.BIDIRECTIONAL,
                )
            )

    def _require_right(
        self, pid: str, right: CapabilityRight, operation: str
    ) -> CapabilityDecision:
        decision = self.host.capability.authorize(pid, self.resource, right)
        if not decision.allowed:
            self.host.audit.record(
                actor=pid,
                action=f"module.agentvfs.{operation}.denied",
                target=self.resource,
                decision={"right": right.value, "reason": decision.reason},
            )
            raise CapabilityDenied(
                f"{pid} denied agentvfs {operation} on {self.resource}: {decision.reason}"
            )
        return decision

    def classify_external_effect(
        self, operation: str, context: dict[str, Any], result: Any
    ) -> ExternalEffectClassification:
        mutates = operation != "status"
        return ExternalEffectClassification(
            rollback_class=(
                ExternalEffectRollbackClass.IRREVERSIBLE
                if mutates
                else ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED
            ),
            rollback_status=(
                ExternalEffectRollbackStatus.NOT_SUPPORTED
                if mutates
                else ExternalEffectRollbackStatus.NOT_REQUIRED
            ),
            state_mutation=mutates,
            information_flow=True,
            metadata={"workspace": self.binding.workspace, "operation": operation},
        )

    def _request(
        self, pid: str, operation: str, right: CapabilityRight, argument: str | None = None
    ) -> dict[str, Any]:
        decision = self._require_right(pid, right, operation)
        context = {"workspace": self.binding.workspace, "operation": operation}
        canonical_args = dict(context)
        if argument is not None:
            canonical_args["argument"] = argument
        line = operation if argument is None else f"{operation} {argument}"
        flow_context = self.host.data_flow.current_context()
        invocation = ProtectedOperationInvocation(
            pid=pid,
            actor=pid,
            target=self.resource,
            decisions=(decision,),
            canonical_args=canonical_args,
            observation=context,
            data_sink=DataSink(self.resource),
            data_flow_context=flow_context,
            data_flow_ingress_context=self.host.data_flow.unclassified_ingress_context(
                flow_context, origin="external:agentvfs"
            ),
            data_flow_payload=line,
            data_flow_operation=f"module.agentvfs.{operation}",
        )
        mutates = operation != "status"
        with self.host.protected_operations.start(
            f"module.agentvfs.{operation}", invocation, provider=self
        ) as protected:
            response = protected.call(
                ProviderPhase(operation, state_mutation=mutates, information_flow=True),
                self.client.request,
                line,
            )
            evidence = ProtectedOperationEvidence(
                event_type=EventType.EXTERNAL_WRITE if mutates else EventType.EXTERNAL_READ,
                event_source=pid,
                event_target=self.resource,
                event_payload=context,
                audit_action=f"module.agentvfs.{operation}",
                audit_actor=pid,
                audit_target=self.resource,
                audit_decision=context,
            )
            return protected.complete(
                response,
                evidence,
                classification_context=context,
                classification_result=response,
            )

    def status(self, pid: str) -> dict[str, Any]:
        return self._request(pid, "status", CapabilityRight.READ)

    def checkpoint(self, pid: str, label: str) -> dict[str, Any]:
        return self._request(pid, "checkpoint", CapabilityRight.WRITE, label)

    def rollback(self, pid: str, target: str) -> dict[str, Any]:
        return self._request(pid, "rollback", CapabilityRight.ADMIN, target)

    def detach(self) -> bool:
        """Shutdown finalizer: release module state without stopping the daemon."""
        self.host.audit.record(
            actor=f"module:{MODULE_ID}",
            action="module.agentvfs.detach",
            target=self.resource,
            decision={"workspace": self.binding.workspace},
        )
        return True


def initialize_agentvfs(runtime: Any) -> None:
    if runtime.get_runtime_attribute(_ADAPTER_ATTR) is not None:
        return
    binding_value = getattr(runtime.substrate, "agentvfs", None)
    if binding_value is None:
        # Inert without a Host binding: the tools stay registered but fail
        # closed at call time because no socket path exists to talk to.
        runtime.set_runtime_attribute(_ADAPTER_ATTR, _UNBOUND)
        return
    if not hasattr(socket, "AF_UNIX"):
        raise ValidationError("agentvfs requires Unix-domain socket support (AF_UNIX)")
    binding = _coerce_binding(binding_value)
    adapter = AgentVfsAdapter(runtime, binding)
    runtime.set_runtime_attribute(_ADAPTER_ATTR, adapter)
    runtime.bind_shutdown_finalizer(adapter.detach)


def _adapter(ctx: ToolContext) -> AgentVfsAdapter:
    runtime = _runtime(ctx)
    adapter = runtime.module_state.get(_ADAPTER_ATTR)
    if adapter == _UNBOUND:
        raise ToolExecutionError(
            "No agentvfs workspace is bound; the Host must set the substrate "
            "'agentvfs' binding before this tool can run.",
            code=ToolErrorCode.EXECUTION_ERROR,
            retryable=False,
        )
    if adapter is None:
        raise ToolExecutionError(
            "agentvfs module has not initialized.",
            code=ToolErrorCode.EXECUTION_ERROR,
            retryable=False,
        )
    return adapter


def _runtime(ctx: ToolContext) -> Any:
    if ctx.runtime is None:
        raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
    return ctx.runtime


def _validate_label(value: str, field: str) -> str:
    if not _WORKSPACE_NAME.fullmatch(value):
        raise ValidationError(
            f"agentvfs {field} {value!r} must match [A-Za-z0-9._-]{{1,80}}"
        )
    return value


def _require_libos_checkpoint_right(runtime: Any, pid: str) -> None:
    """Probe the process's self-checkpoint authority before agentvfs traffic."""
    resource = f"checkpoint:process:{pid}"
    decision = runtime.capability.authorize(pid, resource, CapabilityRight.WRITE)
    if not decision.allowed:
        runtime.audit.record(
            actor=pid,
            action="module.agentvfs.pair_checkpoint.denied",
            target=resource,
            decision={"right": CapabilityRight.WRITE.value, "reason": decision.reason},
        )
        raise CapabilityDenied(
            f"{pid} denied paired libOS checkpoint on {resource}: {decision.reason}"
        )


def _create_paired_libos_checkpoint(
    runtime: Any,
    pid: str,
    workspace: str,
    label: str,
    commit: str,
    reason: str | None,
) -> str:
    try:
        return runtime.checkpoint.create(
            pid,
            reason or f"agentvfs checkpoint {label}",
            actor=pid,
            metadata={
                "agentvfs_workspace": workspace,
                "agentvfs_label": label,
                "agentvfs_commit": commit,
            },
        )
    except Exception as exc:
        raise ToolExecutionError(
            f"agentvfs checkpoint {commit} succeeded but the paired libOS "
            f"checkpoint failed: {exc}",
            code=ToolErrorCode.EXECUTION_ERROR,
            retryable=False,
            details={
                "agentvfs_commit": commit,
                "agentvfs_label": label,
                "pairing": "libos_checkpoint_failed",
            },
        ) from exc


def _inspect_paired_checkpoint(
    runtime: Any,
    pid: str,
    checkpoint_id: str,
    workspace: str,
    target: str,
) -> dict[str, Any]:
    """Validate the complete pair before issuing a destructive socket request."""
    inspected = runtime.checkpoint.inspect(checkpoint_id, actor=pid)
    summary = inspected["checkpoint"]
    if summary["pid"] != pid:
        raise ValidationError(
            f"paired libOS checkpoint {checkpoint_id} belongs to "
            f"{summary['pid']}, not {pid}"
        )
    metadata = summary.get("metadata") or {}
    if metadata.get("agentvfs_workspace") != workspace:
        raise ValidationError(
            f"paired libOS checkpoint {checkpoint_id} does not belong to "
            f"agentvfs workspace {workspace}"
        )
    commit = metadata.get("agentvfs_commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{64}", commit) is None:
        raise ValidationError(
            f"paired libOS checkpoint {checkpoint_id} has no valid agentvfs commit"
        )
    label = metadata.get("agentvfs_label")
    if not isinstance(label, str) or _WORKSPACE_NAME.fullmatch(label) is None:
        raise ValidationError(
            f"paired libOS checkpoint {checkpoint_id} has no valid agentvfs label"
        )
    if target not in {label, commit}:
        raise ValidationError(
            f"agentvfs target {target} does not identify paired libOS "
            f"checkpoint {checkpoint_id}; use its label {label} or commit {commit}"
        )
    return metadata


def _require_commit_pairing(
    metadata: dict[str, Any], checkpoint_id: str, rolled_to: str
) -> None:
    paired_commit = metadata.get("agentvfs_commit")
    if paired_commit != rolled_to:
        raise ValidationError(
            f"paired libOS checkpoint {checkpoint_id} references agentvfs "
            f"commit {paired_commit} but the workspace rolled back to {rolled_to}; "
            "the filesystem rollback stands, re-pair with the matching checkpoint"
        )


@dataclass(frozen=True)
class _LibosRestoreOutcome:
    status: str
    hint: str | None = None
    reconciliation_pending: bool = False
    publication_id: str | None = None


def _restore_libos_paired(
    runtime: Any, pid: str, workspace: str, checkpoint_id: str
) -> _LibosRestoreOutcome:
    """Preserve both admission refusals and committed-but-pending restores."""
    decision = runtime.capability.authorize(
        pid, f"checkpoint:{checkpoint_id}", CapabilityRight.ADMIN
    )
    if not decision.allowed:
        hint = (
            f"{_HOST_RESTORE_HINT}; process lacks admin authority on "
            f"checkpoint:{checkpoint_id}"
        )
        return _pending_restore(
            runtime, pid, workspace, checkpoint_id, "authority", hint
        )
    try:
        result = runtime.checkpoint.restore(pid, checkpoint_id)
    except CapabilityDenied:
        # Restore also checks authority for images it must replace. Those
        # pre-commit refusals leave libOS unchanged, but the filesystem has
        # already rolled back and its completed outcome must remain visible.
        hint = (
            f"{_HOST_RESTORE_HINT}; libOS restore requires additional authority. "
            "The filesystem rollback has completed. Do not repeat it."
        )
        return _pending_restore(
            runtime, pid, workspace, checkpoint_id, "restore_authority", hint
        )
    except ValidationError as exc:
        # Only known pre-commit admission refusals are recoverable here.
        # Snapshot and publication validation failures must still propagate.
        detail = str(exc)
        if (
            detail == "checkpoint restore refused while scheduler is running"
            or detail.startswith(
                "checkpoint restore refused while scheduler futures are active: "
            )
        ):
            reason = "scheduler_busy"
            unblock = "Wait for the scheduler and its active futures to become idle"
        elif detail.startswith(
            "checkpoint restore refused while scoped ObjectTasks are active: "
        ):
            reason = "object_tasks_active"
            unblock = "Host must finish or cancel active scoped ObjectTasks"
        elif detail.startswith(
            "checkpoint restore refused while scoped Durable TaskRuns are active: "
        ):
            reason = "task_runs_active"
            unblock = "Host must finish or cancel active scoped Durable TaskRuns"
        elif detail == "checkpoint restore or recovery is already in progress":
            reason = "restore_busy"
            unblock = "Wait for the current restore or recovery to finish"
        else:
            raise
        hint = (
            f"{unblock} before restoring libOS. {_HOST_RESTORE_HINT}. "
            "The filesystem rollback has completed. Do not repeat it."
        )
        return _pending_restore(runtime, pid, workspace, checkpoint_id, reason, hint)
    status = str(result["status"])
    pending = bool(result["reconciliation_pending"])
    publication_id = str(result["publication_id"])
    hint = None
    if pending or status == LIBOS_RESTORE_WARNINGS:
        hint = (
            "libOS main state was committed but restore reconciliation is pending; "
            f"the Host must complete startup recovery for publication {publication_id} "
            "before resuming. Do not repeat the filesystem rollback."
        )
    return _LibosRestoreOutcome(status, hint, pending, publication_id)


def _pending_restore(
    runtime: Any,
    pid: str,
    workspace: str,
    checkpoint_id: str,
    reason: str,
    hint: str,
) -> _LibosRestoreOutcome:
    runtime.audit.record(
        actor=pid,
        action="module.agentvfs.libos_restore_pending",
        target=f"checkpoint:{checkpoint_id}",
        decision={"reason": reason, "workspace": workspace},
    )
    return _LibosRestoreOutcome(LIBOS_RESTORE_PENDING, hint)


class AgentvfsStatusArgs(BaseModel):
    pass


class AgentvfsStatusOutput(BaseModel):
    workspace: str
    commit: str | None = None
    branch: str | None = None
    version: str | None = None
    telemetry_drops_total: int | None = None


class AgentvfsCheckpointArgs(BaseModel):
    label: str = Field(description="Checkpoint label, charset [A-Za-z0-9._-], 1-80 chars.")
    pair_libos: bool = Field(
        default=False,
        description=(
            "Also create a libOS checkpoint of this process whose metadata records "
            "the agentvfs commit hash, so a later paired rollback can restore both "
            "the filesystem and libOS object/SQL state."
        ),
    )
    reason: str | None = Field(
        default=None,
        description="Optional reason recorded on the paired libOS checkpoint.",
    )


class AgentvfsCheckpointOutput(BaseModel):
    commit: str
    label: str
    libos_checkpoint_id: str | None = Field(
        default=None,
        description="Present only for paired checkpoints: the libOS checkpoint holding this commit hash in its metadata.",
    )


class AgentvfsRollbackArgs(BaseModel):
    target: str = Field(
        description="Rollback target: a checkpoint label or 64-hex commit hash. In paired mode, "
        "must match the paired checkpoint label or commit; its immutable commit is restored."
    )
    pair_libos: bool = Field(
        default=False,
        description=(
            "Validate the named paired libOS checkpoint, workspace, and target before "
            "rolling back to its immutable filesystem commit, then attempt its libOS "
            "restore. The libOS restore is Host-mediated when the process "
            "lacks required authority or the runtime is busy with scheduled work, scoped "
            "tasks, or another restore; the result "
            "then reports libos_restore=pending_host_restore with a hint instead of "
            "restoring in-tool."
        ),
    )
    libos_checkpoint_id: str | None = Field(
        default=None,
        description="Required when pair_libos is true: the checkpoint id returned by a paired agentvfs_checkpoint.",
    )


class AgentvfsRollbackOutput(BaseModel):
    rolled_back_to: str
    paired_libos_checkpoint_id: str | None = None
    libos_restore: str = Field(
        description="restored | restored_with_warnings | pending_host_restore | skipped (non-paired rollback)."
    )
    libos_restore_hint: str | None = Field(
        default=None,
        description="Present when the Host must finish restore or recovery.",
    )
    libos_reconciliation_pending: bool = Field(
        default=False,
        description="True when libOS main state committed but startup recovery is required.",
    )
    libos_publication_id: str | None = Field(
        default=None,
        description="The libOS restore publication, including one awaiting reconciliation.",
    )


class AgentvfsStatusTool(SyncAgentTool[AgentvfsStatusArgs]):
    name = "agentvfs_status"
    description = (
        "Report the Host-bound agentvfs workspace daemon status (branch, commit, "
        "version). Requires agentvfs read capability; never mutates state."
    )
    args_schema = AgentvfsStatusArgs
    output_schema = AgentvfsStatusOutput
    policy = ToolPolicy(
        side_effects=False,
        idempotent=True,
        declared_permissions={"agentvfs.read"},
        timeout_s=None,
    )
    tags = ["agentvfs", "checkpoint", "inspect"]

    def run(self, args: AgentvfsStatusArgs, ctx: ToolContext) -> AgentvfsStatusOutput:
        adapter = _adapter(ctx)
        response = adapter.status(ctx.pid)
        return AgentvfsStatusOutput(
            workspace=adapter.binding.workspace,
            commit=response.get("commit"),
            branch=response.get("branch"),
            version=response.get("version"),
            telemetry_drops_total=response.get("telemetry_drops_total"),
        )


class AgentvfsCheckpointTool(SyncAgentTool[AgentvfsCheckpointArgs]):
    name = "agentvfs_checkpoint"
    description = (
        "Snapshot the process's agentvfs workspace filesystem into the "
        "content-addressed store and return the commit hash. Requires agentvfs "
        "write capability. With pair_libos=true it also records a libOS "
        "checkpoint of this process embedding the agentvfs commit hash, giving "
        "one call two-plane checkpoint semantics."
    )
    args_schema = AgentvfsCheckpointArgs
    output_schema = AgentvfsCheckpointOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"agentvfs.write", "checkpoint.write"},
        timeout_s=None,
    )
    tags = ["agentvfs", "checkpoint", "side_effect"]

    def run(self, args: AgentvfsCheckpointArgs, ctx: ToolContext) -> AgentvfsCheckpointOutput:
        adapter = _adapter(ctx)
        label = _validate_label(args.label, "label")
        if not args.pair_libos:
            response = adapter.checkpoint(ctx.pid, label)
            return AgentvfsCheckpointOutput(commit=str(response["commit"]), label=label)
        runtime = _runtime(ctx)
        _require_libos_checkpoint_right(runtime, ctx.pid)
        commit = str(adapter.checkpoint(ctx.pid, label)["commit"])
        checkpoint_id = _create_paired_libos_checkpoint(
            runtime,
            ctx.pid,
            adapter.binding.workspace,
            label,
            commit,
            args.reason,
        )
        return AgentvfsCheckpointOutput(
            commit=commit, label=label, libos_checkpoint_id=checkpoint_id
        )


class AgentvfsRollbackTool(SyncAgentTool[AgentvfsRollbackArgs]):
    name = "agentvfs_rollback"
    description = (
        "Roll the process's agentvfs workspace filesystem back to a checkpoint "
        "label or commit hash, restoring deleted and overwritten files. "
        "Destructive: requires the stronger agentvfs admin capability. With "
        "pair_libos=true it validates the checkpoint workspace and target before "
        "restoring the paired immutable filesystem commit and libOS object/SQL state. "
        "When the in-tool libOS "
        "restore is not permitted (missing checkpoint or image authority) or is "
        "refused while the scheduler, scoped tasks, or another restore is active, "
        "the filesystem rollback still stands "
        "and the result reports libos_restore=pending_host_restore so the Host "
        "can finish the second plane once the process is quiescent."
    )
    args_schema = AgentvfsRollbackArgs
    output_schema = AgentvfsRollbackOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"agentvfs.admin", "checkpoint.restore"},
        timeout_s=None,
    )
    tags = ["agentvfs", "rollback", "high_risk", "side_effect"]

    def run(self, args: AgentvfsRollbackArgs, ctx: ToolContext) -> AgentvfsRollbackOutput:
        adapter = _adapter(ctx)
        target = _validate_label(args.target, "target")
        if not args.pair_libos:
            response = adapter.rollback(ctx.pid, target)
            return AgentvfsRollbackOutput(
                rolled_back_to=str(response["rolled_back_to"]),
                libos_restore=LIBOS_RESTORE_SKIPPED,
            )
        if not args.libos_checkpoint_id:
            raise ValidationError(
                "agentvfs_rollback with pair_libos=true requires libos_checkpoint_id "
                "from a paired agentvfs_checkpoint result"
            )
        runtime = _runtime(ctx)
        paired_metadata = _inspect_paired_checkpoint(
            runtime,
            ctx.pid,
            args.libos_checkpoint_id,
            adapter.binding.workspace,
            target,
        )
        rolled_to = str(
            adapter.rollback(ctx.pid, paired_metadata["agentvfs_commit"])["rolled_back_to"]
        )
        _require_commit_pairing(paired_metadata, args.libos_checkpoint_id, rolled_to)
        restore = _restore_libos_paired(
            runtime, ctx.pid, adapter.binding.workspace, args.libos_checkpoint_id
        )
        return AgentvfsRollbackOutput(
            rolled_back_to=rolled_to,
            paired_libos_checkpoint_id=args.libos_checkpoint_id,
            libos_restore=restore.status,
            libos_restore_hint=restore.hint,
            libos_reconciliation_pending=restore.reconciliation_pending,
            libos_publication_id=restore.publication_id,
        )


def register_module(ctx: Any) -> None:
    for tool in (
        AgentvfsStatusTool(),
        AgentvfsCheckpointTool(),
        AgentvfsRollbackTool(),
    ):
        ctx.register_tool(tool)

    ctx.register_image(
        AgentImage(
            image_id="agentvfs-agent:v0",
            name="agentvfs-agent",
            default_tools=[
                "process_exit",
                "agentvfs_checkpoint",
                "agentvfs_rollback",
                "agentvfs_status",
            ],
            required_capabilities=[
                {"resource": "agentvfs:*", "rights": ["read", "write", "admin"]}
            ],
            metadata={"module": MODULE_ID},
        )
    )
    ctx.add_startup_hook(initialize_agentvfs)
