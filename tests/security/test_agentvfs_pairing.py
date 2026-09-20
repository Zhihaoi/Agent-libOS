from __future__ import annotations

from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
import socket
from typing import Any

import pytest

from agent_libos import ObjectMetadata, ObjectType, Runtime, TaskRunSpecV1
from agent_libos.config import AgentLibOSConfig
from agent_libos.models import ObjectPatch, ObjectTaskStatus, TaskRunStatus
from agent_libos.models.exceptions import ValidationError
from agent_libos.substrate import LocalResourceProviderSubstrate
from agent_libos.tools.base import ToolContext
from modules.agentvfs.agentvfs_module import AgentvfsRollbackArgs, AgentvfsRollbackTool

from tests.security.test_agentvfs_module import (
    COMMIT_A,
    COMMIT_B,
    TestAgentVfsModule as _ModuleHarness,
    _module_manifest,
    _trust_key,
)
from tests.support.public_errors import assert_public_error_message


pytestmark = pytest.mark.skipif(
    not hasattr(socket, "AF_UNIX"), reason="agentvfs requires Unix-domain sockets"
)


@pytest.fixture
def agentvfs() -> Iterator[_ModuleHarness]:
    harness = _ModuleHarness()
    harness.setup_method()
    try:
        yield harness
    finally:
        harness.teardown_method()


def _assert_no_rollback_effect(agentvfs: _ModuleHarness, runtime: Runtime) -> None:
    assert agentvfs._fake.received == []
    assert "module.agentvfs.rollback" not in {
        record.action for record in runtime.audit.trace()
    }
    assert not any(
        event.type == "external_write"
        and event.target == f"agentvfs:{agentvfs._workspace}"
        for event in runtime.store.list_events()
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"agentvfs_workspace": "another-workspace"},
        {"agentvfs_workspace": None},
        {"agentvfs_commit": None},
        {"agentvfs_commit": "not-a-commit"},
        {"agentvfs_commit": "b" * 63},
        {"agentvfs_commit": 123},
        {"agentvfs_label": None},
        {"agentvfs_label": "c1\nrollback other"},
    ],
)
def test_paired_rollback_rejects_invalid_pair_before_socket_traffic(
    agentvfs: _ModuleHarness,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, Any],
) -> None:
    runtime = agentvfs._open_runtime(monkeypatch)
    pid = agentvfs._spawn(runtime, ["read", "write", "admin"])
    metadata = {
        "agentvfs_workspace": agentvfs._workspace,
        "agentvfs_label": "c1",
        "agentvfs_commit": COMMIT_B,
        **overrides,
    }
    checkpoint_id = runtime.checkpoint.create(
        pid, "invalid pair", actor=pid, metadata=metadata
    )

    result = runtime.tools.call(
        pid,
        "agentvfs_rollback",
        {"target": "c1", "pair_libos": True, "libos_checkpoint_id": checkpoint_id},
    )

    assert not result.ok
    assert_public_error_message(
        result.error, code="validation_error", error_type="ValidationError"
    )
    _assert_no_rollback_effect(agentvfs, runtime)


def test_paired_rollback_rejects_unpaired_checkpoint_before_socket_traffic(
    agentvfs: _ModuleHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = agentvfs._open_runtime(monkeypatch)
    pid = agentvfs._spawn(runtime, ["read", "write", "admin"])
    checkpoint_id = runtime.checkpoint.create(pid, "unpaired", actor=pid)

    result = runtime.tools.call(
        pid,
        "agentvfs_rollback",
        {"target": "c1", "pair_libos": True, "libos_checkpoint_id": checkpoint_id},
    )

    assert not result.ok
    assert_public_error_message(
        result.error, code="validation_error", error_type="ValidationError"
    )
    _assert_no_rollback_effect(agentvfs, runtime)


@pytest.mark.parametrize("target", ["c1", COMMIT_B])
def test_paired_rollback_uses_immutable_commit_when_label_has_moved(
    agentvfs: _ModuleHarness, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    runtime = agentvfs._open_runtime(monkeypatch)
    pid = agentvfs._spawn(runtime, ["read", "write", "admin"])
    created = runtime.tools.call(
        pid, "agentvfs_checkpoint", {"label": "c1", "pair_libos": True}
    )
    assert created.ok, created.error
    # The daemon now resolves the label to a different filesystem snapshot.
    agentvfs._fake.line_responses["rollback c1"] = {
        "ok": True,
        "rolled_back_to": COMMIT_A,
    }

    result = runtime.tools.call(
        pid,
        "agentvfs_rollback",
        {
            "target": target,
            "pair_libos": True,
            "libos_checkpoint_id": created.payload["libos_checkpoint_id"],
        },
    )

    assert result.ok, result.error
    assert result.payload["rolled_back_to"] == COMMIT_B
    assert result.payload["libos_restore"] == "pending_host_restore"
    assert agentvfs._fake.received == ["checkpoint c1", f"rollback {COMMIT_B}"]


def test_paired_rollback_rejects_unexpected_daemon_commit_before_libos_restore(
    agentvfs: _ModuleHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = agentvfs._open_runtime(monkeypatch)
    pid = agentvfs._spawn(runtime, ["read", "write", "admin"])
    created = runtime.tools.call(
        pid, "agentvfs_checkpoint", {"label": "c1", "pair_libos": True}
    )
    assert created.ok, created.error
    checkpoint_id = created.payload["libos_checkpoint_id"]
    runtime.capability.grant(
        subject=pid,
        resource=f"checkpoint:{checkpoint_id}",
        rights=["admin"],
        issued_by="test",
    )
    agentvfs._fake.line_responses[f"rollback {COMMIT_B}"] = {
        "ok": True,
        "rolled_back_to": COMMIT_A,
    }
    restore_calls = []
    monkeypatch.setattr(
        runtime.checkpoint,
        "restore",
        lambda *args, **kwargs: restore_calls.append((args, kwargs)),
    )

    result = runtime.tools.call(
        pid,
        "agentvfs_rollback",
        {"target": "c1", "pair_libos": True, "libos_checkpoint_id": checkpoint_id},
    )

    assert not result.ok
    assert_public_error_message(
        result.error, code="validation_error", error_type="ValidationError"
    )
    assert restore_calls == []
    assert agentvfs._fake.received == ["checkpoint c1", f"rollback {COMMIT_B}"]


def test_paired_rollback_preserves_partial_outcome_when_image_authority_is_missing(
    agentvfs: _ModuleHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = agentvfs._open_runtime(monkeypatch)
    pid = agentvfs._spawn(runtime, ["read", "write", "admin"])
    note = runtime.memory.create_object(
        pid,
        ObjectType.PLAN,
        {"version": 1},
        ObjectMetadata(title="paired note"),
        immutable=False,
        name="agentvfs.paired.note",
    )
    original_image = runtime.get_image("agentvfs-agent:v0")
    created = runtime.tools.call(
        pid, "agentvfs_checkpoint", {"label": "c1", "pair_libos": True}
    )
    assert created.ok, created.error
    checkpoint_id = created.payload["libos_checkpoint_id"]
    runtime.capability.grant(
        subject=pid,
        resource=f"checkpoint:{checkpoint_id}",
        rights=["admin"],
        issued_by="test",
    )
    runtime.memory.update_object(pid, note, ObjectPatch(payload={"version": 2}))
    changed_image = replace(original_image, name="changed image")
    runtime.register_image(changed_image, replace=True)

    result = runtime.tools.call(
        pid,
        "agentvfs_rollback",
        {"target": "c1", "pair_libos": True, "libos_checkpoint_id": checkpoint_id},
    )

    assert result.ok, result.error
    assert result.payload["rolled_back_to"] == COMMIT_B
    assert result.payload["paired_libos_checkpoint_id"] == checkpoint_id
    assert result.payload["libos_restore"] == "pending_host_restore"
    hint = result.payload["libos_restore_hint"]
    assert "additional authority" in hint
    assert "Do not repeat" in hint
    assert "image:agentvfs-agent:v0" not in hint
    assert runtime.get_image(original_image.image_id).name == changed_image.name
    assert runtime.memory.get_object_by_name(
        pid, "agentvfs.paired.note"
    ).payload == {"version": 2}
    pending = [
        record
        for record in runtime.audit.trace()
        if record.action == "module.agentvfs.libos_restore_pending"
    ]
    assert len(pending) == 1
    assert pending[0].target == f"checkpoint:{checkpoint_id}"
    assert pending[0].decision["reason"] == "restore_authority"
    expected_requests = ["checkpoint c1", f"rollback {COMMIT_B}"]
    assert agentvfs._fake.received == expected_requests

    restored = runtime.checkpoint.restore(
        "test", checkpoint_id, require_capability=False
    )

    assert restored["status"] == "restored"
    assert runtime.get_image(original_image.image_id).name == original_image.name
    assert runtime.memory.get_object_by_name(
        pid, "agentvfs.paired.note"
    ).payload == {"version": 1}
    assert agentvfs._fake.received == expected_requests


def test_paired_rollback_preserves_committed_restore_recovery_outcome(
    agentvfs: _ModuleHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = agentvfs._open_runtime(monkeypatch)
    pid = agentvfs._spawn(runtime, ["read", "write", "admin"])
    created = runtime.tools.call(
        pid, "agentvfs_checkpoint", {"label": "c1", "pair_libos": True}
    )
    assert created.ok, created.error
    checkpoint_id = created.payload["libos_checkpoint_id"]
    runtime.capability.grant(
        subject=pid,
        resource=f"checkpoint:{checkpoint_id}",
        rights=["admin"],
        issued_by="test",
    )

    def fail_image_reconciliation(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("injected pairing restore reconciliation failure")

    monkeypatch.setattr(runtime.checkpoint, "_restore_images", fail_image_reconciliation)
    try:
        # Inspect the tool result directly: the core recovery fence intentionally
        # prevents ToolBroker from publishing any subsequent TOOL_RESULT Object.
        result = AgentvfsRollbackTool().run(
            AgentvfsRollbackArgs(
                target="c1", pair_libos=True, libos_checkpoint_id=checkpoint_id
            ),
            ToolContext(trace_id="pairing", call_id="rollback", pid=pid, runtime=runtime),
        )
        assert result.libos_restore == "restored_with_warnings"
        assert result.libos_reconciliation_pending is True
        publication_id = result.libos_publication_id
        assert publication_id
        publication = runtime.store.get_runtime_publication(publication_id)
        assert publication is not None
        assert publication["state"] == "failed"
        assert publication_id in result.libos_restore_hint
        assert "Do not repeat" in result.libos_restore_hint
        assert runtime.lifecycle.state == "close_failed"
        assert agentvfs._fake.received == ["checkpoint c1", f"rollback {COMMIT_B}"]
    finally:
        if runtime.lifecycle.state == "close_failed":
            runtime.release_recovery_diagnostics()
            agentvfs._runtimes.remove(runtime)


def test_paired_restore_warning_fields_survive_broker_projection(
    agentvfs: _ModuleHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = agentvfs._open_runtime(monkeypatch)
    pid = agentvfs._spawn(runtime, ["read", "write", "admin"])
    created = runtime.tools.call(
        pid, "agentvfs_checkpoint", {"label": "c1", "pair_libos": True}
    )
    assert created.ok, created.error
    checkpoint_id = created.payload["libos_checkpoint_id"]
    runtime.capability.grant(
        subject=pid,
        resource=f"checkpoint:{checkpoint_id}",
        rights=["admin"],
        issued_by="test",
    )
    monkeypatch.setattr(
        runtime.checkpoint,
        "restore",
        lambda *args, **kwargs: {
            "status": "restored_with_warnings",
            "reconciliation_pending": True,
            "publication_id": "publication_recovery_required",
        },
    )

    result = runtime.tools.call(
        pid,
        "agentvfs_rollback",
        {"target": "c1", "pair_libos": True, "libos_checkpoint_id": checkpoint_id},
    )

    assert result.ok, result.error
    assert result.payload["libos_restore"] == "restored_with_warnings"
    assert result.payload["libos_reconciliation_pending"] is True
    assert result.payload["libos_publication_id"] == "publication_recovery_required"
    assert "Do not repeat" in result.payload["libos_restore_hint"]


@pytest.mark.parametrize(
    ("blocker", "pending_reason"),
    [
        ("object_task", "object_tasks_active"),
        ("task_run", "task_runs_active"),
        ("restore_busy", "restore_busy"),
    ],
    ids=["object-task", "task-run", "restore-busy"],
)
def test_paired_rollback_preserves_partial_outcome_when_restore_is_blocked(
    agentvfs: _ModuleHarness,
    tmp_path: Path,
    blocker: str,
    pending_reason: str,
) -> None:
    config = AgentLibOSConfig()
    config = replace(
        config,
        task_runs=replace(config.task_runs, plaintext_payloads_enabled=True),
    )
    substrate = LocalResourceProviderSubstrate(str(tmp_path))
    substrate.agentvfs = {
        "workspace": agentvfs._workspace,
        "socket": agentvfs._fake.path,
    }
    manifest = _module_manifest()
    runtime = Runtime.open(
        "local",
        substrate=substrate,
        config=config,
        module_manifests=(str(manifest),),
        trusted_modules=(_trust_key(manifest),),
    )
    agentvfs._runtimes.append(runtime)
    task_run = None
    object_task = None
    if blocker == "object_task":
        image = runtime.get_image("agentvfs-agent:v0")
        runtime.register_image(
            replace(image, default_tools=[*image.default_tools, "receive_process_messages"]),
            replace=True,
        )
        pid = agentvfs._spawn(runtime, ["read", "write", "admin"])
        runtime.capability.grant(pid, "process:spawn", ["write"], issued_by="test")
    elif blocker == "task_run":
        task_run = runtime.task_runs.create(
            TaskRunSpecV1(
                goal="paired rollback active TaskRun",
                display_title="paired rollback active TaskRun",
                image_id="agentvfs-agent:v0",
            ),
            client_request_id="paired-rollback-active-task-run",
        )
        pid = task_run.root_pid
        assert pid is not None
        runtime.capability.grant(
            pid, f"agentvfs:{agentvfs._workspace}", ["read", "write", "admin"],
            issued_by="test",
        )
    else:
        pid = agentvfs._spawn(runtime, ["read", "write", "admin"])
    note = runtime.memory.create_object(
        pid,
        ObjectType.ARTIFACT,
        {"version": 1},
        ObjectMetadata(title="paired task note"),
        immutable=False,
        name="agentvfs.paired.task.note",
    )
    created = runtime.tools.call(
        pid, "agentvfs_checkpoint", {"label": "c1", "pair_libos": True}
    )
    assert created.ok, created.error
    checkpoint_id = created.payload["libos_checkpoint_id"]
    runtime.capability.grant(
        pid, f"checkpoint:{checkpoint_id}", ["admin"], issued_by="test"
    )
    runtime.memory.update_object(pid, note, ObjectPatch(payload={"version": 2}))
    if blocker == "object_task":
        object_task = runtime.object_tasks.start(
            pid, note, "receive_process_messages", {"channel": "never"}
        )
        waiting = runtime.object_tasks.wait(
            object_task.task_id, actor_pid=pid, timeout=2.0
        )
        assert waiting.status is ObjectTaskStatus.WAITING_MESSAGE
    elif task_run is not None:
        assert runtime.task_runs.active_runs_for_pids((pid,))
    # A waiting task is a distinct restore guard, even with an idle scheduler.
    with runtime.scheduler.quiescent_state(reason="paired rollback regression"):
        pass
    publications_before = {
        row["publication_id"]
        for row in runtime.store.list_runtime_publications()
        if row["kind"] == "checkpoint_restore"
    }

    restore_scope = (
        runtime.checkpoint._restore_single_flight()
        if blocker == "restore_busy"
        else nullcontext()
    )
    with restore_scope:
        result = runtime.tools.call(
            pid,
            "agentvfs_rollback",
            {"target": "c1", "pair_libos": True, "libos_checkpoint_id": checkpoint_id},
        )

    assert result.ok, result.error
    assert result.payload["rolled_back_to"] == COMMIT_B
    assert result.payload["paired_libos_checkpoint_id"] == checkpoint_id
    assert result.payload["libos_restore"] == "pending_host_restore"
    assert result.payload["libos_reconciliation_pending"] is False
    assert result.payload["libos_publication_id"] is None
    hint = result.payload["libos_restore_hint"]
    assert "Do not repeat" in hint
    assert any(word in hint.lower() for word in ("finish", "complete", "cancel", "end"))
    if blocker == "restore_busy":
        assert "wait" in hint.lower()
    else:
        assert "task" in hint.lower()
    if object_task is not None:
        assert object_task.task_id not in hint
    if task_run is not None:
        assert task_run.run_id not in hint
    assert runtime.memory.get_object_by_name(
        pid, "agentvfs.paired.task.note"
    ).payload == {"version": 2}
    assert {
        row["publication_id"]
        for row in runtime.store.list_runtime_publications()
        if row["kind"] == "checkpoint_restore"
    } == publications_before
    pending = [
        record for record in runtime.audit.trace()
        if record.action == "module.agentvfs.libos_restore_pending"
    ]
    assert len(pending) == 1
    assert pending[0].target == f"checkpoint:{checkpoint_id}"
    assert pending[0].decision["reason"] == pending_reason
    effects = [
        effect for effect in runtime.store.list_external_effects(pid=pid)
        if effect.provider == "agentvfs" and effect.operation == "rollback"
    ]
    assert len(effects) == 1
    assert effects[0].transaction_state == "committed"
    assert effects[0].rollback_class == "irreversible"
    assert effects[0].event_id and effects[0].record_id
    assert any(
        event.event_id == effects[0].event_id
        and event.type == "external_write"
        and event.target == f"agentvfs:{agentvfs._workspace}"
        for event in runtime.store.list_events()
    )
    expected_requests = ["checkpoint c1", f"rollback {COMMIT_B}"]
    assert agentvfs._fake.received == expected_requests

    if object_task is not None:
        cancelled = runtime.object_tasks.cancel(object_task.task_id, actor_pid=pid)
        assert cancelled.status is ObjectTaskStatus.CANCELLED
    elif task_run is not None:
        latest = runtime.task_runs.get(task_run.run_id)
        cancelled_run = runtime.task_runs.cancel(
            task_run.run_id,
            expected_revision=latest.revision,
            command_id="cancel-before-paired-restore",
        )
        assert cancelled_run.status is TaskRunStatus.CANCELLED
    restored = runtime.checkpoint.restore("test", checkpoint_id, require_capability=False)
    assert restored["status"] == "restored"
    # TaskRun cancellation revokes process grants permanently; verify the
    # restored object through the Host store, without regranting authority.
    restored_note = runtime.store.get_object(note.oid)
    assert restored_note is not None
    assert restored_note.payload == {"version": 1}
    assert agentvfs._fake.received == expected_requests
    # TaskRun completion may redact retained effect payloads. The committed
    # external effect and its append-only audit/event identities must remain.
    retained = next(
        effect for effect in runtime.store.list_external_effects(pid=pid)
        if effect.effect_id == effects[0].effect_id
    )
    assert retained.transaction_state == "committed"
    assert retained.rollback_class == "irreversible"
    assert retained.record_id == effects[0].record_id
    assert retained.event_id == effects[0].event_id
    assert pending[0] in runtime.audit.trace()
    assert any(
        event.event_id == retained.event_id and event.type == "external_write"
        for event in runtime.store.list_events()
    )


def test_paired_rollback_does_not_swallow_unrelated_restore_validation_error(
    agentvfs: _ModuleHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = agentvfs._open_runtime(monkeypatch)
    pid = agentvfs._spawn(runtime, ["read", "write", "admin"])
    created = runtime.tools.call(
        pid, "agentvfs_checkpoint", {"label": "c1", "pair_libos": True}
    )
    assert created.ok, created.error
    checkpoint_id = created.payload["libos_checkpoint_id"]
    runtime.capability.grant(
        pid, f"checkpoint:{checkpoint_id}", ["admin"], issued_by="test"
    )

    def refuse_invalid_checkpoint(*args: Any, **kwargs: Any) -> None:
        raise ValidationError("checkpoint snapshot failed validation")

    monkeypatch.setattr(runtime.checkpoint, "restore", refuse_invalid_checkpoint)
    with pytest.raises(ValidationError, match="checkpoint snapshot failed validation"):
        AgentvfsRollbackTool().run(
            AgentvfsRollbackArgs(
                target="c1", pair_libos=True, libos_checkpoint_id=checkpoint_id
            ),
            ToolContext(trace_id="pairing", call_id="rollback", pid=pid, runtime=runtime),
        )
    assert agentvfs._fake.received == ["checkpoint c1", f"rollback {COMMIT_B}"]
    assert not any(
        record.action == "module.agentvfs.libos_restore_pending"
        for record in runtime.audit.trace()
    )
