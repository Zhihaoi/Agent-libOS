from __future__ import annotations

from collections.abc import Iterator
import socket
from typing import Any

import pytest

from agent_libos import Runtime
from agent_libos.tools.base import ToolContext
from modules.agentvfs.agentvfs_module import AgentvfsRollbackArgs, AgentvfsRollbackTool

from tests.security.test_agentvfs_module import (
    COMMIT_A,
    COMMIT_B,
    TestAgentVfsModule as _ModuleHarness,
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
