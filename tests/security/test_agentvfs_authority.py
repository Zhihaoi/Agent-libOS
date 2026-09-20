"""Finite authority and durable effect evidence for the agentvfs module."""

from __future__ import annotations

import hashlib
import socket
import threading
from pathlib import Path
from typing import Any

import pytest

from tests.security import test_agentvfs_module as module_tests


pytestmark = pytest.mark.skipif(
    not hasattr(socket, "AF_UNIX"), reason="agentvfs control requires Unix sockets"
)


@pytest.fixture
def agentvfs_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Isolate the trusted package so these tests never alter the source tree or
    # depend on another test's bytecode-cache cleanup.
    source_dir = Path(__file__).resolve().parents[2] / "modules" / "agentvfs"
    module_dir = tmp_path / "module"
    module_dir.mkdir()
    source = (source_dir / "agentvfs_module.py").read_bytes()
    (module_dir / "agentvfs_module.py").write_bytes(source)
    manifest_text = (source_dir / "module.yaml").read_text(encoding="utf-8")
    manifest = module_dir / "module.yaml"
    manifest.write_text(
        "\n".join(
            f"sha256: {hashlib.sha256(source).hexdigest()}"
            if line.startswith("sha256:") else line
            for line in manifest_text.splitlines()
        ) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module_tests, "_module_manifest", lambda: manifest)
    harness = module_tests.TestAgentVfsModule()
    harness.setup_method()
    try:
        runtime = harness._open_runtime(monkeypatch)
        pid = harness._spawn(runtime)
        yield runtime, pid, harness
    finally:
        harness.teardown_method()


@pytest.mark.parametrize(
    ("tool", "right", "args"),
    [
        ("agentvfs_status", "read", {}),
        ("agentvfs_checkpoint", "write", {"label": "once"}),
        ("agentvfs_rollback", "admin", {"target": "once"}),
    ],
)
def test_finite_grant_allows_only_one_control_request(
    agentvfs_runtime: Any, tool: str, right: str, args: dict[str, Any]
) -> None:
    runtime, pid, harness = agentvfs_runtime
    cap = runtime.capability.issue_trusted(
        subject=pid,
        resource=f"agentvfs:{harness._workspace}",
        rights=[right],
        issued_by="test",
        uses_remaining=1,
    )
    first = runtime.tools.call(pid, tool, args)
    assert first.ok, first.error
    assert runtime.store.get_capability(cap.cap_id).uses_remaining == 0
    second = runtime.tools.call(pid, tool, args)
    assert not second.ok
    assert len(harness._fake.received) == 1
    effects = [
        effect for effect in runtime.store.list_external_effects(pid=pid)
        if effect.provider == "agentvfs"
    ]
    assert len(effects) == 1
    assert effects[0].transaction_state == "committed"
    assert effects[0].record_id and effects[0].event_id
    assert effects[0].state_mutation == (right != "read")
    assert effects[0].rollback_class == (
        "no_rollback_required" if right == "read" else "irreversible"
    )


@pytest.mark.parametrize(
    "reply",
    [
        None,
        b"not valid JSON\n",
        b'{"ok": true}\n',
        b'{"ok": true, "commit": "not-a-commit"}\n',
    ],
)
def test_ambiguous_daemon_failure_consumes_grant_and_records_unknown_effect(
    agentvfs_runtime: Any, monkeypatch: pytest.MonkeyPatch, reply: bytes | None
) -> None:
    runtime, pid, harness = agentvfs_runtime
    cap = runtime.capability.issue_trusted(
        subject=pid,
        resource=f"agentvfs:{harness._workspace}",
        rights=["write"],
        issued_by="test",
        uses_remaining=1,
    )
    adapter = runtime.module_state.get("_agent_libos_agentvfs_adapter")
    # Receive the command and close without a reply: the caller cannot know
    # whether the daemon committed the mutation before the transport failed.
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        path = str(Path(harness._fake.path).parent / "drop.sock")
        listener.bind(path)
        listener.listen(1)
        listener.settimeout(3)
        received: list[bytes] = []

        def drop_response() -> None:
            conn, _ = listener.accept()
            with conn:
                received.append(conn.recv(4096))
                if reply is not None:
                    conn.sendall(reply)

        thread = threading.Thread(target=drop_response, daemon=True)
        thread.start()
        monkeypatch.setattr(adapter.client, "socket_path", path)
        first = runtime.tools.call(pid, "agentvfs_checkpoint", {"label": "once"})
        thread.join(timeout=3)
    assert not first.ok
    assert received == [b"checkpoint once\n"]
    assert runtime.store.get_capability(cap.cap_id).uses_remaining == 0
    second = runtime.tools.call(pid, "agentvfs_checkpoint", {"label": "again"})
    assert not second.ok
    effects = [
        effect for effect in runtime.store.list_external_effects(pid=pid)
        if effect.provider == "agentvfs"
    ]
    assert len(effects) == 1
    assert effects[0].transaction_state == "unknown"
    assert effects[0].state_mutation is True
    assert effects[0].record_id and effects[0].event_id
    assert any(
        row.action == "module.agentvfs.checkpoint.failed"
        for row in runtime.audit.trace()
    )


def test_connection_failure_restores_finite_grant(
    agentvfs_runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, pid, harness = agentvfs_runtime
    cap = runtime.capability.issue_trusted(
        subject=pid,
        resource=f"agentvfs:{harness._workspace}",
        rights=["write"],
        issued_by="test",
        uses_remaining=1,
    )
    adapter = runtime.module_state.get("_agent_libos_agentvfs_adapter")
    socket_path = adapter.client.socket_path
    monkeypatch.setattr(adapter.client, "socket_path", socket_path + ".missing")
    failed = runtime.tools.call(pid, "agentvfs_checkpoint", {"label": "once"})
    assert not failed.ok
    assert runtime.store.get_capability(cap.cap_id).uses_remaining == 1
    assert harness._fake.received == []
    effects = [
        effect for effect in runtime.store.list_external_effects(pid=pid)
        if effect.provider == "agentvfs"
    ]
    assert effects == []
    monkeypatch.setattr(adapter.client, "socket_path", socket_path)
    retried = runtime.tools.call(pid, "agentvfs_checkpoint", {"label": "once"})
    assert retried.ok, retried.error
    assert runtime.store.get_capability(cap.cap_id).uses_remaining == 0
