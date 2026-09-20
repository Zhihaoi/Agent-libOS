from __future__ import annotations

import json
import socket
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_libos.models.exceptions import ValidationError
from modules.agentvfs import agentvfs_module as agentvfs


class HostStub:
    def __init__(self, binding: Any) -> None:
        self.substrate = SimpleNamespace(agentvfs=binding)
        self.state: dict[str, Any] = {}

    def get_runtime_attribute(self, name: str) -> Any:
        return self.state.get(name)

    def set_runtime_attribute(self, name: str, value: Any) -> None:
        self.state[name] = value


def test_unbound_module_initializes_without_unix_sockets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(socket, "AF_UNIX", raising=False)
    host = HostStub(None)
    agentvfs.initialize_agentvfs(host)
    assert host.state[agentvfs._ADAPTER_ATTR] == agentvfs._UNBOUND


def test_bound_module_rejects_missing_unix_sockets_before_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(socket, "AF_UNIX", raising=False)
    host = HostStub("workspace")
    with pytest.raises(ValidationError, match="requires Unix-domain socket support"):
        agentvfs.initialize_agentvfs(host)
    assert host.state == {}


@pytest.mark.parametrize("workspace", [".", "..", "../other", "", None, 7])
def test_workspace_binding_rejects_invalid_path_components(workspace: Any) -> None:
    with pytest.raises(ValidationError):
        agentvfs._coerce_binding({"workspace": workspace})


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True, "30"])
def test_binding_requires_finite_positive_timeout(timeout: Any) -> None:
    with pytest.raises(ValidationError, match="finite and positive"):
        agentvfs._coerce_binding({"workspace": "test", "request_timeout_s": timeout})


@pytest.mark.parametrize("record", [[], None, {"status": "started", "socket": 42}])
def test_discovery_rejects_malformed_session_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, record: Any
) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    session_dir = tmp_path / "agentvfs" / "test"
    session_dir.mkdir(parents=True)
    (session_dir / "session.json").write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValidationError):
        agentvfs._discover_socket(agentvfs.AgentVfsBinding(workspace="test"))
