"""Write results carry the digest a complete read returns for the same bytes."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
from typing import Any

from agent_libos import Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.memory.object_memory import _stub_result_summary
from agent_libos.models import AgentImage, CapabilityRight
from agent_libos.substrate import LocalResourceProviderSubstrate


def _open_runtime(tmp_path: Path) -> tuple[Runtime, str]:
    config = replace(
        DEFAULT_CONFIG,
        memory=replace(DEFAULT_CONFIG.memory, process_namespace_prefix="write-digest"),
    )
    runtime = Runtime.open(
        "local", config=config, substrate=LocalResourceProviderSubstrate(tmp_path)
    )
    runtime.register_image(
        AgentImage(
            image_id="write-digest:v0",
            name="write-digest",
            version="v0",
            default_tools=[
                "write_text_file",
                "read_text_file",
                "create_object_from_file",
                "write_object_to_file",
            ],
        ),
        actor="test.host",
    )
    pid = runtime.process.spawn(image="write-digest:v0", goal="Verify write digests")
    for path in ("notes.txt", "wide.txt", "export.txt"):
        runtime.filesystem.grant_path(
            pid, path, [CapabilityRight.READ, CapabilityRight.WRITE], issued_by="test.host"
        )
    return runtime, pid


def _dispatch(runtime: Runtime, pid: str, action: str, **args: Any) -> dict[str, Any]:
    return runtime.llm.dispatch(pid, {"action": action, **args})


def test_write_text_file_returns_digest_matching_disk_and_complete_read(tmp_path: Path) -> None:
    runtime, pid = _open_runtime(tmp_path)
    try:
        content = "line one\nline two\n"
        written = _dispatch(runtime, pid, "write_text_file", path="notes.txt", content=content)
        assert written["ok"], written
        digest = written["payload"]["content_sha256"]
        expected = hashlib.sha256(content.encode("utf-8")).hexdigest()
        assert digest == expected
        # The digest describes the bytes actually on disk, not a re-encoding.
        assert hashlib.sha256((tmp_path / "notes.txt").read_bytes()).hexdigest() == expected
        read = _dispatch(runtime, pid, "read_text_file", path="notes.txt")
        assert read["ok"], read
        assert read["payload"]["content_sha256"] == digest
        assert read["payload"]["content"] == content

        # The write digest is a valid compare-and-swap baseline for the next write.
        follow_up = _dispatch(
            runtime,
            pid,
            "write_text_file",
            path="notes.txt",
            content=content + "line three\n",
            expected_content_sha256=digest,
        )
        assert follow_up["ok"], follow_up
        assert follow_up["payload"]["content_sha256"] == hashlib.sha256(
            (content + "line three\n").encode("utf-8")
        ).hexdigest()
        # A stale write digest is rejected like any other stale precondition.
        stale = _dispatch(
            runtime,
            pid,
            "write_text_file",
            path="notes.txt",
            content="lost update\n",
            expected_content_sha256=digest,
        )
        assert not stale["ok"], stale
        assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == content + "line three\n"
    finally:
        runtime.close()


def test_write_digest_follows_the_requested_encoding(tmp_path: Path) -> None:
    runtime, pid = _open_runtime(tmp_path)
    try:
        content = "wide é中\n"
        result = runtime.filesystem.write_text(
            pid=pid, path="wide.txt", text=content, encoding="utf-16"
        )
        assert result.bytes_written == len(content.encode("utf-16"))
        assert result.content_sha256 == hashlib.sha256(content.encode("utf-16")).hexdigest()
        assert result.content_sha256 == hashlib.sha256(
            (tmp_path / "wide.txt").read_bytes()
        ).hexdigest()
        read = _dispatch(runtime, pid, "read_text_file", path="wide.txt", encoding="utf-16")
        assert read["ok"], read
        assert read["payload"]["content_sha256"] == result.content_sha256
        # The Object export path returns the same kind of digest for its destination.
        assert _dispatch(runtime, pid, "create_object_from_file", name="wide", path="wide.txt", encoding="utf-16")["ok"]
        exported = _dispatch(runtime, pid, "write_object_to_file", name="wide", path="export.txt")
        assert exported["ok"], exported
        assert exported["payload"]["content_sha256"] == hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()
    finally:
        runtime.close()


def test_write_digest_survives_tool_result_compaction() -> None:
    digest = "a" * 64
    summary = _stub_result_summary(
        {"path": "notes.txt", "bytes_written": 18, "created": True, "content_sha256": digest}
    )
    assert summary["content_sha256"] == digest
    assert summary["bytes_written"] == 18


# --- content echo -----------------------------------------------------------
#
# In the 2026-09-16 long-horizon runs the model kept reading files it had just
# written because the harness keeps tool results, not the model's own tool-call
# arguments: a digest of bytes it could no longer see was no evidence to it.
# The write result therefore echoes the stored text and takes the place of the
# stale read in the working set.

from agent_libos.memory.object_memory import _observation_supersession_key  # noqa: E402
from agent_libos.models import ObjectType  # noqa: E402


def _feedback(runtime: Runtime, pid: str, tool_name: str, result: dict[str, Any]):
    return runtime.memory.create_object(
        pid, ObjectType.TOOL_RESULT, {"tool_name": tool_name, "result": result}
    )


def _working_set(runtime: Runtime, pid: str, objects: list[Any]):
    goal = runtime.process.get(pid).memory_view.roots[0]
    view = runtime.memory.create_view(pid, [goal, *objects])
    return runtime.memory.materialize_context(
        pid, view, policy="working_set", budget_tokens=100_000, charge_resources=False
    )


def _read_result(path: str, content: str, *, encoding: str = "utf-8", truncated: bool = False) -> dict[str, Any]:
    return {
        "path": path,
        "content": content,
        "encoding": encoding,
        "bytes_read": len(content.encode(encoding)),
        "truncated": truncated,
        "content_sha256": None if truncated else hashlib.sha256(content.encode(encoding)).hexdigest(),
    }


def _write_result(path: str, content: str | None, *, encoding: str = "utf-8", created: bool = False) -> dict[str, Any]:
    stored = (content or "").encode(encoding)
    return {
        "path": path,
        "bytes_written": len(stored),
        "created": created,
        "content_sha256": hashlib.sha256(stored).hexdigest(),
        "content": content,
        "encoding": encoding,
    }


def test_write_text_file_echoes_stored_content_and_encoding(tmp_path: Path) -> None:
    runtime, pid = _open_runtime(tmp_path)
    try:
        content = "alpha\nbeta\n"
        written = _dispatch(runtime, pid, "write_text_file", path="notes.txt", content=content)
        assert written["ok"], written
        assert written["payload"]["content"] == content
        assert written["payload"]["encoding"] == "utf-8"
        wide = _dispatch(
            runtime, pid, "write_text_file", path="wide.txt", content="wide é中\n", encoding="utf-16"
        )
        assert wide["ok"], wide
        assert wide["payload"]["content"] == "wide é中\n"
        assert wide["payload"]["encoding"] == "utf-16"
    finally:
        runtime.close()


def test_write_echo_stops_at_the_default_complete_read_bound(tmp_path: Path) -> None:
    bound = DEFAULT_CONFIG.tools.filesystem_read_max_bytes
    runtime, pid = _open_runtime(tmp_path)
    try:
        fits = _dispatch(runtime, pid, "write_text_file", path="notes.txt", content="x" * bound)
        assert fits["ok"], fits
        assert fits["payload"]["bytes_written"] == bound
        assert fits["payload"]["content"] == "x" * bound
        over = _dispatch(runtime, pid, "write_text_file", path="wide.txt", content="x" * (bound + 1))
        assert over["ok"], over
        assert over["payload"]["bytes_written"] == bound + 1
        assert over["payload"]["content"] is None, "a file a default read could not return completely is not echoed"
        assert over["payload"]["content_sha256"] == hashlib.sha256(b"x" * (bound + 1)).hexdigest()
        assert over["payload"]["encoding"] == "utf-8"
    finally:
        runtime.close()


def test_content_echoing_write_supersedes_the_stale_read_of_its_path(tmp_path: Path) -> None:
    runtime, pid = _open_runtime(tmp_path)
    try:
        stale = _feedback(runtime, pid, "read_text_file", _read_result("notes.txt", "OLD_TEXT"))
        write = _feedback(runtime, pid, "write_text_file", _write_result("notes.txt", "WRITTEN_TEXT"))

        context = _working_set(runtime, pid, [stale, write])

        assert "WRITTEN_TEXT" in context.text
        assert "OLD_TEXT" not in context.text, "the write is the newest complete observation of the path"
        assert context.omitted_objects == [stale.oid]
        reasons = {entry["oid"]: entry["reason"] for entry in context.object_manifest}
        assert reasons[stale.oid] == "superseded"

        # A later complete read is fresher than the echo and replaces it in turn.
        fresh = _feedback(runtime, pid, "read_text_file", _read_result("notes.txt", "SECOND_READ_TEXT"))
        context = _working_set(runtime, pid, [stale, write, fresh])

        assert "SECOND_READ_TEXT" in context.text
        assert "WRITTEN_TEXT" not in context.text
        assert set(context.omitted_objects) == {stale.oid, write.oid}
        reasons = {entry["oid"]: entry["reason"] for entry in context.object_manifest}
        assert reasons[write.oid] == "superseded"
        assert "omitted_object_guidance" not in context.text
    finally:
        runtime.close()


def test_writes_without_an_echo_or_with_another_encoding_keep_the_read(tmp_path: Path) -> None:
    runtime, pid = _open_runtime(tmp_path)
    try:
        read = _feedback(runtime, pid, "read_text_file", _read_result("notes.txt", "KEEP_READ_TEXT"))
        unechoed = _feedback(runtime, pid, "write_text_file", _write_result("notes.txt", None))
        other_encoding = _feedback(
            runtime, pid, "write_text_file", _write_result("notes.txt", "UTF16_TEXT", encoding="utf-16")
        )

        context = _working_set(runtime, pid, [read, unechoed, other_encoding])

        assert "KEEP_READ_TEXT" in context.text, "an unechoed write is an action, not an observation"
        assert "UTF16_TEXT" in context.text
        assert context.omitted_objects == []
    finally:
        runtime.close()


def test_write_supersession_key_matches_only_a_complete_read_of_the_same_path() -> None:
    read_key = _observation_supersession_key(
        {"tool_name": "read_text_file", "result": _read_result("src/a.py", "x")}
    )
    write_key = _observation_supersession_key(
        {"tool_name": "write_text_file", "result": _write_result("src/a.py", "y")}
    )
    assert read_key is not None
    assert write_key == read_key
    truncated_key = _observation_supersession_key(
        {"tool_name": "read_text_file", "result": {**_read_result("src/a.py", "x", truncated=True), "bytes_read": 1}}
    )
    assert truncated_key != write_key
    assert _observation_supersession_key(
        {"tool_name": "write_text_file", "result": _write_result("src/a.py", None)}
    ) is None
    assert _observation_supersession_key(
        {"tool_name": "write_text_file", "ok": False, "result": _write_result("src/a.py", "y")}
    ) is None
    assert _observation_supersession_key(
        {"tool_name": "write_text_file", "result": _write_result("src/b.py", "y")}
    ) != write_key


def test_compacted_write_stub_keeps_the_digest_and_only_the_size_of_the_echo() -> None:
    summary = _stub_result_summary(_write_result("notes.txt", "WRITTEN_TEXT", created=True))
    assert summary["content_sha256"] == hashlib.sha256(b"WRITTEN_TEXT").hexdigest()
    assert summary["encoding"] == "utf-8"
    assert summary["created"] is True
    assert summary["content_chars"] == len("WRITTEN_TEXT")
    assert "content" not in summary
